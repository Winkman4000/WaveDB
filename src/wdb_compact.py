"""Compaction: merge several cold segments into one, paying down the append debt that
buffered flushes run up. This is also where FD labels get VERIFIED.

Stage 2a (this module): merge segments' rows into one new segment (normal encoding),
and along the way VERIFY each candidate label on the merged union — keep survivors,
drop accidental casualties. Verify is ~20x cheaper than rediscovery (measured), which is
the whole point of labeling at flush. The hot buffer (if any) is left untouched; only
cold segments are merged. Query results are identical before and after (verified by tests).

Stage 2b (later): use a verified X->Y to store column Y as references into X's pool.
"""
import os
import pandas as pd, numpy as np
from wdb_engine import Segment
import wdb_encode, wdb_labels, wdb_presence, wdb_override

_PD = {'int': 'Int64', 'float': 'float64', 'string': 'object', 'datetime': 'datetime64[ns]'}

def _choose_fd_specs(kept):
    """From verified labels, pick {dependent: determinant} to encode as mode-3 references.
    Chain-free: a column that is a determinant for any FD stays a normal column, and each
    dependent gets at most one determinant. This guarantees every mode-3 column references
    a normally-stored (readable) determinant in the same segment."""
    dets = {l['det'] for l in kept}
    specs = {}
    for l in sorted(kept, key=lambda l: -l.get('det_uniqueness', 0.0)):
        dep, det = l['dep'], l['det']
        if dep == det or dep in specs or dep in dets:
            continue
        specs[dep] = det
    return specs

def _segment_to_df(seg, schema, phys=None, defaults=None):
    """Reconstruct a segment's LIVE rows as a DataFrame keyed by PHYSICAL column name, using the
    LOGICAL schema for the column set + dtypes. Tombstoned rows (presence sidecar) are dropped here
    -- this is where deleted space is physically reclaimed. A column the segment PREDATES (ADD
    COLUMN) is materialized from its default; a renamed column resolves through phys; a dropped
    column is simply absent from the schema and so reclaimed."""
    phys = phys or {}; defaults = defaults or {}
    pm = seg.presence_mask()           # bool[N] True=live, or None if all live
    liveN = int(pm.sum()) if pm is not None else seg.N
    data = {}
    for c, wt in [(x[0], x[1]) for x in schema]:
        pcol = phys.get(c, c)
        if pcol in seg.cols:
            vals = seg.values(pcol)
            if pm is not None: vals = vals[pm]
            vals = list(vals)
            if seg.cols.get(pcol, {}).get('dt') == 1:
                # inline / front-coded strings come back as bytes or bytearray: the encoder hashes str
                vals = [(v.decode('utf-8', 'replace') if isinstance(v, (bytes, bytearray)) else v) for v in vals]
        else:
            vals = [defaults.get(c)] * liveN      # ADD COLUMN: default materialized at compaction
        if wt == 'datetime':
            data[pcol] = pd.to_datetime(pd.Series(vals), errors='coerce')
        else:
            data[pcol] = pd.array(vals, dtype=_PD.get(wt, 'object'))
    return pd.DataFrame(data)

def compact(catalog, name, seg_files=None):
    """Merge the given cold segments (default: all) into one new segment, verifying labels.
    Returns dict(merged=[...], new_segment=..., rows=N, labels_in=k, labels_kept=k)."""
    tinfo = catalog.get_table(name)
    schema = tinfo['schema']
    all_segs = list(tinfo['segments'])
    targets = seg_files if seg_files is not None else all_segs
    targets = [s for s in targets if s in all_segs]
    def _dirty(sf):
        p = os.path.join(catalog.dbdir, sf)
        return os.path.exists(wdb_presence.path_for(p)) or os.path.exists(wdb_override.path_for(p))
    if len(targets) < 2 and not any(_dirty(sf) for sf in targets):
        return {'merged': [], 'new_segment': None, 'rows': 0, 'labels_in': 0, 'labels_kept': 0}
    # a SINGLE segment carrying tombstones or overrides is rewritten clean: that is how the
    # fast doors come back after DML (THE PRESENCE GATE serves it slowly until then)

    # 1) decode + union the target segments
    dfs = []
    candidate_labels = {}   # (det,dep) -> det_uniqueness (best/any)
    import wdb_dml
    for sf in targets:
        seg = Segment(os.path.join(catalog.dbdir, sf))
        wdb_dml.register_synth(catalog, seg, name)   # synth ADD-COLUMN defaults so overrides on
        dfs.append(_segment_to_df(seg, schema,       # them (and the default itself) materialize
                                  catalog.phys_map(name), tinfo.get('defaults', {})))
        for lab in catalog.segment_labels(name, sf):
            candidate_labels[(lab['det'], lab['dep'])] = lab.get('det_uniqueness', 0.0)
    union = pd.concat(dfs, ignore_index=True)

    # 2) VERIFY each candidate label on the union — keep survivors only
    kept = []
    for (det, dep), dux in candidate_labels.items():
        if wdb_labels.holds(union, det, dep):
            kept.append({'det': det, 'dep': dep, 'det_uniqueness': dux})

    # 3) re-encode the union into one new segment, storing verified FDs as mode-3 references
    from wdb_dml import _next_segment_index
    idx = _next_segment_index(catalog, name)
    new_seg = f"{name}_{idx}.wdb"
    tmp_pq = os.path.join(catalog.dbdir, f"{name}__compact_{idx}.parquet")
    union.to_parquet(tmp_pq, index=False)
    fd_specs = _choose_fd_specs(kept)
    wdb_encode.encode(tmp_pq, os.path.join(catalog.dbdir, new_seg), fd_specs=fd_specs)
    os.remove(tmp_pq)

    # 4) update catalog: drop merged segments (list + labels + disk), add the new one
    from wdb_encode import _crash_point
    _crash_point('compact:segment-written')          # new segment exists; catalog still points at the old ones
    tinfo['segments'] = [s for s in tinfo['segments'] if s not in targets] + [new_seg]
    fl = tinfo.setdefault('fd_labels', {})
    for s in targets:
        fl.pop(s, None)
        p = os.path.join(catalog.dbdir, s)
        if os.path.exists(p): os.remove(p)
        sc = wdb_presence.path_for(p)               # drop the now-stale presence sidecar
        if os.path.exists(sc): os.remove(sc)
        oc = wdb_override.path_for(p)                # drop the now-stale override sidecar
        if os.path.exists(oc): os.remove(oc)
        # EVERY derived artefact of the removed segment goes with it (roads, censuses, shelves,
        # reverse roads...): dictionary codes and row positions are reborn with the segment
        for fn in os.listdir(catalog.dbdir):
            if fn.startswith(s + '.') and fn != s:
                try: os.remove(os.path.join(catalog.dbdir, fn))
                except OSError: pass
    fl[new_seg] = kept
    catalog.save()
    _crash_point('compact:catalog-saved')            # catalog points at the new segment; old files still on disk
    return {'merged': targets, 'new_segment': new_seg, 'rows': len(union),
            'labels_in': len(candidate_labels), 'labels_kept': len(kept),
            'fd_encoded': len(fd_specs)}
