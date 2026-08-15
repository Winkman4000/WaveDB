"""wdb_shelves -- the birth ledger (Jackson's eager-shelf law).

Shelves are few and small, but birth-on-touch made them lucky accidents of
query order: a consumer lane that only OPENS a shelf ran slow forever if the
birthing lane's query shape never fired. The cure: every birth RECORDS its
recipe into shelves.json beside the segment; Database.open REPLAYS the
ledger, birthing anything missing -- so a successor file inherits the whole
shelf dynasty at launch and no query ever waits on a trigger it can't send.
"""
import json
import os

_KINDS = ('gbc2', 'gdc2', 'gdc', 'tier2')


def _ledger_path(seg):
    return os.path.join(os.path.dirname(seg.path), 'shelves.json')


def record(seg, kind, **args):
    """Append a birth recipe (deduped, atomic, quiet on failure)."""
    import controller
    controller.plan_epoch_bump()
    try:
        p = _ledger_path(seg)
        rows = []
        if os.path.exists(p):
            rows = json.load(open(p))
        ent = {'kind': kind, 'args': args}
        if ent in rows:
            return
        rows.append(ent)
        tmp = p + '.tmp'
        json.dump(rows, open(tmp, 'w'))
        os.replace(tmp, p)
    except Exception:
        pass


def replay(db):
    """Birth every recorded shelf that is missing for the current segments."""
    if os.environ.get('WDB_EAGER_SHELVES', '1') == '0':
        return
    try:
        tables = list(db.cat.list_tables())
    except Exception:
        return
    for t in tables:
        try:
            paths = db.cat.segment_paths(t)
        except Exception:
            continue
        for sp in paths:
            try:
                seg = db.open_segment(sp, t)
            except Exception:
                continue
            p = _ledger_path(seg)
            if not os.path.exists(p):
                continue
            try:
                rows = json.load(open(p))
            except Exception:
                continue
            for ent in rows:
                kind = ent.get('kind')
                a = ent.get('args', {})
                try:
                    if kind == 'gbc2':
                        import wdb_gbshelf
                        if wdb_gbshelf.open_shelf(seg, a['col']) is None:
                            wdb_gbshelf.birth(seg, a['col'])
                    elif kind == 'gbc3':
                        import wdb_gbshelf
                        if wdb_gbshelf.open_dense(seg, a['col']) is None:
                            wdb_gbshelf.birth_dense(seg, a['col'])
                    elif kind == 'gdc2':
                        import wdb_groupdistinct as GD
                        if GD._gdc_load(seg, a['k'], a['t']) is None:
                            GD.birth2(seg, a['k'], a['t'])
                    elif kind == 'plist':
                        import wdb_funnel
                        wdb_funnel._plist(seg, a['col'])
                    elif kind == 'tier2':
                        import wdb_pairdistinct as PD
                        PD._tier_shelf(seg, a['a'], a['b'], a['u'])
                    elif kind == 'gdc':
                        import wdb_groupdistinct as GD
                        GD.birth_gdc(seg, a['k'], a['t'])
                except Exception:
                    pass
