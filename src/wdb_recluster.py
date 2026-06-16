"""
wdb_recluster -- reorder an existing segment by a key WITHOUT re-encoding from source.

A clustered segment is just the same rows in key order + a tiny .cluster sidecar. The
dictionaries are invariant under a row permutation (same distinct values -> same codes),
so reclustering does NOT need to read the source, factorize/sort the strings, or rebuild
the dictionaries. It only needs to permute each column's per-row CODES into key order and
re-serialise -- reusing each dict verbatim. This is the cheap, low-memory mutation path
(one column resident at a time), and the building block for in-place segment maintenance.

  dict columns (mode 0/1, incl. all large strings): valb = seg.dict_vals(nm) reused as-is,
      codes = seg._raw_codes(nm)[order], re-serialised. No dict rebuild.
  other columns (mode 2/3/4/5/6): decode -> permute -> _prep_column (cheap; numeric/positional).

Output is byte-for-byte a normal WaveDB segment + a .cluster sidecar, identical in content
to encode(cluster_by=key) but produced from the existing segment instead of the source.
"""
import os, struct, pickle
import numpy as np
import wdb_encode as ENC
from wdb_engine import Segment


def _col_values_masked(seg, nm):
    """Decode a column to a (masked, if has_null) array for the re-encode fallback path."""
    c = seg.cols[nm]
    vals = seg.values(nm)
    if not c['has_null']:
        return vals
    if vals.dtype.kind == 'f':
        mask = np.isnan(vals.astype(float, copy=False))
    else:
        mask = np.array([v is None for v in vals], dtype=bool)
    return np.ma.masked_array(vals, mask=mask)


def recluster(seg_path, key, out_path, verbose=False):
    """Write a key-clustered copy of the segment at seg_path to out_path (+ .cluster sidecar)."""
    seg = Segment(seg_path)
    N = int(seg.N)
    cols = list(seg.cols)
    if key not in seg.cols:
        raise KeyError("cluster key %r not in segment columns %s" % (key, cols))

    order, cluster_meta = ENC._cluster_order(_col_values_masked(seg, key), N)
    cluster_meta['key'] = key
    order = np.asarray(order)

    zc = ENC.zstd.ZstdCompressor(level=ENC.ZSTD_LEVEL)
    out = bytearray(b'WVDB4') + struct.pack('<H', len(cols)) + struct.pack('<I', N)
    for nm in cols:
        c = seg.cols[nm]; mode = c['mode']
        if mode in (0, 1):                                  # dict column: reuse valb, permute codes
            valb = list(seg.dict_vals(nm))
            codes = seg._raw_codes(nm)[order].astype(np.uint64)
            p = dict(nm=nm, dtype=c['dt'], has_null=c['has_null'], V=c['V'], valb=valb,
                     codes=codes, aux=c['aux'], bits=c['bits'], mode=mode, uniq=None)
            blob, _ = ENC._serialize_column(p, zc)
        else:                                               # numeric/positional: decode -> permute -> re-encode
            vals = _col_values_masked(seg, nm)[order]
            blob, _ = ENC._serialize_column(ENC._prep_column(nm, vals), zc)
        out += blob
        if verbose:
            print("  %-20s mode=%d done" % (nm, mode), flush=True)

    open(out_path, 'wb').write(out)
    with open(out_path + '.cluster', 'wb') as f:
        pickle.dump(cluster_meta, f, protocol=4)
    return {'rows': N, 'cols': len(cols), 'bytes': len(out), 'key': key,
            'n_key_values': len(cluster_meta['values'])}


if __name__ == '__main__':
    import sys, time
    if len(sys.argv) < 4:
        print("usage: wdb_recluster.py <in.wdb> <key> <out.wdb>"); sys.exit(1)
    t = time.time()
    r = recluster(sys.argv[1], sys.argv[2], sys.argv[3], verbose=True)
    print("RECLUSTER DONE in %.0fs: %s" % (time.time() - t, r), flush=True)
