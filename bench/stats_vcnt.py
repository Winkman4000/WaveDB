"""THE CENSUS OF THE LOAD, added to an existing database: the rows-per-code of every dictionary
column of at most VCNT_MAX codes, written into the load statistics beside each segment (the file
the encoder writes as its last step; a fresh load writes the counts itself). Every other array in
the file is kept byte for byte. Prints the time it took -- the load time this adds.

Usage: PYTHONPATH=src python bench/stats_vcnt.py DB_DIR
"""
import sys, os, glob, time
import numpy as np
import wdb_engine, wdb_blockstats as BS

if __name__ == '__main__':
    for path in sorted(glob.glob(os.path.join(sys.argv[1], '*.wdb'))):
        p = BS.stats_path(path)
        z = np.load(p, allow_pickle=False)
        out = {k: z[k] for k in z.files}
        seg = wdb_engine.Segment(path)
        t = time.perf_counter(); n = 0
        for col in seg.order:
            vc = BS.value_counts(seg, col)
            if vc is not None:
                out[col + '.vcnt'] = vc; n += 1
            seg._codes.pop(col, None)
        dt = time.perf_counter() - t
        tmp = p + '.partial.npz'
        np.savez(tmp, **out)
        os.replace(tmp, p)
        print('%s: %d columns counted in %.1f s, stats file %.1f KB' % (os.path.basename(path), n, dt, os.path.getsize(p) / 1024))
        z2 = np.load(p, allow_pickle=False)
        for k in z.files:
            assert np.array_equal(z[k], z2[k]), ('stats array changed', k)
