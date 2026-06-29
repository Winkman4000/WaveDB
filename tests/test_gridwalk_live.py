"""wdb_gridwalk_live: buffered-mode 2-key COUNT(*) top-K (one cold segment + a growing hot buffer)
served by the gridwalk base + GwMaint, and ALWAYS equal to the generic wdb_merge path and to a
brute-force pandas ground truth.

Master property: for every scenario, the live answer == wdb_merge answer == pandas group-by
(tie-aware: counts and the unambiguous above-boundary identity). On top of that we assert the routing
contract -- the fast path actually engages for clean in-dictionary inserts, and correctly DECLINES
(falling back to wdb_merge) when the hot buffer adds a value new to the segment dictionary."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
from wdb_db import Database
import wdb_encode, wdb_gridwalk_live as LIVE


def _build_buffered(d, seed=5, N=300, nk=8, nn=6):
    rng = np.random.default_rng(seed)
    base = pd.DataFrame({'s': rng.choice([f'k{i}' for i in range(nk)], N),
                         'n': rng.integers(0, nn, N).astype('int64')})
    db = Database.create(d); db.run("CREATE TABLE t (s VARCHAR, n BIGINT)")
    pq = os.path.join(d, 'base.parquet'); base.to_parquet(pq, index=False)
    wdb_encode.encode(pq, os.path.join(d, 't_0.wdb')); db.cat.add_segment('t', 't_0.wdb')
    db.set_table_mode('t', 'buffered')
    return db, base


def _rows(r):
    return list(r[0] if isinstance(r, tuple) else r)


def _counts(rr):
    return sorted(int(x[2]) for x in rr)


def _ident_above(rr):
    if not rr:
        return set()
    bnd = min(int(x[2]) for x in rr)
    return set((str(x[0]), int(x[1])) for x in rr if int(x[2]) > bnd)


def _pandas_top(df, lim):
    g = df.groupby(['s', 'n']).size().reset_index(name='c')
    g = g[g['c'] >= 2].sort_values('c', ascending=False)
    return _counts([(r.s, r.n, r.c) for r in g.head(lim).itertuples()])


def _live_eq_merge_eq_pandas(db, df_all, q, lim):
    h0 = LIVE._HITS
    live = _rows(db.run(q)); hit = LIVE._HITS > h0
    LIVE.disable()
    try:
        merge = _rows(db.run(q))
    finally:
        LIVE.enable()
    assert _counts(live) == _counts(merge), (q, _counts(live), _counts(merge))
    assert _ident_above(live) == _ident_above(merge), q
    assert _counts(live) == _pandas_top(df_all, lim), (q, _counts(live), _pandas_top(df_all, lim))
    return hit


def test_live_accelerates_and_matches():
    d = os.path.join(tempfile.gettempdir(), f'lv_{uuid.uuid4().hex[:8]}')
    try:
        db, base = _build_buffered(d)
        hot = pd.DataFrame({'s': ['k1', 'k1', 'k2', 'k3', 'k1', 'k2', 'k0'],
                            'n': [2, 2, 3, 3, 2, 3, 1]})
        all_df = base.copy()
        for s, n in zip(hot['s'], hot['n']):
            db.run(f"INSERT INTO t VALUES ('{s}', {n})")
            all_df = pd.concat([all_df, pd.DataFrame({'s': [s], 'n': [n]})], ignore_index=True)
        any_hit = False
        for lim in (3, 5, 10):
            q = f"SELECT s, n, COUNT(*) FROM t GROUP BY s, n ORDER BY COUNT(*) DESC LIMIT {lim}"
            any_hit |= _live_eq_merge_eq_pandas(db, all_df, q, lim)
        assert any_hit, "clean in-dictionary buffered top-K should engage the live fast path"
    finally:
        LIVE.enable(); shutil.rmtree(d, ignore_errors=True)


def test_live_unordered_matches():
    d = os.path.join(tempfile.gettempdir(), f'lvu_{uuid.uuid4().hex[:8]}')
    try:
        db, base = _build_buffered(d)
        all_df = base.copy()
        for s, n in [('k1', 2), ('k1', 2), ('k4', 5), ('k4', 5)]:
            db.run(f"INSERT INTO t VALUES ('{s}', {n})")
            all_df = pd.concat([all_df, pd.DataFrame({'s': [s], 'n': [n]})], ignore_index=True)
        gt = all_df.groupby(['s', 'n']).size()
        gt_map = {(str(k[0]), int(k[1])): int(v) for k, v in gt.items()}
        n_heavy = int((gt.values >= 2).sum())
        lim = 5
        q = "SELECT s, n, COUNT(*) FROM t GROUP BY s, n LIMIT 5"          # unordered
        h0 = LIVE._HITS
        live = _rows(db.run(q)); hit = LIVE._HITS > h0
        # An unordered LIMIT is non-deterministic by SQL semantics -- merge returns an ARBITRARY set
        # of groups (including count-1), the fast path returns heavy cells by count; both are valid and
        # need not be the same set. The live contract is only: it engaged, returned exactly LIMIT
        # distinct REAL groups, each heavy (>=2) and carrying its TRUE full-data count.
        assert hit, "unordered buffered top-K should engage the live fast path"
        assert len(live) == min(lim, n_heavy), (len(live), n_heavy)
        seen = set()
        for s, n, c in [(str(x[0]), int(x[1]), int(x[2])) for x in live]:
            assert (s, n) not in seen, ("dup group", s, n); seen.add((s, n))
            assert c >= 2, ("not heavy", s, n, c)
            assert gt_map.get((s, n)) == c, ("wrong count", s, n, c, gt_map.get((s, n)))
    finally:
        LIVE.enable(); shutil.rmtree(d, ignore_errors=True)


def test_live_declines_new_dictionary_value():
    d = os.path.join(tempfile.gettempdir(), f'lvn_{uuid.uuid4().hex[:8]}')
    try:
        db, base = _build_buffered(d)
        all_df = base.copy()
        for s, n in [('k1', 2), ('BRANDNEW', 9)]:                        # BRANDNEW not in base dict
            db.run(f"INSERT INTO t VALUES ('{s}', {n})")
            all_df = pd.concat([all_df, pd.DataFrame({'s': [s], 'n': [n]})], ignore_index=True)
        q = "SELECT s, n, COUNT(*) FROM t GROUP BY s, n ORDER BY COUNT(*) DESC LIMIT 5"
        h0 = LIVE._HITS
        live = _rows(db.run(q)); declined = LIVE._HITS == h0
        assert declined, "a hot value new to the dictionary must decline to wdb_merge"
        LIVE.disable()
        try:
            merge = _rows(db.run(q))
        finally:
            LIVE.enable()
        assert _counts(live) == _counts(merge)                           # merge answered it, correctly
    finally:
        LIVE.enable(); shutil.rmtree(d, ignore_errors=True)


def test_live_declines_unsupported_shapes():
    d = os.path.join(tempfile.gettempdir(), f'lvs_{uuid.uuid4().hex[:8]}')
    try:
        db, base = _build_buffered(d)
        for s, n in [('k1', 2), ('k1', 2)]:
            db.run(f"INSERT INTO t VALUES ('{s}', {n})")
        # WHERE and single-key are not the gridwalk shape -> live declines, merge still correct
        for q in ["SELECT s, n, COUNT(*) FROM t WHERE n > 0 GROUP BY s, n ORDER BY COUNT(*) DESC LIMIT 3",
                  "SELECT s, COUNT(*) FROM t GROUP BY s ORDER BY COUNT(*) DESC LIMIT 3"]:
            h0 = LIVE._HITS
            r1 = _rows(db.run(q)); assert LIVE._HITS == h0, f"should decline: {q}"
            LIVE.disable()
            try:
                r2 = _rows(db.run(q))
            finally:
                LIVE.enable()
            assert [tuple(x) for x in r1] == [tuple(x) for x in r2], q
    finally:
        LIVE.enable(); shutil.rmtree(d, ignore_errors=True)
