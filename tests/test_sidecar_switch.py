"""THE SWITCH (Jackson's operator law): sidecars are an extension the operator turns on. With the
switch off a database answers every query from its segments and the RAM shelf and writes NOTHING
derived to disk -- the number on disk is the number that was loaded. Same answers either way.
THE SENTINEL (strict in the suite) raises on any file born while off: a missed gate is a defect."""
import sys, os, tempfile, uuid, json, subprocess
from decimal import Decimal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb, wdb_encode, wdb_sidecar
from wdb_db import Database

_DB = None; _CON = None; _DIR = None


def _fixture():
    global _DB, _CON, _DIR
    if _DB is not None: return _DB, _CON, _DIR
    _CON = duckdb.connect()
    d = os.path.join(tempfile.gettempdir(), f'sidesw_{uuid.uuid4().hex[:8]}'); os.makedirs(d, exist_ok=True)
    _CON.execute("CREATE TABLE dim AS SELECT i AS did, 'L'||(i%7) AS label, 'City'||(i%23) AS city, CAST(i*1.5 AS DOUBLE) AS w FROM range(2000) t(i)")
    _CON.execute("CREATE TABLE fact AS SELECT i AS fid, (i*7)%2000 AS dimid, CAST(((i%50)+1)*10.0 AS DOUBLE) AS amt, "
                 "'user'||(i%997) AS who, 'https://site'||(i%31)||'.example/'||(i%5) AS url, (i%13) AS k FROM range(60000) t(i)")
    _DB = Database.create(d); _DIR = d
    wt = {'BIGINT': 'int', 'INTEGER': 'int', 'VARCHAR': 'string', 'DOUBLE': 'float'}
    for tbl, order in (('dim', 'did'), ('fact', 'fid')):
        desc = _CON.execute(f"DESCRIBE {tbl}").fetchall()
        pq = os.path.join(d, f'{tbl}.parquet')
        _CON.execute(f"COPY (SELECT * FROM {tbl} ORDER BY {order}) TO '{pq}' (FORMAT parquet)")
        _DB.cat.add_table(tbl, [[c[0], wt[c[1]]] for c in desc])
        seg = f'{tbl}_0.wdb'; wdb_encode.encode(pq, os.path.join(d, seg)); _DB.cat.add_segment(tbl, seg)
        os.remove(pq)
    _DIR = d
    return _DB, _CON, _DIR


_CORPUS = [
    "SELECT COUNT(*) FROM fact f JOIN dim d ON f.dimid = d.did WHERE d.label = 'L3'",
    "SELECT d.label, COUNT(*) FROM fact f JOIN dim d ON f.dimid = d.did GROUP BY d.label",
    "SELECT d.city, SUM(f.amt) FROM fact f JOIN dim d ON f.dimid = d.did GROUP BY d.city ORDER BY 2 DESC LIMIT 5",
    "SELECT MIN(d.label), MAX(d.city) FROM fact f JOIN dim d ON f.dimid = d.did WHERE f.k = 4",
    "SELECT who, COUNT(*) FROM fact GROUP BY who ORDER BY 2 DESC, 1 LIMIT 10",
    "SELECT k, COUNT(DISTINCT who) FROM fact GROUP BY k ORDER BY k",
    "SELECT COUNT(*) FROM fact WHERE url LIKE '%site7.%'",
    "SELECT COUNT(*) FROM fact WHERE lower(who) = 'user5'",
    "SELECT k, COUNT(*), SUM(amt) FROM fact WHERE amt > 100 GROUP BY k ORDER BY k",
    "SELECT url, COUNT(*) FROM fact GROUP BY url ORDER BY 2 DESC, 1 LIMIT 3",
    "SELECT who, k, COUNT(*) FROM fact GROUP BY who, k ORDER BY 3 DESC, 1, 2 LIMIT 7",
    "SELECT COUNT(DISTINCT dimid) FROM fact WHERE k < 5",
    "SELECT MIN(amt), MAX(amt), AVG(amt) FROM fact WHERE who LIKE 'user1%'",
    # THE VANILLA LAW's shapes: with the switch off these must answer from the streaming reads
    "SELECT who FROM fact WHERE who = 'user5'",                     # unordered rows, no LIMIT
    "SELECT AVG(dimid), SUM(k), COUNT(*), MIN(dimid), MAX(k) FROM fact",   # the exact scalar, no block stats
    "SELECT COUNT(*) FROM fact WHERE who = 'user7'",                # scalar count, no census
    "SELECT dimid, COUNT(*) FROM fact GROUP BY dimid ORDER BY 2 DESC, 1 LIMIT 5",   # top-k by count
]


def _norm(rows):
    out = []
    for r in rows:
        out.append(tuple(round(float(c), 3) if isinstance(c, (float, Decimal)) else
                         (c if isinstance(c, int) and not isinstance(c, bool) else str(c)) for c in r))
    return sorted(out)


def _derived(d):
    return sorted(n for n in os.listdir(d) if not wdb_sidecar.is_data_file(n))


class _NoEnv:
    """the suite sets WDB_SIDECARS=1 so births happen everywhere; the switch tests want the CATALOG to decide"""
    def __enter__(self):
        self.saved = os.environ.pop('WDB_SIDECARS', None); return self
    def __exit__(self, *a):
        if self.saved is not None: os.environ['WDB_SIDECARS'] = self.saved


def _run_corpus(db, con):
    got = {}
    for q in _CORPUS:
        g = _norm(db.run(q)[0]); e = _norm([tuple(r) for r in con.execute(q).fetchall()])
        assert g == e, f"mismatch vs duck under sidecars={wdb_sidecar.setting(db.cat.dbdir)}: {q}\n got {g[:3]}\n exp {e[:3]}"
        got[q] = g
    return got


def test_new_database_is_born_off():
    db, con, d = _fixture()
    with _NoEnv():
        assert json.load(open(os.path.join(d, 'catalog.json'))).get('sidecars') == 'off'
        assert wdb_sidecar.setting(d) == 'off' and not wdb_sidecar.births_on(d)


def test_off_births_nothing_and_answers_the_same():
    db, con, d = _fixture()
    with _NoEnv():
        wdb_sidecar.set_setting(d, 'off')
        before = _derived(d)
        off = _run_corpus(db, con)                      # strict sentinel: any newborn raises inside run()
        assert _derived(d) == before, 'files born while off: %s' % (set(_derived(d)) - set(before))
        wdb_sidecar.set_setting(d, 'on')
        db2 = Database.open(d)
        on = _run_corpus(db2, con)
        assert on == off
        born = set(_derived(d)) - set(before)
        assert born, 'the corpus births nothing even when on -- the test proves nothing'
        for n in born:
            assert wdb_sidecar.classify(n) is not None, 'a derived file the registry cannot name: %s' % n


def test_drop_returns_the_loaded_database():
    db, con, d = _fixture()
    with _NoEnv():
        wdb_sidecar.set_setting(d, 'on')
        _run_corpus(Database.open(d), con)
        assert _derived(d), 'nothing to drop'
        gone, freed = wdb_sidecar.drop(d, print_out=False)
        assert gone > 0 and freed > 0
        assert _derived(d) == [], _derived(d)
        assert wdb_sidecar.setting(d) == 'off'
        assert sorted(n for n in os.listdir(d) if n.endswith('.wdb')) == ['dim_0.wdb', 'fact_0.wdb']
        _run_corpus(Database.open(d), con)             # still answers, still births nothing
        assert _derived(d) == []


def test_environment_overrides_the_catalog():
    db, con, d = _fixture()
    with _NoEnv():
        wdb_sidecar.set_setting(d, 'off')
        os.environ['WDB_SIDECARS'] = '1'
        try:
            assert wdb_sidecar.births_on(d)
            _run_corpus(Database.open(d), con)
            assert _derived(d), 'WDB_SIDECARS=1 did not license births'
        finally:
            del os.environ['WDB_SIDECARS']
        wdb_sidecar.drop(d, print_out=False)
        os.environ['WDB_SIDECARS'] = '0'
        try:
            wdb_sidecar.set_setting(d, 'on')
            assert not wdb_sidecar.births_on(d)
            _run_corpus(Database.open(d), con)
            assert _derived(d) == []
        finally:
            del os.environ['WDB_SIDECARS']
        wdb_sidecar.set_setting(d, 'off')


def test_cli_status_on_off_drop():
    db, con, d = _fixture()
    with _NoEnv():
        wdb = os.path.join(os.path.dirname(__file__), '..', 'bin', 'wdb')
        env = dict(os.environ); env['PYTHONPATH'] = os.path.join(os.path.dirname(__file__), '..', 'src')
        def cli(*args):
            r = subprocess.run([sys.executable, wdb, 'sidecars', d] + list(args), capture_output=True, text=True, env=env)
            assert r.returncode == 0, r.stderr
            return r.stdout
        assert 'sidecars: on' in cli('on')
        assert json.load(open(os.path.join(d, 'catalog.json')))['sidecars'] == 'on'
        assert 'sidecars: on' in cli('status')
        assert 'sidecars: off' in cli('off')
        assert 'sidecars: off' in cli('status')
        out = cli('drop')
        assert 'DROP' in out and wdb_sidecar.setting(d) == 'off'



def test_load_statistics_are_data():
    """THE STATISTICS OF THE LOAD (B): the encoder writes block statistics beside the segment as
    DATA -- present with the switch off, kept by drop, and the block-stats read answers from them
    exactly, under strict sentinel, with nothing born."""
    import wdb_blockstats, numpy as np
    db, con, d = _fixture()
    with _NoEnv():
        wdb_sidecar.set_setting(d, 'off')
        sp = wdb_blockstats.stats_path(os.path.join(d, 'fact_0.wdb'))
        assert os.path.exists(sp), 'the encoder did not write the load statistics'
        assert wdb_sidecar.is_data_file(os.path.basename(sp))
        z = np.load(sp, allow_pickle=False)
        assert int(z['N']) == 60000 and 'dimid.sum' in z.files and 'k.cmax' in z.files
        # THE EXCEPTION LIST: fid is unique (V == N) -> a differentiator with zero exception rows, when it
        # is a dictionary column (under WDB_SEQ_NARROW_OK the suite stores the sequence as mode 4: no list)
        import wdb_engine
        fmode = wdb_engine.Segment(os.path.join(d, 'fact_0.wdb')).cols['fid'].get('mode')
        if fmode in (0, 1, 2):
            assert 'fid.rep' in z.files and z['fid.rep'].size == 0, ('the unique column has no exception list', sorted(z.files))
        else:
            assert 'fid.rep' not in z.files, ('a non-dictionary column has no exception list', fmode)
        assert 'k.rep' not in z.files, 'a 13-value column is no differentiator'
        before = _derived(d)
        h0 = wdb_blockstats._HITS
        for q in ("SELECT AVG(dimid), SUM(k), COUNT(*), MIN(dimid), MAX(k) FROM fact",
                  "SELECT SUM(dimid) FROM fact", "SELECT COUNT(k) FROM fact"):
            g = _norm(db.run(q)[0]); e = _norm([tuple(r) for r in con.execute(q).fetchall()])
            assert g == e, (q, g, e)
        assert wdb_blockstats._HITS > h0, 'the block-stats read did not serve from the load statistics'
        assert _derived(d) == before
        gone, freed = wdb_sidecar.drop(d, print_out=False)
        assert os.path.exists(sp), 'drop removed the load statistics: they are data, not a sidecar'
        # the kernel agrees with a straight computation, block by block
        seg = db.open_segment(db.cat.segment_paths('fact')[0], 'fact')
        st = wdb_blockstats.compute(seg, 'k')
        vals = np.asarray(seg.values('k'), dtype=np.int64)
        BR = wdb_blockstats._BR
        for j in range(st['cnt'].size):
            v = vals[j * BR:(j + 1) * BR]
            assert st['cnt'][j] == v.size and st['nn'][j] == v.size
            assert abs(st['sum'][j] - float(v.sum())) < 1e-6, j


def test_vanilla_positions_build_no_lists():
    """THE VANILLA LAW, the position lists: with the switch off the funnel's positions come from
    a scan of the blocks whose load-time min/max can hold the code -- exact, windowed, and nothing
    built: _plist refuses (FAIL-LOUD), the frame-presence map declines, no memo is left behind."""
    import numpy as np, wdb_funnel, wdb_fpm
    from wdb_engine import Segment
    db, con, d = _fixture()
    with _NoEnv():
        wdb_sidecar.set_setting(d, 'off')
        sp = os.path.join(d, 'fact_0.wdb')
        for f in os.listdir(d):
            if f.endswith('.plist') or f.endswith('.fpm'):
                os.remove(os.path.join(d, f))
        seg = Segment(sp)
        for col in ('dimid', 'k'):                     # (dimid may be a mode-4 sequence; k a dictionary)
            codes = np.asarray(seg._raw_codes(col)).astype(np.int64)
            for code in (int(codes[12345]), int(codes[0]), int(codes[-1])):
                assert np.array_equal(wdb_funnel.positions(seg, col, code), np.flatnonzero(codes == code)), (col, code)
                exp = np.flatnonzero(codes == code); exp = exp[(exp >= 1000) & (exp < 40000)]
                assert np.array_equal(wdb_funnel.positions(seg, col, code, 1000, 40000), exp), (col, code)
        codes = np.asarray(seg._raw_codes('dimid')).astype(np.int64)
        assert not wdb_funnel.plist_ready(seg, 'dimid')
        try:
            wdb_funnel._plist(seg, 'dimid'); raise RuntimeError('the plist was built under vanilla')
        except AssertionError:
            pass
        assert wdb_fpm.eq_positions(seg, 'dimid', int(codes[7])) is None
        assert not seg.__dict__.get('_plistmemo')
        assert _derived(d) == [] or all(not f.endswith(('.plist', '.fpm')) for f in _derived(d))


def test_vanilla_sample_draws_exact_counts():
    """THE VANILLA DRAW: a LIMIT with no ORDER BY over (int, sparse string) samples from the sparse
    dress's stored rows -- no position lists -- and every returned count is the exact count."""
    import numpy as np, pandas as pd, wdb_sampletop
    con = duckdb.connect()
    d = os.path.join(tempfile.gettempdir(), f'sidesw_st_{uuid.uuid4().hex[:8]}'); os.makedirs(d)
    rng = np.random.default_rng(17); n = 200_000
    ph = rng.integers(0, 3000, n)
    df = pd.DataFrame({'u': rng.integers(0, 5000, n).astype(np.int64),
                       'sp': np.where(rng.random(n) < 0.15, np.char.add('phrase-', ph.astype(str)), '')})
    con.register('t', df)
    pq = os.path.join(d, 't.parquet'); df.to_parquet(pq, index=False)
    with _NoEnv():
        db = Database.create(d)
        db.cat.add_table('t', [['u', 'int'], ['sp', 'string']])
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb')); db.cat.add_segment('t', 't_0.wdb'); os.remove(pq)
        wdb_sidecar.set_setting(d, 'off')
        db = Database.open(d)
        from wdb_engine import Segment
        assert Segment(os.path.join(d, 't_0.wdb')).cols['sp']['code_enc'] in (8, 9)
        h0 = wdb_sampletop._HITS
        rows = db.run("SELECT u, sp, COUNT(*) FROM t GROUP BY u, sp LIMIT 10")[0]
        assert wdb_sampletop._HITS > h0, 'the sample lane did not serve'
        assert len(rows) == 10
        for u, s, n in rows:
            exp = con.execute("SELECT COUNT(*) FROM t WHERE u = ? AND sp = ?", [int(u), str(s)]).fetchone()[0]
            assert int(n) == exp, (u, s, n, exp)
        assert _derived(d) == []


def test_lists_positions_exact():
    """positions() FROM THE POSITION LISTS (switch on): exact against the codes, whole-table and
    windowed (the vanilla test covers the scan; this one the lists)"""
    import numpy as np, wdb_funnel
    from wdb_engine import Segment
    db, con, d = _fixture()
    with _NoEnv():
        wdb_sidecar.set_setting(d, 'on')
        seg = Segment(os.path.join(d, 'fact_0.wdb'))
        codes = np.asarray(seg._raw_codes('k')).astype(np.int64)
        assert wdb_funnel.plist_ready(seg, 'k')
        for code in (int(codes[12345]), int(codes[0]), int(codes[-1])):
            exp = np.flatnonzero(codes == code)
            assert np.array_equal(wdb_funnel.positions(seg, 'k', code), exp), code
            ew = exp[(exp >= 1000) & (exp < 40000)]
            assert np.array_equal(wdb_funnel.positions(seg, 'k', code, 1000, 40000), ew), code
        assert os.path.exists(wdb_funnel._plist_path(seg, 'k')), 'positions did not use the lists'


def test_lists_sample_draws_exact_counts():
    """THE LISTS DRAW (switch on): a LIMIT with no ORDER BY over (int, sparse string) draws group
    keys from the position lists -- each draw's count from two neighbouring offsets, no census --
    and every returned count is the exact count; the pluck (one values_at batch) returns the
    strings themselves"""
    import numpy as np, pandas as pd, wdb_sampletop, wdb_funnel
    con = duckdb.connect()
    d = os.path.join(tempfile.gettempdir(), f'sidesw_st_{uuid.uuid4().hex[:8]}'); os.makedirs(d)
    rng = np.random.default_rng(19); n = 200_000
    ph = rng.integers(0, 3000, n)
    df = pd.DataFrame({'u': rng.integers(0, 5000, n).astype(np.int64),
                       'sp': np.where(rng.random(n) < 0.15, np.char.add('phrase-', ph.astype(str)), '')})
    con.register('t', df)
    pq = os.path.join(d, 't.parquet'); df.to_parquet(pq, index=False)
    with _NoEnv():
        db = Database.create(d)
        db.cat.add_table('t', [['u', 'int'], ['sp', 'string']])
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb')); db.cat.add_segment('t', 't_0.wdb'); os.remove(pq)
        wdb_sidecar.set_setting(d, 'on')
        db = Database.open(d)
        from wdb_engine import Segment
        seg = Segment(os.path.join(d, 't_0.wdb'))
        assert seg.cols['sp']['code_enc'] in (8, 9)
        assert wdb_funnel.plist_ready(seg, 'sp')
        for rep in range(3):
            h0 = wdb_sampletop._HITS
            rows = db.run("SELECT u, sp, COUNT(*) FROM t GROUP BY u, sp LIMIT 10")[0]
            assert wdb_sampletop._HITS > h0, 'the sample lane did not serve'
            assert len(rows) == 10
            assert len({(int(u), s) for u, s, n in rows}) == 10
            for u, s, n in rows:
                assert isinstance(s, str), (type(s), s)
                exp = con.execute("SELECT COUNT(*) FROM t WHERE u = ? AND sp = ?", [int(u), str(s)]).fetchone()[0]
                assert int(n) == exp, (u, s, n, exp)
        assert os.path.exists(os.path.join(d, 't_0.wdb.sp.plist')), 'the draws did not use the lists'
