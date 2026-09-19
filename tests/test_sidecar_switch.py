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
