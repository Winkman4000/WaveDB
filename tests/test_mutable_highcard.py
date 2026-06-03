"""Mutable layer x high-cardinality sequential-key columns -- the seam mode-4 plugs in
UNDER. Drives a buffered table with a sequential id (+ attributes) through
insert -> flush (multiple cold segments) -> DELETE -> UPDATE (literal & expression) ->
compact -> reopen, applying IDENTICAL DML to a DuckDB table and comparing after every phase.
When mode-4 lands, this proves the override/presence layers and compaction still behave
correctly on the exact column type mode-4 encodes."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, duckdb
from wdb_db import Database

def _dir(): return os.path.join(tempfile.gettempdir(), f'mhc_{uuid.uuid4().hex[:8]}')
def _norm(rows):
    return sorted([tuple(int(x) if isinstance(x,(np.integer,)) else x for x in r) for r in rows],
                  key=lambda t: tuple(str(x) for x in t))

def _pair(d):
    db = Database.create(d)
    db.run("CREATE TABLE t (id INT, grp VARCHAR, val INT)")
    db.set_table_mode('t', 'buffered')
    con = duckdb.connect(); con.execute("CREATE TABLE t (id INTEGER, grp VARCHAR, val INTEGER)")
    return db, con

def _both(db, con, sql):
    db.run(sql); con.execute(sql)

def _ins(db, con, rows):
    vals = ",".join(f"({i},'{g}',{v})" for i,g,v in rows)
    _both(db, con, f"INSERT INTO t VALUES {vals}")

def _cmp(db, con, sel="SELECT id,grp,val FROM t"):
    w = _norm(db.run(sel)[0]); dk = _norm(con.execute(sel).fetchall())
    assert w == dk, f"\n SQL: {sel}\n wdb : {w[:6]}\n duck: {dk[:6]}\n (len {len(w)} vs {len(dk)})"

def _seq_batches(n_batches=3, per=40, base=1_000_000):
    grps = ['alpha','beta','gamma','delta']; k = 0; out = []
    for _ in range(n_batches):
        b = []
        for _ in range(per):
            b.append((base + k, grps[k % 4], (k * 7) % 500)); k += 1
        out.append(b)
    return out

def test_seqkey_delete_across_segments_then_compact():
    d = _dir(); db, con = _pair(d)
    for b in _seq_batches(): _ins(db, con, b); db.flush('t')          # 3 cold segments
    _cmp(db, con)
    _both(db, con, "DELETE FROM t WHERE val < 100")                    # spans all segments
    _cmp(db, con)
    _cmp(db, con, "SELECT grp, COUNT(*), SUM(val) FROM t GROUP BY grp")
    db.compact('t')                                                    # physical reclamation
    _cmp(db, con); _cmp(db, con, "SELECT MIN(id), MAX(id), COUNT(*) FROM t")
    shutil.rmtree(d)

def test_seqkey_update_literal_and_expr_then_compact():
    d = _dir(); db, con = _pair(d)
    for b in _seq_batches(): _ins(db, con, b); db.flush('t')
    _both(db, con, "UPDATE t SET val = 999 WHERE grp = 'alpha'")       # literal, spans segments
    _cmp(db, con)
    _both(db, con, "UPDATE t SET val = val + 1 WHERE grp = 'beta'")    # expression
    _cmp(db, con)
    _cmp(db, con, "SELECT grp, SUM(val) FROM t GROUP BY grp")
    db.compact('t')                                                    # folds overrides into dict
    _cmp(db, con); _cmp(db, con, "SELECT grp, SUM(val), COUNT(*) FROM t GROUP BY grp")
    shutil.rmtree(d)

def test_seqkey_update_the_key_column():
    # updating the sequential key itself (mode-4 would store this as an override/exception)
    d = _dir(); db, con = _pair(d)
    for b in _seq_batches(2): _ins(db, con, b); db.flush('t')
    _both(db, con, "UPDATE t SET id = id + 5000000 WHERE grp = 'gamma'")
    _cmp(db, con); _cmp(db, con, "SELECT id FROM t WHERE id > 5000000 ORDER BY id ASC LIMIT 10")
    db.compact('t'); _cmp(db, con)
    shutil.rmtree(d)

def test_seqkey_full_lifecycle_reopen():
    d = _dir(); db, con = _pair(d)
    for b in _seq_batches(4, per=30): _ins(db, con, b); db.flush('t')  # 4 segments
    _both(db, con, "INSERT INTO t VALUES (2000001,'alpha',12),(2000002,'beta',13)")  # hot buffer
    _both(db, con, "DELETE FROM t WHERE id BETWEEN 1000010 AND 1000030")
    _both(db, con, "UPDATE t SET val = val + 100 WHERE val < 50")
    _both(db, con, "DELETE FROM t WHERE grp = 'delta'")
    _cmp(db, con)
    db.compact('t'); _cmp(db, con)
    db2 = Database.open(d)                                             # reopen: state persisted
    w = _norm(db2.run("SELECT id,grp,val FROM t")[0]); dk = _norm(con.execute("SELECT id,grp,val FROM t").fetchall())
    assert w == dk, f"reopen mismatch: {len(w)} vs {len(dk)} rows"
    shutil.rmtree(d)

def test_seqkey_delete_all_then_query():
    d = _dir(); db, con = _pair(d)
    for b in _seq_batches(2): _ins(db, con, b); db.flush('t')
    _both(db, con, "DELETE FROM t WHERE id >= 1000000")                # everything
    _cmp(db, con); _cmp(db, con, "SELECT COUNT(*) FROM t")
    db.compact('t'); _cmp(db, con)
    shutil.rmtree(d)
