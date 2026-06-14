"""Regression: SQL LIKE must match across embedded newlines.

The LIKE->regex conversion previously omitted re.DOTALL, so '.' would not span a
newline and a value containing '\r\n' (or '\n') before the matched substring was
silently NOT matched. On ClickBench Q20 (URL LIKE '%google%') this dropped 3 rows
whose URL held a newline before 'google'. All LIKE paths -- the wdb_sql per-row
matcher and the fused wdb_join code-LUT / pandas _mask -- must use DOTALL so that
SQL '%' correctly matches any character including newlines.
"""
import sys, os, tempfile, uuid, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from wdb_db import Database

ROWS = [
    (0, "http://site/google/search"),         # plain match
    (1, "http://host/path\r\nmore/google/x"),  # newline BEFORE google -> the bug case
    (2, "http://other/yandex/q"),              # no match
    (3, "http://up/GOOGLE/q"),                 # capital -> LIKE is case-sensitive -> no match
    (4, "http://a\nb\ngoogle"),                # multiple newlines then google -> must match
]

def _db():
    d = os.path.join(tempfile.gettempdir(), f"likenl_{uuid.uuid4().hex[:8]}")
    db = Database.create(d)
    db.run("CREATE TABLE t (id INT, url VARCHAR)")
    db.set_table_mode("t", "buffered")
    for i, u in ROWS:
        db.run("INSERT INTO t VALUES (" + str(i) + ", '" + u + "')")
    db.flush("t")                              # cold segment -> exercises the fused LIKE path
    return db, d

def test_like_matches_across_newline_default_and_escalate():
    db, d = _db()
    try:
        q = "SELECT COUNT(*) FROM t WHERE url LIKE '%google%'"
        assert db.run(q)[0][0][0] == 3, "default path must count newline-spanning matches"
        assert db.run(q, escalate=True)[0][0][0] == 3, "fused path must count newline-spanning matches"
        ids = sorted(r[0] for r in db.run("SELECT id FROM t WHERE url LIKE '%google%'")[0])
        assert ids == [0, 1, 4], ids
    finally:
        shutil.rmtree(d, ignore_errors=True)

def test_like_case_sensitive_still_excludes_capital():
    db, d = _db()
    try:
        n = db.run("SELECT COUNT(*) FROM t WHERE url LIKE '%GOOGLE%'")[0][0][0]
        assert n == 1, n                       # DOTALL must not relax case sensitivity
    finally:
        shutil.rmtree(d, ignore_errors=True)
