"""TEXT FUNCTIONS ON THE DICTIONARY STREAM (2026-10-05): LENGTH, LIKE (with '_'), the prefix, SUBSTR and LOWER judged
once per distinct value by compiled kernels on the dictionary's byte stream. The kernels equal Python (ASCII,
Cyrillic of both cases, other scripts, emoji, malformed bytes flagged); through SQL every function, in filters and
in GROUP BY, answers as DuckDB does -- NULLs included."""
import sys, os, uuid, tempfile, shutil, subprocess, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd

TMP = tempfile.gettempdir()
WDB = os.path.join(os.path.dirname(__file__), '..', 'bin', 'wdb')
FLOOR = dict(WDB_LOAD_ANSWERS='0', WDB_SIDECARS='0', WDB_QMEM_STRICT='1')
ALPHA = list('abcxyzABCXYZ_%. ') + list('абвгдежзийклмнопрстуфхцчшщъыьэюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯёЁѐЀ') \
    + list('αβΓΔéÉüÜßİĞ') + ['😀', '中']


def _words(rng, n, maxlen=12):
    return [''.join(rng.choice(ALPHA, rng.integers(0, maxlen + 1))) for _ in range(n)]


def _stream(strs):
    bs = [s.encode() if isinstance(s, str) else s for s in strs]
    off = np.zeros(len(bs) + 1, np.int64); np.cumsum([len(b) for b in bs], out=off[1:])
    return np.frombuffer(b''.join(bs) or b'\x00', np.uint8).copy(), off


def _like_ref(s, pat):
    rx = ''.join('.*' if c == '%' else ('.' if c == '_' else re.escape(c)) for c in pat)
    return re.fullmatch(rx, s, re.DOTALL) is not None


def test_kernels_equal_python():
    import wdb_kernels as K, wdb_sql
    rng = np.random.default_rng(5)
    words = _words(rng, 3000)
    blob, off = _stream(words)
    out = np.empty(len(words), np.int64); K.pstr_charlen(blob, off, out)
    assert out.tolist() == [len(w) for w in words]
    pats = ['%', '', 'a%', '%a', '%a%', '_', '__', '_б%', '%в_', 'А%я', '%а%б%', '_%_', 'a_c', '%_X%', '😀%',
            '%中', '_%😀', 'Ё%', '%ѐ_%', 'ab', '%%', 'a%%b', '%_%_%_%']
    for p in pats:
        got = wdb_sql._like_stream(blob, off, p)
        assert got.tolist() == [_like_ref(w, p) for w in words], p
    for skip, take in ((0, 3), (0, 0), (0, -1), (2, 4), (5, 1)):
        ss = np.empty(len(words), np.int64); se = np.empty(len(words), np.int64); bad = np.empty(len(words), np.bool_)
        K.psubstr_bounds(blob, off, skip, take, ss, se, bad)
        assert not bad.any()
        got = [blob[a:b].tobytes().decode() for a, b in zip(ss, se)]
        assert got == [w[skip:] if take < 0 else w[skip:skip + take] for w in words], (skip, take)
    low = np.empty_like(blob); fl = np.empty(len(words), np.bool_)
    K.plower_ru(blob, off, low, fl)
    for i, w in enumerate(words):
        if not fl[i]:
            assert low[off[i]:off[i + 1]].tobytes().decode() == w.lower(), w
        else:
            assert any(ord(c) > 0x7F and not (0x400 <= ord(c) <= 0x45F) for c in w), w
    gid = np.empty(len(words), np.int64); rep = np.empty(len(words), np.int64)
    G = K.str_dedupe(blob, off, gid, rep)
    first = {}
    for w in words: first.setdefault(w, len(first))
    assert G == len(first) and gid.tolist() == [first[w] for w in words]
    bads = [b'ok', b'\xff', b'\xd0', b'a\xd0\x90', b'\xe0\x80\x80', b'\xed\xa0\x80', b'\xf4\x90\x80\x80', b'\xc3\xa9']
    b2, o2 = _stream(bads)
    ss = np.empty(len(bads), np.int64); se = np.empty(len(bads), np.int64); bad = np.empty(len(bads), np.bool_)
    K.psubstr_bounds(b2, o2, 0, 2, ss, se, bad)
    def strict(b):
        try: b.decode('utf-8'); return True
        except UnicodeDecodeError: return False
    assert (~bad).tolist() == [strict(b) for b in bads]


def test_text_functions_through_sql_equal_duck():
    import duckdb
    from wdb_db import Database
    rng = np.random.default_rng(9)
    n = 200_000
    vocab = _words(rng, 20_000, 16)
    t = rng.choice(vocab, n)
    u = np.array(rng.choice(vocab[:3000], n), dtype=object)
    u[rng.random(n) < 0.05] = None                                     # a nullable text column
    df = pd.DataFrame({'id': np.arange(n, dtype=np.int64), 't': t, 'u': u, 'g': rng.integers(0, 50, n)})
    sqls = [
        "SELECT COUNT(*) FROM s WHERE LENGTH(t) > 10",
        "SELECT COUNT(*) FROM s WHERE LENGTH(t) = 3",
        "SELECT COUNT(*) FROM s WHERE LENGTH(u) = 0",
        "SELECT COUNT(*) FROM s WHERE LENGTH(u) <> 4",
        "SELECT LENGTH(t) AS n, COUNT(*) AS c FROM s GROUP BY n ORDER BY c DESC, n LIMIT 8",
        "SELECT LENGTH(t) AS n, COUNT(*) AS c FROM s WHERE g = 7 GROUP BY n ORDER BY c DESC, n LIMIT 8",
        "SELECT COUNT(*) FROM s WHERE t LIKE '_б%'",
        "SELECT COUNT(*) FROM s WHERE t LIKE '%а%б%'",
        "SELECT COUNT(*) FROM s WHERE t LIKE 'А%'",
        "SELECT COUNT(*) FROM s WHERE t LIKE '%_%_%_%'",
        "SELECT COUNT(*) FROM s WHERE t NOT LIKE '_a%'",
        "SELECT COUNT(*) FROM s WHERE u LIKE '__'",
        "SELECT COUNT(*) FROM s WHERE u LIKE '%'",
        "SELECT COUNT(*) FROM s WHERE t LIKE 'ab%'",
        "SELECT SUBSTR(t, 1, 2) AS p, COUNT(*) AS c FROM s GROUP BY p ORDER BY c DESC, p LIMIT 10",
        "SELECT SUBSTR(t, 1, 1) AS p, COUNT(*) AS c FROM s WHERE g < 10 GROUP BY p ORDER BY c DESC, p LIMIT 10",
        "SELECT LOWER(t) AS l, COUNT(*) AS c FROM s GROUP BY l ORDER BY c DESC, l LIMIT 10",
        "SELECT LOWER(t) AS l, COUNT(*) AS c FROM s WHERE t <> '' GROUP BY l ORDER BY c DESC, l LIMIT 10",
    ]
    d = os.path.join(TMP, 'sf_' + uuid.uuid4().hex[:8]); db_dir = d + '_db'; os.makedirs(d)
    old = {k: os.environ.get(k) for k in FLOOR}
    import wdb_sql, wdb_scalar, wdb_lmap, wdb_kernels
    served = {}
    def _count(mod, name, served_if=lambda r: True):
        orig = getattr(mod, name)
        def f(*a, **k):
            r = orig(*a, **k)
            if served_if(r): served[name] = served.get(name, 0) + 1
            return r
        setattr(mod, name, f)
        return mod, name, orig
    patches = [_count(wdb_kernels, 'pstr_charlen'), _count(wdb_sql, '_like_stream', lambda r: r is not None),
               _count(wdb_scalar, '_surrogate_prefix', lambda r: r is not None),
               _count(wdb_lmap, '_build_stream', lambda r: r is not None)]
    try:
        pq = os.path.join(d, 's.parquet'); df.to_parquet(pq, index=False)
        subprocess.run([sys.executable, WDB, 'load', db_dir, 's', pq], check=True, capture_output=True,
                       env=dict(os.environ, **FLOOR))
        con = duckdb.connect(); con.execute("CREATE VIEW s AS SELECT * FROM read_parquet('%s')" % pq)
        os.environ.update(FLOOR)
        db = Database.open(db_dir)
        for sql in sqls:
            r = db.run(sql); got = [tuple(x) for x in (r[0] if isinstance(r, tuple) else r)]
            want = [tuple(x) for x in con.execute(sql).fetchall()]
            assert [tuple(str(v) for v in g) for g in got] == [tuple(str(v) for v in w) for w in want], (sql, got[:5], want[:5])
        print('    served by the stream:', served)
        assert served.get('pstr_charlen') and served.get('_like_stream') and served.get('_build_stream'), served
        # the prefix surrogate directly (the toy's GROUP BY SUBSTR takes another road): sorted distinct prefixes,
        # every code mapped to its own
        seg = db.open_segment(db.cat.segment_paths('s')[0], 's')
        pc = db.cat.phys_map('s').get('t', 't')
        assert seg.cols[pc].get('mode') in (0, 1)
        dv = [v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in seg._typed_dict(pc)]
        for ln in (0, 1, 2, 5, None):
            got = wdb_scalar._surrogate_prefix(seg, {'col': pc, 'fn': 'substr', 'arg': (1, ln)})
            assert got is not None, ln
            inv, sv = got
            ref = [s if ln is None else s[:ln] for s in dv]
            uniq = sorted(set(ref))
            assert list(sv) == uniq and [uniq[i] for i in inv[:len(ref)]] == ref, ln
    finally:
        for mod, name, orig in patches:
            setattr(mod, name, orig)
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
        shutil.rmtree(d, ignore_errors=True); shutil.rmtree(db_dir, ignore_errors=True)
