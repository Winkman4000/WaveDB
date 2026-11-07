"""pairdistinct widened (2026-11-07): GROUP BY a, b with COUNT(DISTINCT u), WHERE b <> <any literal of b's kind>,
ORDER BY the distinct count DESC LIMIT k. The old literal '', the stored sparse default (the planes are the filter),
a value that exists, a literal the column never holds (every row), columns with NULLs (never kept), a column with no
'', an int b -- each served by pairdistinct and equal to DuckDB (ties legal at the cut), in RAM and through the
saved shelf; a shelf born for one literal is never read for another; a literal of the other kind declines."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

Q = "SELECT %s, %s, COUNT(DISTINCT u) AS du FROM t WHERE %s <> %s GROUP BY %s, %s ORDER BY du DESC LIMIT %d"
FORMS = [('k', 's', "''"), ('k', 's', "'x'"), ('k', 's', "'v7'"), ('k', 's', "'zz'"),
         ('k', 'nn', "''"), ('k', 'nn', "'m2'"), ('k', 'nn', "'zz'"),
         ('k', 'g', "'g2'"), ('k', 'g', "'zz'"), ('c', 'g', "'g0'"),
         ('k', 'n', "''"), ('k', 'n', "'n3'"), ('f', 'n', "'n3'"),
         ('c', 'k', '1'), ('c', 'k', '7'), ('g', 'k', '2'), ('nn', 'f', '0')]


def _q(a, b, lit, lim):
    return Q % (a, b, b, lit, a, b, lim)


def _serve(db, q):
    import controller, wdb_pairusers as PU
    pu0 = PU.ON[0]; PU.ON[0] = False             # pairusers runs ahead of pairdistinct and would take these shapes
    try:
        got = db.run(q)
    finally:
        PU.ON[0] = pu0
    return got, controller._SERVED[0]


def test_every_literal_served_and_duckdbs():
    import test_sparse_default as T
    db, con = T._fixture()
    bad = []
    for rnd in range(2):                         # the second round reads the shelves the first one saved
        for a, b, lit in FORMS:
            for lim in (1, 5, 20):
                q = _q(a, b, lit, lim)
                got, who = _serve(db, q)
                if who != 'pairdistinct':
                    bad.append(('not served', who, q))
                elif not T._legal(q, T._rows(got), con):
                    bad.append(('wrong', rnd, q, T._rows(got)[:4], T._rows(con.execute(q).fetchall())[:4]))
    assert not bad, bad


def test_shelf_per_literal_and_ram_path():
    """with the narrow read off every form goes through the shelf: one file per literal; equal with births off (RAM)"""
    import test_sparse_default as T, wdb_pairdistinct as PD
    db, con = T._fixture()
    import wdb_join
    seg, _x = wdb_join._solo_segment(db, 't')
    old = os.environ.get('WDB_SIDECARS')
    PD._NARROW[0] = False
    try:
        for sw in ('on', 'off'):
            os.environ['WDB_SIDECARS'] = sw
            for rnd in range(2):                 # on: the first round births each file, the second reads it back
                for b, lit in (('s', "'x'"), ('s', "'v7'"), ('s', "''"), ('s', "'zz'"), ('nn', "''"), ('nn', "'m2'"),
                               ('g', "'g2'")):
                    q = _q('k', b, lit, 5)
                    got, who = _serve(db, q)
                    assert who == 'pairdistinct' and T._legal(q, T._rows(got), con), (sw, rnd, q, who)
            if sw == 'on':
                names = [f for f in os.listdir(os.path.dirname(seg.path)) if f.startswith('tier2__k__s__u__')]
                assert len(names) == 4, names    # each literal its own file
    finally:
        PD._NARROW[0] = True
        if old is None: os.environ.pop('WDB_SIDECARS', None)
        else: os.environ['WDB_SIDECARS'] = old


def test_planes_serve_the_default():
    """s's stored default is 'x': `s <> 'x'` is answered from the planes (the narrow read), never a full pass"""
    import test_sparse_default as T, wdb_pairdistinct as PD, wdb_join
    db, con = T._fixture()
    seg, _x = wdb_join._solo_segment(db, 't')
    if seg.cols['s'].get('code_enc') not in (8, 9):
        return
    seen = []
    real = PD._narrow
    PD._narrow = lambda *a9: (lambda r: seen.append(r is not None) or r)(real(*a9))
    try:
        got, who = _serve(db, _q('k', 's', "'x'", 5))
        assert who == 'pairdistinct' and seen == [True], (who, seen)
        seen.clear()
        got, who = _serve(db, _q('k', 's', "'v7'", 5))
        assert who == 'pairdistinct' and seen == [False], (who, seen)   # not the default: the shelf
    finally:
        PD._narrow = real


def test_other_kind_declines():
    import test_sparse_default as T
    db, con = T._fixture()
    for a, b, lit in (('k', 's', '3'), ('c', 'k', "'1'"), ('c', 'k', '1.5')):
        q = _q(a, b, lit, 5)
        try:
            got, who = _serve(db, q)
        except Exception:
            continue
        assert who != 'pairdistinct', q


if __name__ == '__main__':
    for nm, fn in list(globals().items()):
        if nm.startswith('test_'):
            fn(); print('PASS', nm)
