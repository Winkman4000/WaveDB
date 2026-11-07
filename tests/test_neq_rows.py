"""wdb_wherescan.neq_rows (2026-11-07): the rows of `col <> lit` for any literal -- the planes when the stored sparse
default IS the literal, else one pass over the codes; NULL rows never; a literal of the other kind declines."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))


def test_every_literal_keeps_duckdbs_rows():
    import numpy as np, sqlglot
    import test_sparse_default as T, wdb_join, wdb_wherescan as WS
    db, con = T._fixture()
    seg, _x = wdb_join._solo_segment(db, 't')
    for col, lit in (('s', "'x'"), ('s', "''"), ('s', "'v7'"), ('s', "'zz'"), ('n', "'n3'"), ('n', "''"),
                     ('nn', "''"), ('nn', "'m2'"), ('g', "'g2'"), ('g', "'zz'"), ('k', '1'), ('c', '62'), ('k', '7')):
        node = sqlglot.parse_one('SELECT 1 FROM t WHERE %s <> %s' % (col, lit)).args['where'].this.expression
        r = WS.neq_rows(seg, col, node)
        assert r is not None, (col, lit)
        pos, codes = r
        want = np.asarray([x[0] for x in con.execute('SELECT id FROM t WHERE %s <> %s ORDER BY id' % (col, lit)).fetchall()])
        assert np.array_equal(pos, want), (col, lit, pos.size, want.size)
        assert np.array_equal(codes, np.asarray(seg._raw_codes(col))[pos]), (col, lit)
    for col, lit in (('s', '3'), ('k', "'1'")):
        node = sqlglot.parse_one('SELECT 1 FROM t WHERE %s <> %s' % (col, lit)).args['where'].this.expression
        assert WS.neq_rows(seg, col, node) is None, (col, lit)


if __name__ == '__main__':
    test_every_literal_keeps_duckdbs_rows(); print('PASS')
