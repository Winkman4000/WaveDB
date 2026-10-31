"""THE CODES ARE THE POINTER (2026-10-31): wdb_join._codes_pointer -- when lineitem's l_orderkey dictionary is
orders' o_orderkey column in row order, lineitem's codes equal the pointer _direct_pointer builds from the keys,
row for row; read-only; declined for a key whose dictionary is only a subset of the parent's (o_custkey), with the
switch off, and with the sidecar switch on; _hash_pointer gives the same pointer both ways."""
import sys, os, json, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import test_join_fast as J


@contextlib.contextmanager
def _off():
    old = os.environ.get('WDB_SIDECARS')
    os.environ['WDB_SIDECARS'] = '0'
    try:
        yield
    finally:
        if old is None: os.environ.pop('WDB_SIDECARS', None)
        else: os.environ['WDB_SIDECARS'] = old


def _segs(db, c, p):
    return (db.open_segment(db.cat.segment_paths(c)[0], c), db.open_segment(db.cat.segment_paths(p)[0], p))


def _mark(cs, ps, ck, pk):
    return {'c': [cs.path, 1, 1], 'p': [ps.path, 1, 1], 'ck': ck, 'pk': pk}


def test_codes_equal_the_built_pointer():
    import wdb_join as WJ
    db, _con = J._fixture()
    cs, ps = _segs(db, 'lineitem', 'orders')
    with _off():
        side = cs.path + '.l_orderkey__orders.o_orderkey.test'
        WJ._JPTR_CODES.pop(side, None)
        got = WJ._codes_pointer(cs, 'l_orderkey', ps, 'o_orderkey', side, _mark(cs, ps, 'l_orderkey', 'o_orderkey'))
        assert got is not None, 'the fixture is the case: its dictionary is the parent keys'
        built = WJ._direct_pointer(WJ._key_values(cs, 'l_orderkey'), WJ._key_values(ps, 'o_orderkey'))
        assert built is not None and built is not False
        assert np.array_equal(np.asarray(got, np.int64), np.asarray(built, np.int64))
        assert not got.flags.writeable
        old = WJ._CODES_PTR[0]
        try:
            WJ._CODES_PTR[0] = False
            assert WJ._codes_pointer(cs, 'l_orderkey', ps, 'o_orderkey', side, _mark(cs, ps, 'l_orderkey', 'o_orderkey')) is None
        finally:
            WJ._CODES_PTR[0] = old
    if __import__('wdb_sidecar').births_on(os.path.dirname(cs.path)):
        assert WJ._codes_pointer(cs, 'l_orderkey', ps, 'o_orderkey', side, _mark(cs, ps, 'l_orderkey', 'o_orderkey')) is None


def test_a_subset_dictionary_is_not_the_pointer():
    import wdb_join as WJ
    db, _con = J._fixture()
    cs, ps = _segs(db, 'orders', 'customer')
    with _off():
        side = cs.path + '.o_custkey__customer.c_custkey.test'
        WJ._JPTR_CODES.pop(side, None)
        assert WJ._codes_pointer(cs, 'o_custkey', ps, 'c_custkey', side, _mark(cs, ps, 'o_custkey', 'c_custkey')) is None
        assert WJ._JPTR_CODES[side][1] is False                  # the verdict is kept: decided once


def test_hash_pointer_same_both_ways():
    import wdb_join as WJ
    db, _con = J._fixture()
    cs, ps = _segs(db, 'lineitem', 'orders')
    old = WJ._CODES_PTR[0]
    try:
        with _off():
            out = {}
            for on in (True, False):
                WJ._CODES_PTR[0] = on
                out[on] = np.asarray(WJ._hash_pointer(db, 'lineitem', 'l_orderkey', cs, 'orders', 'o_orderkey', ps), np.int64)
            assert np.array_equal(out[True], out[False])
    finally:
        WJ._CODES_PTR[0] = old
