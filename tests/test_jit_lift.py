"""ONE COMPILE PER SHAPE (c125, 2026-11-04): the fused kernels' source no longer carries a query's literals -- every
number in the predicate and the value expressions is a call argument -- so variants of one shape that differ only in
their numbers reuse ONE compiled kernel (TPC-H Q5 / Q7 / Q12 / Q17 / Q19 compiled a new kernel per variant: 0.4-1.2 s
on each new variant's first run). Answers like DuckDB; switched off, each variant compiles its own kernel again."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

# literals inside the fused predicate (an OR of conjunctions the cascade leaves to the pred -- the Q7 / Q19 shape)
SHAPE = ("SELECT o_orderpriority, SUM(l_quantity), COUNT(*) FROM lineitem, orders WHERE l_orderkey = o_orderkey AND "
         "((l_quantity < {a} AND l_discount > 0.05) OR (l_quantity > {b} AND l_discount < 0.02)) GROUP BY o_orderpriority")
VARIANTS = [(10, 40), (15, 45), (5, 35), (20, 30)]


def _same(db, con, q):
    import controller, test_join_fast as J
    controller.plan_epoch_bump()
    exp = sorted(J._norm([tuple(r) for r in con.execute(q).fetchall()]))
    r = db.run(q)
    got = sorted(J._norm(r[0] if isinstance(r, tuple) else r))
    assert len(got) == len(exp), (q, len(got), len(exp))
    for g, e in zip(got, exp):
        assert J._eq(g, e), (q, g, e)


def _new_kernels(db, con):
    import wdb_exprjit as X
    grow = []
    for a, b in VARIANTS:
        n0 = len(X._CACHE)
        _same(db, con, SHAPE.format(a=a, b=b))
        grow.append(len(X._CACHE) - n0)
    return grow


# The kernel cache is never cleared here: dropping compiled kernels leaves dead weak references behind, and a later
# test that scans gc.get_objects() (test_regexgroup_general) trips over them. Growth is measured from wherever the
# cache stands: with the lift only the first variant may add a kernel (none if the shape ran before).


def test_one_compile_per_shape():
    import test_join_fast as J
    db, con = J._fixture()
    grow = _new_kernels(db, con)
    assert sum(grow[1:]) == 0, grow


def test_switched_off_compiles_per_variant():
    import wdb_exprjit as X, test_join_fast as J
    db, con = J._fixture()
    o0 = X._LIFT[0]
    try:
        X._LIFT[0] = False
        grow = _new_kernels(db, con)
        assert all(g >= 1 for g in grow), grow       # the old way: each variant's numbers are new source
    finally:
        X._LIFT[0] = o0
