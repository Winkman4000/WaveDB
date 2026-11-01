"""THE PLANES SERVED IT ALL (2026-11-01): a joined GROUP BY whose whole WHERE is date bands the field planes test
exactly (the partition pre-pass banks every conjunct and hands the survivors off as rows) answers like DuckDB --
it raised IndexError (an empty leftover list indexed for a pred) before c103."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))


def test_date_bands_only():
    import test_join_fast as J
    J._match("SELECT n.n_name, COUNT(*), SUM(l.l_extendedprice) FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey "
             "JOIN customer c ON o.o_custkey = c.c_custkey JOIN nation n ON c.c_nationkey = n.n_nationkey "
             "WHERE l.l_shipdate >= DATE '1995-01-01' AND l.l_shipdate < DATE '1996-01-01' GROUP BY n.n_name",
             expect_fast=False)
    J._match("SELECT o.o_orderpriority, COUNT(*), SUM(l.l_quantity) FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey "
             "WHERE l.l_receiptdate >= DATE '1994-01-01' AND l.l_receiptdate < DATE '1995-01-01' GROUP BY o.o_orderpriority",
             expect_fast=False)
    J._match("SELECT l.l_returnflag, COUNT(*) FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey "
             "WHERE l.l_shipdate >= DATE '1993-06-01' AND l.l_shipdate < DATE '1993-09-01' "
             "AND l.l_commitdate >= DATE '1993-01-01' GROUP BY l.l_returnflag",
             expect_fast=False)
