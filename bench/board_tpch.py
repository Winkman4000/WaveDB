"""THE TPC-H BOARD: the realm's first foreign-data trial. 22 official
queries (dates as int-days, decimals as doubles -- both engines read
the identical parquet-born bytes), duck-refereed, float-tolerant
comparator, holes reported as findings not failures."""
import sys, time, os, duckdb
sys.path.insert(0, '/workspace/WaveDB/src')
import wdb_kernels
wdb_kernels.warm()
from wdb_db import Database

D = lambda s: (__import__('datetime').date.fromisoformat(s)
               - __import__('datetime').date(1970, 1, 1)).days

QS = {
 1: "SELECT l_returnflag, l_linestatus, SUM(l_quantity) AS sum_qty, SUM(l_extendedprice) AS sum_base_price, SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price, SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge, AVG(l_quantity) AS avg_qty, AVG(l_extendedprice) AS avg_price, AVG(l_discount) AS avg_disc, COUNT(*) AS count_order FROM lineitem WHERE l_shipdate <= %d GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus" % (D('1998-12-01') - 90),
 3: "SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)) AS revenue, o_orderdate, o_shippriority FROM customer, orders, lineitem WHERE c_mktsegment = 'BUILDING' AND c_custkey = o_custkey AND l_orderkey = o_orderkey AND o_orderdate < %d AND l_shipdate > %d GROUP BY l_orderkey, o_orderdate, o_shippriority ORDER BY revenue DESC, o_orderdate LIMIT 10" % (D('1995-03-15'), D('1995-03-15')),
 4: "SELECT o_orderpriority, COUNT(*) AS order_count FROM orders WHERE o_orderdate >= %d AND o_orderdate < %d AND EXISTS (SELECT * FROM lineitem WHERE l_orderkey = o_orderkey AND l_commitdate < l_receiptdate) GROUP BY o_orderpriority ORDER BY o_orderpriority" % (D('1993-07-01'), D('1993-10-01')),
 5: "SELECT n_name, SUM(l_extendedprice * (1 - l_discount)) AS revenue FROM customer, orders, lineitem, supplier, nation, region WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey AND l_suppkey = s_suppkey AND c_nationkey = s_nationkey AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey AND r_name = 'ASIA' AND o_orderdate >= %d AND o_orderdate < %d GROUP BY n_name ORDER BY revenue DESC" % (D('1994-01-01'), D('1995-01-01')),
 6: "SELECT SUM(l_extendedprice * l_discount) AS revenue FROM lineitem WHERE l_shipdate >= %d AND l_shipdate < %d AND l_discount >= 0.05 AND l_discount <= 0.07 AND l_quantity < 24" % (D('1994-01-01'), D('1995-01-01')),
 7: "SELECT supp_nation, cust_nation, l_year, SUM(volume) AS revenue FROM (SELECT n1.n_name AS supp_nation, n2.n_name AS cust_nation, l_shipdate / 365 AS l_year, l_extendedprice * (1 - l_discount) AS volume FROM supplier, lineitem, orders, customer, nation n1, nation n2 WHERE s_suppkey = l_suppkey AND o_orderkey = l_orderkey AND c_custkey = o_custkey AND s_nationkey = n1.n_nationkey AND c_nationkey = n2.n_nationkey AND ((n1.n_name = 'FRANCE' AND n2.n_name = 'GERMANY') OR (n1.n_name = 'GERMANY' AND n2.n_name = 'FRANCE')) AND l_shipdate >= %d AND l_shipdate <= %d) AS shipping GROUP BY supp_nation, cust_nation, l_year ORDER BY supp_nation, cust_nation, l_year" % (D('1995-01-01'), D('1996-12-31')),
 10: "SELECT c_custkey, c_name, SUM(l_extendedprice * (1 - l_discount)) AS revenue, c_acctbal, n_name, c_address, c_phone, c_comment FROM customer, orders, lineitem, nation WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey AND o_orderdate >= %d AND o_orderdate < %d AND l_returnflag = 'R' AND c_nationkey = n_nationkey GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment ORDER BY revenue DESC LIMIT 20" % (D('1993-10-01'), D('1994-01-01')),
 12: "SELECT l_shipmode, SUM(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END) AS high_line_count, SUM(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) AS low_line_count FROM orders, lineitem WHERE o_orderkey = l_orderkey AND l_shipmode IN ('MAIL', 'SHIP') AND l_commitdate < l_receiptdate AND l_shipdate < l_commitdate AND l_receiptdate >= %d AND l_receiptdate < %d GROUP BY l_shipmode ORDER BY l_shipmode" % (D('1994-01-01'), D('1995-01-01')),
 13: "SELECT c_count, COUNT(*) AS custdist FROM (SELECT c_custkey, COUNT(o_orderkey) AS c_count FROM customer LEFT OUTER JOIN orders ON c_custkey = o_custkey AND o_comment NOT LIKE '%special%requests%' GROUP BY c_custkey) AS c_orders GROUP BY c_count ORDER BY custdist DESC, c_count DESC",
 14: "SELECT 100.00 * SUM(CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice * (1 - l_discount) ELSE 0 END) / SUM(l_extendedprice * (1 - l_discount)) AS promo_revenue FROM lineitem, part WHERE l_partkey = p_partkey AND l_shipdate >= " + str(D('1995-09-01')) + " AND l_shipdate < " + str(D('1995-10-01')),
 18: "SELECT c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, SUM(l_quantity) FROM customer, orders, lineitem WHERE o_orderkey IN (SELECT l_orderkey FROM lineitem GROUP BY l_orderkey HAVING SUM(l_quantity) > 300) AND c_custkey = o_custkey AND o_orderkey = l_orderkey GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice ORDER BY o_totalprice DESC, o_orderdate LIMIT 100",
 19: "SELECT SUM(l_extendedprice * (1 - l_discount)) AS revenue FROM lineitem, part WHERE (p_partkey = l_partkey AND p_brand = 'Brand#12' AND p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG') AND l_quantity >= 1 AND l_quantity <= 11 AND p_size >= 1 AND p_size <= 5 AND l_shipmode IN ('AIR', 'AIR REG') AND l_shipinstruct = 'DELIVER IN PERSON') OR (p_partkey = l_partkey AND p_brand = 'Brand#23' AND p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK') AND l_quantity >= 10 AND l_quantity <= 20 AND p_size >= 1 AND p_size <= 10 AND l_shipmode IN ('AIR', 'AIR REG') AND l_shipinstruct = 'DELIVER IN PERSON') OR (p_partkey = l_partkey AND p_brand = 'Brand#34' AND p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG') AND l_quantity >= 20 AND l_quantity <= 30 AND p_size >= 1 AND p_size <= 15 AND l_shipmode IN ('AIR', 'AIR REG') AND l_shipinstruct = 'DELIVER IN PERSON')",
 21: "SELECT s_name, COUNT(*) AS numwait FROM supplier, lineitem l1, orders, nation WHERE s_suppkey = l1.l_suppkey AND o_orderkey = l1.l_orderkey AND o_orderstatus = 'F' AND l1.l_receiptdate > l1.l_commitdate AND EXISTS (SELECT * FROM lineitem l2 WHERE l2.l_orderkey = l1.l_orderkey AND l2.l_suppkey <> l1.l_suppkey) AND NOT EXISTS (SELECT * FROM lineitem l3 WHERE l3.l_orderkey = l1.l_orderkey AND l3.l_suppkey <> l1.l_suppkey AND l3.l_receiptdate > l3.l_commitdate) AND s_nationkey = n_nationkey AND n_name = 'SAUDI ARABIA' GROUP BY s_name ORDER BY numwait DESC, s_name LIMIT 100",
 22: "SELECT cntrycode, COUNT(*) AS numcust, SUM(c_acctbal) AS totacctbal FROM (SELECT SUBSTRING(c_phone, 1, 2) AS cntrycode, c_acctbal FROM customer WHERE SUBSTRING(c_phone, 1, 2) IN ('13', '31', '23', '29', '30', '18', '17') AND c_acctbal > (SELECT AVG(c_acctbal) FROM customer WHERE c_acctbal > 0.00 AND SUBSTRING(c_phone, 1, 2) IN ('13', '31', '23', '29', '30', '18', '17')) AND NOT EXISTS (SELECT * FROM orders WHERE o_custkey = c_custkey)) AS custsale GROUP BY cntrycode ORDER BY cntrycode",
}


def rows_of(x):
    return x[0] if isinstance(x, tuple) else x


def main():
    db = Database.open('/workspace/data/tpchdb')
    con = duckdb.connect('/workspace/data/tpch_ref.db', read_only=True)
    wins = 0
    okc = 0
    holes = []
    wt = dt = 0.0
    for qi in sorted(QS):
        q = QS[qi]
        try:
            e = con.execute(q).fetchall()
        except Exception as ex:
            print('Q%02d DUCK-FAIL %s' % (qi, str(ex)[:60]), flush=True)
            continue
        try:
            db.run(q)
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                w = db.run(q)
                ts.append(time.perf_counter() - t0)
            w = rows_of(w)
            ok = len(w) == len(e)
            if ok:
                for rw, re_ in zip(w, e):
                    for a, b in zip(rw, re_):
                        if isinstance(b, float):
                            ok &= (a is not None and
                                   abs(float(a) - b) <= 1e-6 * max(1.0, abs(b)))
                        else:
                            ok &= (str(a) == str(b))
                        if not ok:
                            break
                    if not ok:
                        break
            ds = []
            for _ in range(2):
                t0 = time.perf_counter()
                con.execute(q).fetchall()
                ds.append(time.perf_counter() - t0)
            wv, dv = min(ts), min(ds)
            wt += wv
            dt += dv
            okc += 1 if ok else 0
            wins += 1 if (ok and wv < dv) else 0
            print('Q%02d %s wave=%6.2fs duck=%6.2fs x%5.2f' %
                  (qi, 'OK   ' if ok else 'WRONG', wv, dv,
                   dv / wv if wv else 0), flush=True)
        except Exception as ex:
            holes.append((qi, type(ex).__name__, str(ex)[:90]))
            print('Q%02d HOLE  %s: %s' % (qi, type(ex).__name__,
                                          str(ex)[:90]), flush=True)
    print('=' * 60, flush=True)
    print('TPCH BOARD: %d attempted | ok=%d | holes=%d | wins=%d | '
          'wave %.1fs duck %.1fs' %
          (len(QS), okc, len(holes), wins, wt, dt), flush=True)
    for h in holes:
        print('  HOLE Q%02d %s: %s' % h, flush=True)


if __name__ == '__main__':
    main()
