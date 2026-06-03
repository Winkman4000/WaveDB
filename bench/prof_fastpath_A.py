"""Steady-state (cached-segment) sub-step breakdown for query [A]."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, sqlglot
from wdb_db import Database
import wdb_sql, wdb_agg, wdb_fkptr

db = Database.open('/tmp/jbprof_sf1.0/wdb')
def t(label, fn, n=5):
    fn(); ts=[]
    for _ in range(n): a=time.perf_counter(); fn(); ts.append(time.perf_counter()-a)
    print(f"  {label:46s} {min(ts)*1000:8.1f} ms"); return fn()

qA = ("SELECT c.c_mktsegment, COUNT(*), SUM(o.o_totalprice) FROM orders o "
      "JOIN customer c ON o.o_custkey=c.c_custkey GROUP BY c.c_mktsegment")

oseg = db.open_segment(db.cat.segment_paths('orders')[0], 'orders')      # cache warm
cseg = db.open_segment(db.cat.segment_paths('customer')[0], 'customer')
osp  = db.cat.segment_paths('orders')[0]

print("=== [A] cached-segment sub-steps (orders 1.5M x customer 150K) ===")
t("open_segment(orders)  [cached]", lambda: db.open_segment(osp, 'orders'))
ptr = t("wdb_fkptr.load(orders,o_custkey) 1.5M", lambda: wdb_fkptr.load(osp, 'o_custkey'))
t("cseg.codes(c_mktsegment) 150K", lambda: cseg.codes('c_mktsegment'))
full = cseg.codes('c_mktsegment')
t("gather full[ptr] 1.5M (scattered)", lambda: full[ptr])
t("wdb_sql._col(orders,o_totalprice) 1.5M decode", lambda: wdb_sql._col(oseg, 'o_totalprice'))
price,_ = wdb_sql._col(oseg, 'o_totalprice')
gcodes = full[ptr].astype(np.int64); K=int(gcodes.max())+1
t("group_counts", lambda: wdb_agg.group_counts(gcodes, K))
t("group_agg SUM", lambda: wdb_agg.group_agg(gcodes, K, 'SUM', price.astype(np.float64)))
t("FULL db.run(qA)", lambda: db.run(qA))
