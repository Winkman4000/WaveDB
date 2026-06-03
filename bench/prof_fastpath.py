"""Where does db.run's fast path spend its time? Break down query [B] step by step."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, sqlglot
from wdb_db import Database
from wdb_engine import Segment
import wdb_dml, wdb_sql, wdb_agg, wdb_fkptr

DIR = '/tmp/jbprof_sf1.0/wdb'
db = Database.open(DIR)

def t(label, fn, n=5):
    fn(); ts=[]
    for _ in range(n): a=time.perf_counter(); r=fn(); ts.append(time.perf_counter()-a)
    print(f"  {label:42s} {min(ts)*1000:8.1f} ms"); return fn()

qB = ("SELECT o.o_orderpriority, COUNT(*), SUM(l.l_quantity) FROM lineitem l "
      "JOIN orders o ON l.l_orderkey=o.o_orderkey GROUP BY o.o_orderpriority")

print("=== fast-path sub-steps for [B] (lineitem 6M x orders 1.5M) ===")
t("sqlglot.parse_one", lambda: sqlglot.parse_one(qB))

lpaths = db.cat.segment_paths('lineitem'); opaths = db.cat.segment_paths('orders')
t("Segment(lineitem)        construct", lambda: Segment(lpaths[0]))
t("Segment(orders)          construct", lambda: Segment(opaths[0]))

lseg = Segment(lpaths[0]); oseg = Segment(opaths[0])
t("register_synth(lineitem)", lambda: wdb_dml.register_synth(db.cat, lseg, 'lineitem'))
t("register_synth(orders)", lambda: wdb_dml.register_synth(db.cat, oseg, 'orders'))
wdb_dml.register_synth(db.cat, lseg, 'lineitem'); wdb_dml.register_synth(db.cat, oseg, 'orders')
t("lseg.presence_mask()", lambda: lseg.presence_mask())

ptr = t("wdb_fkptr.load(lineitem,l_orderkey) 6M", lambda: wdb_fkptr.load(lpaths[0], 'l_orderkey'))
t("oseg.codes(o_orderpriority) 1.5M", lambda: oseg.codes('o_orderpriority'))
full = oseg.codes('o_orderpriority')
t("gather full[ptr] 6M", lambda: full[ptr])
t("wdb_sql._col(lineitem,l_quantity) 6M decode", lambda: wdb_sql._col(lseg, 'l_quantity'))
qty,_ = wdb_sql._col(lseg, 'l_quantity')
gcodes = full[ptr].astype(np.int64); K=int(gcodes.max())+1
t("group_counts kernel", lambda: wdb_agg.group_counts(gcodes, K))
t("group_agg SUM kernel", lambda: wdb_agg.group_agg(gcodes, K, 'SUM', qty.astype(np.float64)))

import wdb_join
def whole():
    h=wdb_join._FAST_HITS; r=db.run(qB); return r
t("FULL db.run(qB)", whole)
