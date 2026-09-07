"""SQL SCOPE PROBE: what can the engine answer? ~80 constructs vs duck on a small
realm (x: 200K rows sliced from H2O G1 with a NULL-bearing copy; d: a 100-row dim).
Classifies OK / WRONG / HOLE (loud decline) / CRASH (an unnamed failure = defect).
usage: python3 bench/sql_scope.py gen | python3 bench/sql_scope.py run
"""
import sys, os, time, json, signal, traceback
sys.path.insert(0, 'src')
import numpy as np

ROOT = '/workspace/data/scope'
DB = ROOT + '/db'

def gen():
    import pyarrow as pa, pyarrow.parquet as pq
    import wdb_encode
    os.makedirs(DB, exist_ok=True)
    t = pq.read_table('/workspace/data/h2o/G1.parquet').slice(0, 200_000)
    rng = np.random.default_rng(7)
    n = t.num_rows
    # a nullable int and a nullable string, a date-ish int, a bool
    nul_i = rng.integers(0, 50, n).astype(np.int32); mask = rng.random(n) < 0.1
    nul_i = pa.array(np.where(mask, None, nul_i).tolist(), type=pa.int32())
    nul_s = pa.array([None if m else ('s%d' % v) for m, v in zip(mask.tolist(), rng.integers(0, 20, n).tolist())])
    days = pa.array((rng.integers(0, 3650, n) + 8000).astype(np.int32))
    flag = pa.array((rng.random(n) < 0.5).tolist())
    t = t.append_column('ni', nul_i).append_column('ns', nul_s).append_column('d', days).append_column('b', flag)
    d = pa.table({'id4': pa.array(np.arange(1, 101, dtype=np.int32)), 'dname': pa.array(['name%03d' % i for i in range(1, 101)]),
                  'w': pa.array(np.round(rng.random(100) * 10, 3))})
    cat = {'tables': {}}
    for name, tb in (('x', t), ('d', d)):
        p = '%s/%s.parquet' % (ROOT, name); pq.write_table(tb, p)
        wdb_encode.encode(p, '%s/%s_0.wdb' % (DB, name))
        sch = []
        for f in tb.schema:
            ty = str(f.type)
            sch.append([f.name, 'float' if 'double' in ty or 'float' in ty else ('str' if 'string' in ty else ('bool' if ty == 'bool' else 'int'))])
        cat['tables'][name] = {'segments': ['%s_0.wdb' % name], 'columns': tb.column_names, 'schema': sch}
    json.dump(cat, open(DB + '/catalog.json', 'w'))
    print('scope realm:', n, 'rows', flush=True)

Q = [
 # --- aggregates
 ('agg', 'count_star', "SELECT COUNT(*) FROM x"),
 ('agg', 'count_col_nulls', "SELECT COUNT(ni) FROM x"),
 ('agg', 'count_distinct', "SELECT COUNT(DISTINCT id2) FROM x"),
 ('agg', 'sum_distinct', "SELECT SUM(DISTINCT v1) FROM x"),
 ('agg', 'avg_nulls', "SELECT AVG(ni) FROM x"),
 ('agg', 'min_max_str', "SELECT MIN(id3), MAX(id3) FROM x"),
 ('agg', 'stddev_var', "SELECT STDDEV(v3), VARIANCE(v3) FROM x"),
 ('agg', 'stddev_pop', "SELECT STDDEV_POP(v3) FROM x"),
 ('agg', 'median', "SELECT id4, MEDIAN(v3) FROM x GROUP BY id4"),
 ('agg', 'quantile', "SELECT QUANTILE_CONT(v3, 0.9) FROM x"),
 ('agg', 'mode', "SELECT MODE(id4) FROM x"),
 ('agg', 'string_agg', "SELECT id1, STRING_AGG(DISTINCT id4, ',') FROM x GROUP BY id1"),
 ('agg', 'bool_and', "SELECT BOOL_AND(b), BOOL_OR(b) FROM x"),
 ('agg', 'any_value', "SELECT id1, ANY_VALUE(v1) FROM x GROUP BY id1"),
 ('agg', 'filter_clause', "SELECT COUNT(*) FILTER (WHERE v1 > 3) FROM x"),
 ('agg', 'agg_expr', "SELECT SUM(v1 * v2) / COUNT(*) FROM x"),
 ('agg', 'agg_case', "SELECT SUM(CASE WHEN id4 > 50 THEN v3 ELSE 0 END) FROM x"),
 ('agg', 'count_if', "SELECT SUM(CASE WHEN b THEN 1 ELSE 0 END) FROM x"),
 # --- grouping
 ('group', 'group_two', "SELECT id1, id4, COUNT(*) FROM x GROUP BY id1, id4"),
 ('group', 'group_expr', "SELECT id4 % 7 AS k, SUM(v1) FROM x GROUP BY id4 % 7"),
 ('group', 'group_having', "SELECT id4, COUNT(*) AS c FROM x GROUP BY id4 HAVING COUNT(*) > 2100"),
 ('group', 'having_nonproj', "SELECT id4 FROM x GROUP BY id4 HAVING SUM(v1) > 6000"),
 ('group', 'group_null_key', "SELECT ni, COUNT(*) FROM x GROUP BY ni"),
 ('group', 'group_nullstr_key', "SELECT ns, COUNT(*) FROM x GROUP BY ns"),
 ('group', 'rollup', "SELECT id1, id4, SUM(v1) FROM x GROUP BY ROLLUP (id1, id4)"),
 ('group', 'cube', "SELECT id1, b, SUM(v1) FROM x GROUP BY CUBE (id1, b)"),
 ('group', 'grouping_sets', "SELECT id1, id4, SUM(v1) FROM x GROUP BY GROUPING SETS ((id1), (id4))"),
 ('group', 'group_all', "SELECT id1, SUM(v1) FROM x GROUP BY ALL"),
 ('group', 'group_bool', "SELECT b, COUNT(*) FROM x GROUP BY b"),
 # --- windows
 ('window', 'row_number_topk', "SELECT id4, v3 FROM (SELECT id4, v3, ROW_NUMBER() OVER (PARTITION BY id4 ORDER BY v3 DESC) AS rn FROM x) t WHERE rn <= 2"),
 ('window', 'rank', "SELECT id4, v3, RANK() OVER (PARTITION BY id4 ORDER BY v3 DESC) AS r FROM x WHERE id4 = 1"),
 ('window', 'dense_rank', "SELECT id4, DENSE_RANK() OVER (ORDER BY id4) AS r FROM x WHERE id4 < 4"),
 ('window', 'running_sum', "SELECT id6, SUM(v1) OVER (ORDER BY v3, id6 ROWS UNBOUNDED PRECEDING) AS rs FROM x WHERE id4 = 1"),
 ('window', 'lag_lead', "SELECT id6, LAG(v1) OVER (ORDER BY v3, id6) AS prev, LEAD(v1) OVER (ORDER BY v3, id6) AS nxt FROM x WHERE id4 = 1"),
 ('window', 'ntile', "SELECT id6, NTILE(4) OVER (ORDER BY v3) AS q FROM x WHERE id4 = 1"),
 ('window', 'first_value', "SELECT id4, FIRST_VALUE(v3) OVER (PARTITION BY id4 ORDER BY v3) AS f FROM x WHERE id4 < 3"),
 ('window', 'count_over', "SELECT id4, COUNT(*) OVER (PARTITION BY id4) AS c FROM x WHERE id4 < 3"),
 ('window', 'qualify', "SELECT id4, v3 FROM x QUALIFY ROW_NUMBER() OVER (PARTITION BY id4 ORDER BY v3 DESC) = 1"),
 # --- joins
 ('join', 'inner_agg', "SELECT d.dname, SUM(x.v1) FROM x JOIN d ON x.id4 = d.id4 GROUP BY d.dname"),
 ('join', 'inner_rows', "SELECT x.id6, d.dname FROM x JOIN d ON x.id4 = d.id4 WHERE x.id4 = 5"),
 ('join', 'left_rows', "SELECT x.id6, d.dname FROM x LEFT JOIN d ON x.id4 = d.id4 AND d.id4 < 50 WHERE x.id4 IN (5, 60)"),
 ('join', 'right_join', "SELECT d.dname, COUNT(x.id6) FROM x RIGHT JOIN d ON x.id4 = d.id4 GROUP BY d.dname"),
 ('join', 'full_outer', "SELECT COUNT(*) FROM x FULL OUTER JOIN d ON x.id4 = d.id4"),
 ('join', 'cross_join', "SELECT COUNT(*) FROM d a CROSS JOIN d b"),
 ('join', 'self_join', "SELECT COUNT(*) FROM d a JOIN d b ON a.id4 = b.id4 + 1"),
 ('join', 'using', "SELECT COUNT(*) FROM x JOIN d USING (id4)"),
 ('join', 'multi_col_on', "SELECT COUNT(*) FROM x a JOIN x b ON a.id4 = b.id4 AND a.id5 = b.id5 WHERE a.id6 = 7"),
 ('join', 'nonequi', "SELECT COUNT(*) FROM d a JOIN d b ON a.w < b.w"),
 ('join', 'three_way_rows', "SELECT x.id6, d.dname, e.w FROM x JOIN d ON x.id4 = d.id4 JOIN d e ON x.id5 = e.id4 WHERE x.id4 = 3 AND x.id5 = 4"),
 ('join', 'join_where_both', "SELECT SUM(x.v1) FROM x JOIN d ON x.id4 = d.id4 WHERE d.w > 5 AND x.v2 > 7"),
 ('join', 'semi_in', "SELECT COUNT(*) FROM x WHERE id4 IN (SELECT id4 FROM d WHERE w > 5)"),
 ('join', 'anti_notin', "SELECT COUNT(*) FROM x WHERE id4 NOT IN (SELECT id4 FROM d WHERE w > 5)"),
 ('join', 'exists_corr', "SELECT COUNT(*) FROM x WHERE EXISTS (SELECT 1 FROM d WHERE d.id4 = x.id4 AND d.w > 5)"),
 # --- subqueries
 ('subq', 'scalar_uncorr', "SELECT COUNT(*) FROM x WHERE v3 > (SELECT AVG(v3) FROM x)"),
 ('subq', 'scalar_corr', "SELECT id4, v3 FROM x WHERE v3 > (SELECT MAX(v3) - 0.001 FROM x x2 WHERE x2.id4 = x.id4)"),
 ('subq', 'scalar_in_select', "SELECT id4, v1 - (SELECT AVG(v1) FROM x) AS dev FROM x WHERE id6 = 3"),
 ('subq', 'from_subquery', "SELECT k, COUNT(*) FROM (SELECT id4 % 3 AS k FROM x) t GROUP BY k"),
 ('subq', 'in_list', "SELECT COUNT(*) FROM x WHERE id4 IN (1, 2, 3)"),
 ('subq', 'any_all', "SELECT COUNT(*) FROM x WHERE v3 > ALL (SELECT w FROM d)"),
 ('subq', 'cte', "WITH t AS (SELECT id4, SUM(v1) AS s FROM x GROUP BY id4) SELECT COUNT(*) FROM t WHERE s > 6000"),
 ('subq', 'cte_recursive', "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 10) SELECT SUM(n) FROM r"),
 # --- set ops
 ('setop', 'union_all', "SELECT id4 FROM d WHERE w > 9 UNION ALL SELECT id4 FROM d WHERE w < 1"),
 ('setop', 'union', "SELECT id1 FROM x UNION SELECT id1 FROM x"),
 ('setop', 'intersect', "SELECT id4 FROM x INTERSECT SELECT id4 FROM d WHERE w > 5"),
 ('setop', 'except', "SELECT id4 FROM d EXCEPT SELECT id4 FROM x WHERE id4 < 90"),
 # --- expressions
 ('expr', 'case_multi', "SELECT CASE WHEN v1 < 2 THEN 'lo' WHEN v1 < 4 THEN 'mid' ELSE 'hi' END AS band, COUNT(*) FROM x GROUP BY 1"),
 ('expr', 'coalesce', "SELECT COUNT(*) FROM x WHERE COALESCE(ni, -1) = -1"),
 ('expr', 'nullif', "SELECT SUM(NULLIF(v1, 3)) FROM x"),
 ('expr', 'cast', "SELECT SUM(CAST(id4 AS DOUBLE) / 3) FROM x"),
 ('expr', 'string_fns', "SELECT COUNT(*) FROM x WHERE UPPER(id1) = 'ID001' AND LENGTH(id3) > 6"),
 ('expr', 'concat', "SELECT COUNT(DISTINCT id1 || '-' || id2) FROM x"),
 ('expr', 'like_ilike', "SELECT COUNT(*) FROM x WHERE id3 LIKE 'id00%' AND id2 ILIKE 'ID01%'"),
 ('expr', 'regexp', "SELECT COUNT(*) FROM x WHERE REGEXP_MATCHES(id3, '^id0+9')"),
 ('expr', 'substring_replace', "SELECT COUNT(*) FROM x WHERE SUBSTRING(id3, 3, 2) = '01' AND REPLACE(id1, 'id', 'x') = 'x005'"),
 ('expr', 'math', "SELECT SUM(ROUND(v3, 1)), SUM(FLOOR(v3)), SUM(ABS(v1 - 3)), SUM(SQRT(v3)), SUM(LOG(v3 + 1)) FROM x"),
 ('expr', 'int_div_mod', "SELECT SUM(id4 // 7), SUM(id4 % 7) FROM x"),
 ('expr', 'between', "SELECT COUNT(*) FROM x WHERE v3 BETWEEN 10 AND 20"),
 ('expr', 'is_null', "SELECT COUNT(*) FROM x WHERE ni IS NULL OR ns IS NOT NULL"),
 ('expr', 'bool_col', "SELECT COUNT(*) FROM x WHERE b AND v1 > 2"),
 ('expr', 'not_in_list', "SELECT COUNT(*) FROM x WHERE id1 NOT IN ('id001', 'id002')"),
 ('expr', 'date_arith', "SELECT COUNT(*) FROM x WHERE d + 30 > 11000"),
 ('expr', 'div_by_zero', "SELECT SUM(v1 / (id4 - 5)) FROM x WHERE id4 <> 5"),
 ('expr', 'nested_arith', "SELECT SUM((v1 + v2) * (v3 - 1) / 2) FROM x"),
 # --- ordering / limits / distinct
 ('order', 'order_multi', "SELECT id4, id5, COUNT(*) AS c FROM x GROUP BY id4, id5 ORDER BY c DESC, id4, id5 LIMIT 5"),
 ('order', 'order_expr', "SELECT id4, SUM(v1) AS s FROM x GROUP BY id4 ORDER BY s * 2 DESC LIMIT 3"),
 ('order', 'nulls_last', "SELECT ni, COUNT(*) FROM x GROUP BY ni ORDER BY ni NULLS LAST LIMIT 3"),
 ('order', 'offset', "SELECT id4, COUNT(*) AS c FROM x GROUP BY id4 ORDER BY id4 LIMIT 5 OFFSET 10"),
 ('order', 'distinct_multi', "SELECT DISTINCT id1, id2 FROM x WHERE id4 = 1"),
 ('order', 'distinct_on', "SELECT DISTINCT ON (id4) id4, v3 FROM x WHERE id4 < 3 ORDER BY id4, v3 DESC"),
 ('order', 'select_rows_where', "SELECT id6, v3 FROM x WHERE id4 = 1 AND v3 > 99"),
 ('order', 'top_no_group', "SELECT id6, v3 FROM x ORDER BY v3 DESC LIMIT 5"),
 # --- edge cases
 ('edge', 'empty_result', "SELECT id4, SUM(v1) FROM x WHERE v3 > 1000 GROUP BY id4"),
 ('edge', 'scalar_over_empty', "SELECT SUM(v1), COUNT(*), MIN(v1) FROM x WHERE v3 > 1000"),
 ('edge', 'literal_only', "SELECT 1 + 1, 'a'"),
 ('edge', 'values', "SELECT * FROM (VALUES (1, 'a'), (2, 'b')) v(n, s)"),
 ('edge', 'count_null_str', "SELECT COUNT(ns), COUNT(DISTINCT ns) FROM x"),
 ('edge', 'sum_all_null', "SELECT SUM(ni) FROM x WHERE ni IS NULL"),
 ('edge', 'string_compare', "SELECT COUNT(*) FROM x WHERE id3 > 'id05'"),
 ('edge', 'float_eq', "SELECT COUNT(*) FROM x WHERE v3 = 50.0"),
]

def run():
    import duckdb
    import wdb_kernels; wdb_kernels.warm()
    from wdb_db import Database
    db = Database.open(DB)
    con = duckdb.connect()
    for name in ('x', 'd'):
        con.execute("CREATE VIEW %s AS SELECT * FROM read_parquet('%s/%s.parquet')" % (name, ROOT, name))
    def norm(v):
        if isinstance(v, bool): return 'b:%d' % v
        if isinstance(v, float): return '%.6g' % v
        if v is None: return 'NULL'
        try:
            import decimal
            if isinstance(v, decimal.Decimal): return '%.6g' % float(v)
        except Exception: pass
        return str(v)
    def same(w, e):
        if len(w) != len(e): return False
        kw = sorted(tuple(norm(v) for v in r) for r in w); ke = sorted(tuple(norm(v) for v in r) for r in e)
        return kw == ke
    tally = {}
    def _alarm(sig, frm): raise TimeoutError('timeout')
    signal.signal(signal.SIGALRM, _alarm)
    for cat, name, q in Q:
        try:
            e = con.execute(q).fetchall()
        except Exception as ex:
            print('%-7s %-20s DUCK-ERR %s' % (cat, name, str(ex)[:60]), flush=True); continue
        signal.alarm(60)
        try:
            w = db.run(q); w = w[0] if isinstance(w, tuple) else w
            signal.alarm(0)
            status = 'OK' if same(w, e) else 'WRONG'
            detail = '' if status == 'OK' else ' wave=%s duck=%s' % (str(w[:2])[:50], str(e[:2])[:50])
        except NotImplementedError as ex:
            signal.alarm(0); status = 'HOLE'; detail = ' ' + str(ex)[:70]
        except BaseException as ex:
            signal.alarm(0); status = 'CRASH'; detail = ' %s: %s' % (type(ex).__name__, str(ex)[:60])
        tally[status] = tally.get(status, 0) + 1
        print('%-7s %-20s %-5s%s' % (cat, name, status, detail), flush=True)
    print('SCOPE: %d constructs | %s' % (len(Q), ' '.join('%s=%d' % kv for kv in sorted(tally.items()))), flush=True)

if __name__ == '__main__':
    gen() if sys.argv[1] == 'gen' else run()
