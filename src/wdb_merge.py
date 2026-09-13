"""WaveDB two-tier merge-read: combine a cold segment + a hot buffer into one answer.

The hard part is aggregates: you cannot average two averages. So we push MERGEABLE
PARTIALS to each tier (AVG -> SUM+COUNT, SUM->SUM, MIN/MAX, COUNT(*)->count), strip
HAVING/ORDER/LIMIT for the per-tier queries, merge partials per group key, then
reconstruct (AVG = sum/count) and apply HAVING/ORDER/LIMIT once on the merged result.

Cold tier is queried by our own engine; hot tier (a small uncompressed parquet) by
DuckDB. The novel WaveDB value lives entirely in the cold tier.
"""
import copy, datetime, sqlglot, sqlglot.expressions as E
import numpy as np
import duckdb
import wdb_sql
from wdb_engine import Segment

_AGG = (E.Count, E.Sum, E.Min, E.Max, E.Avg)

def _inner(p):
    return p.this if isinstance(p, E.Alias) else p

def _is_cdistinct(node):
    return (isinstance(node, E.Count) and isinstance(node.this, E.Distinct)
            and len(node.this.expressions) == 1 and isinstance(node.this.expressions[0], E.Column))

def _classify(proj):
    """Return (keys, plan, partial_select_sql_exprs).
    plan[i] describes how to build final column i from merged partials."""
    keys = []          # list of key column names, in first-seen order
    partials = []      # list of (sqlglot_expr_sql, ) appended after keys
    plan = []          # per original projection
    def add_partial(sql_expr):
        partials.append(sql_expr); return len(keys) + len(partials) - 1
    for p in proj:
        node = _inner(p)
        if _is_cdistinct(node):                       # COUNT(DISTINCT col): merged via value-set union, not partials
            plan.append(('cdistinct', node.this.expressions[0].name)); continue
        # NOT MERGEABLE from partials: an aggregate over DISTINCT (SUM(DISTINCT v) summed 5x across
        # 5 segments), a FILTER clause, or an aggregate wrapped in arithmetic -- decline by name
        if isinstance(node, E.AggFunc) and isinstance(node.this, E.Distinct):
            raise NotImplementedError(f"multi-segment merge: {node.sql()} is not mergeable from partials (the union view serves it)")
        if isinstance(node, E.Filter) or (not isinstance(node, (E.Column, E.AggFunc)) and node.find(E.AggFunc) is not None):
            raise NotImplementedError(f"multi-segment merge: {node.sql()[:60]} is not mergeable from partials (the union view serves it)")
        if isinstance(node, E.Column):
            nm = node.name
            if nm not in keys: keys.append(nm)
            plan.append(('key', keys.index(nm)))
        elif isinstance(node, E.Count):
            if isinstance(node.this, E.Star) or node.this is None:
                plan.append(('count_star', add_partial('COUNT(*)')))
            else:
                c = node.this.sql(); plan.append(('sum', add_partial(f'COUNT({c})')))
        elif isinstance(node, E.Sum):
            c = node.this.sql(); plan.append(('sum', add_partial(f'SUM({c})')))
        elif isinstance(node, E.Min):
            c = node.this.sql(); plan.append(('min', add_partial(f'MIN({c})')))
        elif isinstance(node, E.Max):
            c = node.this.sql(); plan.append(('max', add_partial(f'MAX({c})')))
        elif isinstance(node, E.Avg):
            c = node.this.sql()
            s = add_partial(f'SUM({c})'); n = add_partial(f'COUNT({c})')
            plan.append(('avg', s, n))
        else:
            raise NotImplementedError(f"unsupported projection in buffered table: {node.sql()}")
    return keys, plan, partials

def _partial_sql(tree, table_ref, keys, partials):
    """Render a partial-aggregate SQL string against table_ref (no HAVING/ORDER/LIMIT)."""
    sel = []
    for i, k in enumerate(keys): sel.append(f'"{k}" AS _k{i}')
    for j, pe in enumerate(partials): sel.append(f'{pe} AS _p{j}')
    where = tree.args.get('where')
    wsql = f" WHERE {where.this.sql(dialect='duckdb')}" if where is not None else ""
    gsql = f" GROUP BY {', '.join(chr(34)+k+chr(34) for k in keys)}" if keys else ""
    return f"SELECT {', '.join(sel)} FROM {table_ref}{wsql}{gsql}"

def _num(x): return 0 if x is None else x

def _mm(v):
    """Normalize a MIN/MAX partial for cross-tier comparison: the cold tier renders datetime as an
    ISO string (via _pyval) while DuckDB returns a datetime object. Render the DuckDB value the same
    way so min/max compares like-with-like (ISO strings order correctly) and the output is uniform."""
    if isinstance(v, (datetime.datetime, datetime.date)):
        return wdb_sql._pyval(np.datetime64(v))
    return v

def _merge_partials(row_lists, n_keys, plan):
    """Merge N lists of partial rows (keys first, then partials) by group key."""
    acc = {}
    for rows in row_lists:
        for r in rows:
            key = tuple(r[:n_keys]); parts = r[n_keys:]
            if key not in acc:
                acc[key] = [None] * len(parts)
            cur = acc[key]
            for slot, val in enumerate(parts):
                # determine op for this partial slot from plan
                op = _slot_op(slot + n_keys, plan)
                if op == 'count_star':
                    cur[slot] = _num(cur[slot]) + _num(val)
                elif op == 'sum':                       # None-preserving: an all-NULL group sums to NULL, not 0
                    if val is not None:
                        cur[slot] = val if cur[slot] is None else cur[slot] + val
                elif op == 'min':
                    a = _mm(cur[slot]); b = _mm(val)
                    cur[slot] = b if a is None else (a if b is None else min(a, b))
                elif op == 'max':
                    a = _mm(cur[slot]); b = _mm(val)
                    cur[slot] = b if a is None else (a if b is None else max(a, b))
                else:  # avg-sum or avg-count slots are summed
                    cur[slot] = _num(cur[slot]) + _num(val)
    return acc

def _slot_op(abs_slot, plan):
    for entry in plan:
        if entry[0] in ('sum', 'count_star', 'min', 'max') and entry[1] == abs_slot:
            return entry[0]
        if entry[0] == 'avg' and abs_slot in (entry[1], entry[2]):
            return 'sum'
    return 'sum'

def merge_query(segs, hot_parquet, sql, col_map=None):
    """segs: list of cold Segments (0..N). hot_parquet: path or None."""
    if segs is None: segs = []
    tree = sqlglot.parse_one(sql, read='duckdb')
    if not isinstance(tree, E.Select): raise NotImplementedError("merge: only SELECT")
    if tree.args.get('joins') or tree.args.get('with'):
        raise NotImplementedError("merge: JOIN/CTE not supported")
    proj = tree.expressions
    group = tree.args.get('group')
    has_group = group is not None
    # ANY aggregate counts -- MODE, QUANTILE, BOOL_AND, STRING_AGG had fallen into the row-union
    # path and came back once per segment (the segments board, 2026-09-13: 7 silent wrongs)
    has_agg = any(isinstance(_inner(p), E.AggFunc) or _inner(p).find(E.AggFunc) is not None for p in proj)

    # ---- no aggregates, no GROUP BY: plain row union ----
    if not has_agg and not has_group:
        rows = []
        # strip ORDER/LIMIT for per-tier; apply post-merge
        base = copy.deepcopy(tree)
        base.set('order', None); base.set('limit', None); base.set('offset', None)
        bsql = base.sql(dialect='duckdb')
        for seg in segs:
            r, _ = wdb_sql.execute(seg, bsql, col_map=col_map); rows += list(r)
        if hot_parquet is not None:
            con = duckdb.connect()
            base_p = wdb_sql._to_physical(base, col_map)   # hot parquet carries physical names
            rows += [tuple(x) for x in con.execute(_duck_from(base_p, hot_parquet)).fetchall()]
        if tree.args.get('distinct') is not None:        # SELECT DISTINCT: per-tier dedup is not enough
            seen = set(); ded = []                       # -> dedup the cross-tier union (order-preserving)
            for r in rows:
                if r not in seen: seen.add(r); ded.append(r)
            rows = ded
        rows = wdb_sql._apply_order(rows, proj, tree.args.get('order'))
        lim = wdb_sql._limit(tree); off = wdb_sql._offset(tree)
        if lim is not None or off:
            rows = rows[off: off + lim] if lim is not None else rows[off:]
        return rows, [wdb_sql._alias(p) for p in proj]

    # ---- aggregates / GROUP BY: partial merge ----
    keys, plan, partials = _classify(proj)
    has_cd = any(e[0] == 'cdistinct' for e in plan)
    row_lists = []
    if keys or partials:
        psql = _partial_sql(tree, "tbl", keys, partials)
        for seg in segs:
            r, _ = wdb_sql.execute(seg, psql, col_map=col_map); row_lists.append(list(r))
        if hot_parquet is not None:
            tree_p = wdb_sql._to_physical(tree, col_map)       # hot parquet carries physical names;
            keys_p, _, partials_p = _classify(tree_p.expressions)  # same structure as cold, by position
            psql_h = _partial_sql(tree_p, f"'{hot_parquet}'", keys_p, partials_p)
            con = duckdb.connect()
            row_lists.append([tuple(x) for x in con.execute(psql_h).fetchall()])
        acc = _merge_partials(row_lists, len(keys), plan)
    else:
        acc = {(): []}                                # pure COUNT(DISTINCT), no GROUP BY -> one global group

    # COUNT(DISTINCT col): the mergeable partial is the SET of distinct values per group, not a count
    # (summing per-segment counts double-counts values that span segments). Gather distinct
    # (keys..., value) pairs from every tier, union per group key, then count the non-null values.
    cd = {}
    if has_cd:
        where = tree.args.get('where')
        wsql = f" WHERE {where.this.sql(dialect='duckdb')}" if where is not None else ""
        for entry in plan:
            if entry[0] != 'cdistinct': continue
            col = entry[1]; sets = cd.setdefault(col, {})
            sel = ", ".join([f'"{k}" AS _k{i}' for i, k in enumerate(keys)] + [f'"{col}" AS _v'])
            gsql = " GROUP BY " + ", ".join(chr(34) + g + chr(34) for g in (keys + [col]))
            qsql = f"SELECT {sel} FROM tbl{wsql}{gsql}"
            pair_lists = []
            for seg in segs:
                r, _ = wdb_sql.execute(seg, qsql, col_map=col_map); pair_lists.append(r)
            if hot_parquet is not None:
                pk = [(col_map.get(k, k) if col_map else k) for k in keys]
                pc = (col_map.get(col, col) if col_map else col)
                pw = wdb_sql._to_physical(copy.deepcopy(tree), col_map).args.get('where')
                pwsql = f" WHERE {pw.this.sql(dialect='duckdb')}" if pw is not None else ""
                psel = ", ".join([f'"{k}"' for k in pk] + [f'"{pc}"'])
                pg = " GROUP BY " + ", ".join(chr(34) + g + chr(34) for g in (pk + [pc]))
                con = duckdb.connect()
                pair_lists.append([tuple(x) for x in
                                   con.execute(f"SELECT {psel} FROM '{hot_parquet}'{pwsql}{pg}").fetchall()])
            for rows in pair_lists:
                for r in rows:
                    kt = tuple(r[:len(keys)]); val = r[len(keys)]
                    if val is None: continue                  # COUNT(DISTINCT) ignores NULL
                    sets.setdefault(kt, set()).add(_mm(val))
            for kt in sets:
                acc.setdefault(kt, [None] * len(partials))     # emit groups seen only via the distinct gather

    out = []
    for key, parts in acc.items():
        full = list(key) + list(parts)
        row = []
        for entry in plan:
            if entry[0] == 'key':
                row.append(key[entry[1]])
            elif entry[0] == 'cdistinct':
                row.append(len(cd.get(entry[1], {}).get(key, ())))
            elif entry[0] in ('sum', 'count_star', 'min', 'max'):
                v = full[entry[1]]
                # INTEGER EMISSION across segments: a SUM/MIN/MAX of an integer-typed expression is an int
                # (SUM(id4 // 7) came back 67895654.0 from the merge, an int from a single segment)
                if isinstance(v, float) and v == v and float(v).is_integer():
                    try:
                        _pn = _inner(proj[len(row)])
                        if isinstance(_pn, (E.Sum, E.Min, E.Max)) and wdb_sql._expr_is_int(_pn.this, segs[0], lambda x: (col_map or {}).get(x, x)):
                            v = int(v)
                    except Exception:
                        pass
                row.append(v)
            elif entry[0] == 'avg':
                s = full[entry[1]]; n = full[entry[2]]
                row.append(None if not n else s / n)
        out.append(tuple(row))

    having = tree.args.get('having')
    if having is not None:
        out = wdb_sql._apply_having(out, proj, having.this, lambda x: x)
    out = wdb_sql._apply_order(out, proj, tree.args.get('order'))
    lim = wdb_sql._limit(tree); off = wdb_sql._offset(tree)
    if lim is not None or off:
        out = out[off: off + lim] if lim is not None else out[off:]
    return out, [wdb_sql._alias(p) for p in proj]

def _duck_from(base_tree, parquet):
    t = copy.deepcopy(base_tree)
    t.find(E.From).this.replace(E.Table(this=E.Literal.string(parquet)))
    return t.sql(dialect='duckdb')
