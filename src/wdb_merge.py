"""WaveDB two-tier merge-read: combine a cold segment + a hot buffer into one answer.

The hard part is aggregates: you cannot average two averages. So we push MERGEABLE
PARTIALS to each tier (AVG -> SUM+COUNT, SUM->SUM, MIN/MAX, COUNT(*)->count), strip
HAVING/ORDER/LIMIT for the per-tier queries, merge partials per group key, then
reconstruct (AVG = sum/count) and apply HAVING/ORDER/LIMIT once on the merged result.

Cold tier is queried by our own engine; hot tier (a small uncompressed parquet) by
DuckDB. The novel WaveDB value lives entirely in the cold tier.
"""
import copy, sqlglot, sqlglot.expressions as E
import duckdb
import wdb_sql
from wdb_engine import Segment

_AGG = (E.Count, E.Sum, E.Min, E.Max, E.Avg)

def _inner(p):
    return p.this if isinstance(p, E.Alias) else p

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
                if op in ('sum', 'count_star'):
                    cur[slot] = _num(cur[slot]) + _num(val)
                elif op == 'min':
                    cur[slot] = val if cur[slot] is None else (cur[slot] if val is None else min(cur[slot], val))
                elif op == 'max':
                    cur[slot] = val if cur[slot] is None else (cur[slot] if val is None else max(cur[slot], val))
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

def merge_query(segs, hot_parquet, sql):
    """segs: list of cold Segments (0..N). hot_parquet: path or None."""
    if segs is None: segs = []
    tree = sqlglot.parse_one(sql, read='duckdb')
    if not isinstance(tree, E.Select): raise NotImplementedError("merge: only SELECT")
    if tree.args.get('joins') or tree.args.get('with'):
        raise NotImplementedError("merge: JOIN/CTE not supported")
    proj = tree.expressions
    group = tree.args.get('group')
    has_group = group is not None
    has_agg = any(isinstance(_inner(p), _AGG) for p in proj)

    # ---- no aggregates, no GROUP BY: plain row union ----
    if not has_agg and not has_group:
        rows = []
        # strip ORDER/LIMIT for per-tier; apply post-merge
        base = copy.deepcopy(tree)
        base.set('order', None); base.set('limit', None)
        bsql = base.sql(dialect='duckdb')
        for seg in segs:
            r, _ = wdb_sql.execute(seg, bsql); rows += list(r)
        if hot_parquet is not None:
            con = duckdb.connect()
            rows += [tuple(x) for x in con.execute(_duck_from(base, hot_parquet)).fetchall()]
        rows = wdb_sql._apply_order(rows, proj, tree.args.get('order'))
        lim = wdb_sql._limit(tree)
        if lim is not None: rows = rows[:lim]
        return rows, [wdb_sql._alias(p) for p in proj]

    # ---- aggregates / GROUP BY: partial merge ----
    keys, plan, partials = _classify(proj)
    row_lists = []
    if segs:
        psql = _partial_sql(tree, "tbl", keys, partials)
        for seg in segs:
            r, _ = wdb_sql.execute(seg, psql); row_lists.append(list(r))
    if hot_parquet is not None:
        psql_h = _partial_sql(tree, f"'{hot_parquet}'", keys, partials)
        con = duckdb.connect()
        row_lists.append([tuple(x) for x in con.execute(psql_h).fetchall()])
    acc = _merge_partials(row_lists, len(keys), plan)

    out = []
    for key, parts in acc.items():
        full = list(key) + list(parts)
        row = []
        for entry in plan:
            if entry[0] == 'key':
                row.append(key[entry[1]])
            elif entry[0] in ('sum', 'count_star', 'min', 'max'):
                row.append(full[entry[1]])
            elif entry[0] == 'avg':
                s = full[entry[1]]; n = full[entry[2]]
                row.append(None if not n else s / n)
        out.append(tuple(row))

    having = tree.args.get('having')
    if having is not None:
        out = wdb_sql._apply_having(out, proj, having.this, lambda x: x)
    out = wdb_sql._apply_order(out, proj, tree.args.get('order'))
    lim = wdb_sql._limit(tree)
    if lim is not None: out = out[:lim]
    return out, [wdb_sql._alias(p) for p in proj]

def _duck_from(base_tree, parquet):
    t = copy.deepcopy(base_tree)
    t.find(E.From).this.replace(E.Table(this=E.Literal.string(parquet)))
    return t.sql(dialect='duckdb')
