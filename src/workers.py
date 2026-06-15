"""
workers.py -- THE WORKERS LAYER.

A *worker* transforms what a read produced into the returned result. In your model:
stored data -> a READ retrieves a pattern -> a WORKER transforms it. Reads live in
read_methods; the controller plugs read -> worker; the transforms live here.

Today the workers are the post-read finalizers -- HAVING, ORDER BY, LIMIT/OFFSET --
which every read was re-invoking inline. `finalize` composes them in SQL order. Like
read_methods over the operator machinery, this layer is thin: the implementations
live in wdb_sql (the shared SQL utilities); workers.py is the named home the reads
call so "what can I do to a read's output" is answerable in one place.
"""
import wdb_sql


def finalize(rows, proj, order, lim, having=None, seg_col=None):
    """The standard post-read finalize: HAVING, then ORDER BY, then LIMIT -- SQL order.
    `order` may be None (no-op). `lim` None means no limit. `having` is the HAVING
    predicate node (or None). Returns the transformed rows."""
    if having is not None:
        rows = wdb_sql._apply_having(rows, proj, having, seg_col)
    rows = wdb_sql._apply_order(rows, proj, order)
    if lim is not None:
        rows = rows[:lim]
    return rows


def take_sorted(decode, ncodes, lim, skip):
    """Pop a value-sorted dictionary from the front -- ascending code == ascending value,
    since the dict is stored sorted -- skipping `skip` codes (e.g. the null code) and any
    value that decodes to '' (the filtered-out empty string), emitting the first `lim`
    values. decode(code) -> value, called only for the handful of codes inspected. The
    read supplies the sorted structure; this pops it. No row scan, no counts: the dict
    order IS the sort."""
    rows = []
    code = 0
    while code < ncodes and len(rows) < lim:
        if code in skip:
            code += 1; continue
        v = decode(code)
        if v == '' or v is None:
            code += 1; continue
        rows.append((v,))
        code += 1
    return rows
