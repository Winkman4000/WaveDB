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
