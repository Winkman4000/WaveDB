"""
wdb_policies -- named, reusable decision guards ("policies") that gate which operator answers a query.

Phoenix-plug model: each policy is a tiny PURE predicate. Given the query shape, it returns True if the
query PASSES the guard (may proceed) or False if it does not match (the operator halts and the router
tries the next path). The operator's actual compute lives elsewhere; these only decide eligibility --
separating the protocol (who may run) from the mechanism (how the answer is computed).

Two rules keep this honest (the same constraint an Elixir guard clause has):
  1. A policy tests ONLY cheap, static facts -- knowable from the parsed query tree and column metadata
     (shape, types, segment count, nullability, pending deletes). No scanning, no counting, no decode.
  2. A cost that can only be known by touching data (e.g. "does the packed pair-id overflow int64?") is
     NOT a policy. The operator tests that internally and falls through. Forcing a measured cost into a
     policy would just rebuild the maze with extra steps.

Naming reads as the condition that must be TRUE to proceed: no_where -> "there is no WHERE clause".
"""
import wdb_sql
E = wdb_sql.E


# --- shared query-shape guards (read the parsed tree only) -- reused across many operators ---

def no_joins(tree):
    """No JOIN in the query (single-table operators only)."""
    return not tree.args.get('joins')

def no_select_distinct(tree):
    """Not a SELECT DISTINCT (row de-duplication is a different shape)."""
    return tree.args.get('distinct') is None

def no_having(tree):
    """No HAVING clause (post-aggregate filter is a different shape)."""
    return tree.args.get('having') is None

def no_where(tree):
    """No WHERE clause (filter-free operators only)."""
    return tree.args.get('where') is None

def has_where(tree):
    """There IS a WHERE clause -- the filtered operators require one (the inverse of no_where)."""
    return tree.args.get('where') is not None

def single_group_key(tree):
    """Exactly one GROUP BY column."""
    g = tree.args.get('group')
    return g is not None and len(g.expressions) == 1

def has_limit(tree):
    """There IS a LIMIT -- bounded top-N shapes only."""
    return wdb_sql._limit(tree) is not None

def has_group_key(tree):
    """At least one GROUP BY column. The looser form of single_group_key (==1), for multi-key
    operators that group by several columns at once."""
    g = tree.args.get('group')
    return g is not None and len(g.expressions) >= 1


# --- shared segment / column guards (read column metadata only) ---

def columns_exist(seg, *cols):
    """Every named (physical) column is present in the segment."""
    return all(c in seg.cols for c in cols)

def no_deleted_rows(seg):
    """No pending DELETEs -- a presence mask would make a per-row scan unsound for these operators."""
    return seg.presence_mask() is None

def key_not_nullable(seg, col):
    """The column is declared non-nullable (NULL-as-its-own-group is a deferred case)."""
    return not seg.cols[col].get('has_null')


def not_positional(seg, col):
    """The column is not positional (mode-4) encoded, so it has value-identity codes. NOTE: necessary
    but not sufficient for full decodability -- the operator still confirms its by-code decoder exists,
    which is a measured/edge check that stays internal."""
    return seg.cols[col]['mode'] != 4
