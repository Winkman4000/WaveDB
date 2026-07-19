"""wdb_qmem: query-scoped memory -- Jackson's law.

A query may use any RAM it needs WHILE it runs. When the outermost run() returns,
it is as if the query was never there: every structure derived from data dies --
decompressed codes, decoded dictionaries, resident arrays, pair structures,
sidecar loads, count memos. Nothing data-derived persists in RAM or on disk to
make the next query easier. Cold is the truth; the file is the only memory.

What stays, and why it is lawful:
  - the np.memmap of the .wdb: it IS the file (OS-owned pages, evicted under
    pressure, identical treatment to every other process reading any file)
  - catalog + column layout metadata: file-shape, kilobytes, not data
  - compiled kernels (numba, exprjit): program, not data

Module caches register their dict here at import; Segment arrays are dropped via
drop_derived(). db.run() flushes at depth 0 so recursive runs (subquery rewrites,
CTE flattening) share memory within ONE outer query and forget together."""

_REGISTRY = []


def register(d):
    """Register a module-level cache dict for end-of-query clearing."""
    _REGISTRY.append(d)
    return d


def flush(db=None):
    """End of the outermost query: forget everything data-derived."""
    for d in _REGISTRY:
        d.clear()
    if db is not None:
        for _p, ent in list(getattr(db, '_seg_cache', {}).items()):
            ent[1].drop_derived()
        getattr(db, '_ptr_cache', {}).clear()
