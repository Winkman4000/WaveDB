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


_TIER1 = []


def register_tier1(d):
    """A module cache of SOURCE DATA decoded (tier 1): outlives the query for the hot runs, dies
    with it under WDB_HOT_KEEP=0 (the pure-cold A/B). Never a query's computed result."""
    _TIER1.append(d)
    return d


def flush(db=None):
    """End of the outermost query: forget everything data-derived."""
    for d in _REGISTRY:
        d.clear()
    if __import__('os').environ.get('WDB_HOT_KEEP', '1') == '0':
        for d in _TIER1:
            d.clear()
    if db is not None:
        for _p, ent in list(getattr(db, '_seg_cache', {}).items()):
            ent[1].drop_derived()
        for a in ('_ptr_cache', '_gd_cache', '_union_cache', '_dc_ctx', '_uniq_memo'):
            d = getattr(db, a, None)
            if hasattr(d, 'clear'):
                d.clear()
    try:
        import wdb_shelf
        wdb_shelf.SHELF.keep_only(wdb_shelf.vocabulary())    # tier 1 only; nothing under WDB_HOT_KEEP=0
    except Exception:
        pass
    if __import__('os').environ.get('WDB_QMEM_STRICT') and db is not None:
        left = audit(db)
        if left:
            raise AssertionError('JACKSON\'S LAW (wdb_qmem): data-derived state outlived the query: %s' % left[:8])


# Module-level containers that are NOT data (program, catalog, planning keyed by the catalog stamp,
# locks, settings, the ledger, open files). Everything else that is still full after a query's end
# is residue. Registered caches are cleared by flush and so never appear.
_MODULE_KEEP = frozenset({
    ('controller', '_BYNAME'), ('wdb_exprjit', '_CACHE'), ('wdb_db', '_PW_SEGS'),
    ('wdb_db', '_ROAD_DECLINED'), ('wdb_join', '_JPTR_ASKED'), ('wdb_join', '_JPTR_NOT'),
    ('wdb_ledger', '_STAGES'), ('wdb_ledger', '_OPEN'), ('wdb_semijoin', '_INV_LOCKS'),
    ('wdb_sidecar', '_SETTING_CACHE'), ('wdb_sidecar', '_SENTINEL'), ('wdb_blockstats', '_LOADED'),
    ('wdb_heavypair', '_DISABLED'), ('wdb_qmem', '_REGISTRY'), ('wdb_qmem', '_MODULE_KEEP'),
    ('wdb_qmem', '_BASE'), ('wdb_sidecar', '_stamp_cache'), ('wdb_db', '_PROGRAM'),
    ('wdb_sidecar', '_exists_cache'),        # which sidecar FILES exist: the directory, not data
    ('wdb_calib', '_CACHE'),                 # the machine card: this hardware's primitive speeds
    ('controller', '_PLANS'), ('wdb_semijoin', '_PLANS'), ('wdb_semijoin', '_KMAX'),   # plans: program
    ('wdb_qmem', '_TIER1'),
})


_MODULE_KEEP_ALL = frozenset({'wdb_sidecar', 'wdb_ledger'})   # files, settings, the diary: never data

_BASE = {}                           # (module, name) -> size when the program finished loading


def _module_containers():
    import sys
    from collections import OrderedDict
    for mn, m in list(sys.modules.items()):
        if m is None or not (mn.startswith('wdb_') or mn in ('controller', 'read_methods')):
            continue
        for k, v in list(vars(m).items()):
            if k.startswith('_') and not k.startswith('__') and isinstance(v, (dict, set, OrderedDict)):
                yield mn, k, v


def baseline():
    """At the database's start, after the program is loaded and before any query: every module
    container present now is PROGRAM (constant tables, registries). Only growth past this is residue."""
    if not _BASE:
        for mn, k, v in _module_containers():
            _BASE[(mn, k)] = len(v)


def audit(db):
    """THE LAW'S WITNESS: what data-derived state is still alive now. Empty after every query's end
    is the law; WDB_QMEM_STRICT makes flush raise on anything listed. Returns [(where, name, size)]."""
    import sys
    from collections import OrderedDict
    out = []
    try:
        import wdb_engine
        keep = wdb_engine.seg_keep(); ckeep = wdb_engine.col_keep()
    except Exception:
        keep = frozenset(); ckeep = frozenset()
    for _p, ent in list(getattr(db, '_seg_cache', {}).items()):
        seg = ent[1]
        for k, v in seg.__dict__.items():
            if k.startswith('_') and k not in keep and isinstance(v, (dict, set, list)) and len(v):
                out.append(('segment', k, len(v)))
        for nm, c in seg.cols.items():
            for k in c:
                if k.startswith('_') and k not in ckeep:
                    out.append(('column', '%s.%s' % (nm, k), 1))
    for a in ('_ptr_cache', '_gd_cache', '_union_cache', '_dc_ctx', '_uniq_memo'):
        d = getattr(db, a, None)
        if d:
            out.append(('database', a, len(d)))
    for mn, k, v in _module_containers():
        if len(v) > _BASE.get((mn, k), 0) and (mn, k) not in _MODULE_KEEP \
                and mn not in _MODULE_KEEP_ALL and not any(v is t for t in _TIER1):
            out.append(('module', '%s.%s' % (mn, k), len(v)))
    try:
        import wdb_shelf
        for key, it in list(wdb_shelf.SHELF._items.items()):
            if it[2] not in wdb_shelf.vocabulary():
                out.append(('shelf', it[2], 1))
    except Exception:
        pass
    return out
