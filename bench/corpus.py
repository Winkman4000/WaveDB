"""Canonical query corpus generator. SINGLE SOURCE OF TRUTH for the unified benchmark
corpus across both datasets (TPC-H lineitem via our matrix; ClickBench hits).

Inputs:
  - bench/catalog.py QUERIES         (our matrix, ids M1..M30, table=lineitem)
  - bench/clickbench_queries.sql     (ClickBench Q0..Q42, table=hits)

Dedup: queries collapse to unique SHAPES at MEDIUM grain -- same agg set, distinct
flavor, group arity bucket (0/1/2+), filtered?, top-K?, join?, having. A shape that
both suites exercise is tagged 'both' (same shape proven on two datasets = robustness).

Generates:
  - docs/corpus.md     canonical numbered shapes, benches tag, concrete per-suite SQL
  - docs/crosswalk.md  canonical# <-> M#/Q# provenance, both directions

Run: PYTHONPATH=src python3 bench/corpus.py
"""
import os, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'src'))
import sqlglot, sqlglot.expressions as E
from catalog import QUERIES

AGGS = ('Count', 'Sum', 'Avg', 'Min', 'Max')

def _clickbench():
    raw = open(os.path.join(HERE, 'clickbench_queries.sql')).read()
    out = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith('--'):
            continue
        out.append(s.rstrip(';'))
    return out

def feats(sql):
    t = sqlglot.parse_one(sql)
    aggs = set(); cdist = False
    for n in t.find_all(E.AggFunc):
        nm = type(n).__name__
        if nm in AGGS: aggs.add(nm.upper())
    for n in t.find_all(E.Count):
        if isinstance(n.this, E.Distinct): cdist = True
    g = t.args.get('group'); ga = len(g.expressions) if g else 0
    return dict(aggs=frozenset(aggs), cdist=cdist, seldist=t.args.get('distinct') is not None,
                ga=ga, having=t.args.get('having') is not None, order=t.args.get('order') is not None,
                limit=t.args.get('limit') is not None, offset=t.args.get('offset') is not None,
                joins=len(t.args.get('joins') or []), haswhere=t.args.get('where') is not None)

def medium(f):
    return (f['aggs'], f['cdist'], f['seldist'], '2+' if f['ga'] >= 2 else str(f['ga']),
            f['having'], (f['order'] or f['limit']), f['joins'] > 0, f['haswhere'])

def bucket(f):
    if f['joins'] > 0: return (8, 'Join')
    if f['cdist'] or f['seldist']:
        return (6, 'DISTINCT / COUNT(DISTINCT) — grouped') if f['ga'] > 0 else (5, 'DISTINCT / COUNT(DISTINCT)')
    if not f['aggs'] and (f['order'] or f['limit']): return (7, 'Projection top-K')
    if f['ga'] > 0: return (4, 'GROUP BY + filter') if f['haswhere'] else (3, 'GROUP BY')
    if f['haswhere']: return (2, 'Filtered aggregate')
    if f['aggs']: return (1, 'Whole-table aggregate')
    return (9, 'Other')

def label(members, f):
    ours = [m for m in members if m['src'] == 'ours']
    if ours: return ours[0]['name']
    parts = []
    if f['cdist']: parts.append('COUNT(DISTINCT)')
    if f['aggs']: parts.append('+'.join(sorted(f['aggs'])))
    if f['seldist']: parts.append('DISTINCT')
    if not parts: parts.append('projection')
    s = ' '.join(parts)
    if f['ga'] > 0: s += f", GROUP BY {'2+' if f['ga'] >= 2 else f['ga']}"
    if f['haswhere']: s += " +WHERE"
    if f['order'] or f['limit']: s += " +topK"
    if f['having']: s += " +HAVING"
    if f['offset']: s += " +OFFSET"
    return s

def shape_str(f):
    bits = []
    bits.append('aggs={%s}' % (','.join(sorted(f['aggs'])) or '—'))
    if f['cdist']: bits.append('COUNT(DISTINCT)')
    if f['seldist']: bits.append('SELECT DISTINCT')
    bits.append('group=%s' % ('2+' if f['ga'] >= 2 else f['ga']))
    bits.append('where=%s' % ('y' if f['haswhere'] else 'n'))
    bits.append('topK=%s' % ('y' if (f['order'] or f['limit']) else 'n'))
    if f['having']: bits.append('HAVING')
    if f['joins']: bits.append('JOIN')
    return '; '.join(bits)

def build():
    recs = [{'src': 'ours', 'id': f'M{i}', 'name': n, 'sql': s, 'f': feats(s)}
            for i, (c, n, s, e) in enumerate(QUERIES, 1)]
    recs += [{'src': 'clickbench', 'id': f'Q{i}', 'name': '', 'sql': s, 'f': feats(s)}
             for i, s in enumerate(_clickbench())]
    clusters = {}
    for r in recs:
        clusters.setdefault(medium(r['f']), []).append(r)
    entries = []
    for sig, members in clusters.items():
        f = members[0]['f']; srcs = {m['src'] for m in members}
        tag = 'both' if srcs == {'ours', 'clickbench'} else ('ours' if srcs == {'ours'} else 'clickbench')
        entries.append({'bucket': bucket(f), 'label': label(members, f), 'shape': shape_str(f),
                        'tag': tag, 'members': members,
                        'M': [m['id'] for m in members if m['src'] == 'ours'],
                        'Q': [m['id'] for m in members if m['src'] == 'clickbench']})
    rank = {'both': 0, 'ours': 1, 'clickbench': 2}
    entries.sort(key=lambda e: (e['bucket'][0], rank[e['tag']], e['label'].lower()))
    for i, e in enumerate(entries, 1):
        e['cid'] = f'C{i:02d}'
    return entries

SRC_LABEL = {'ours': 'ours · lineitem', 'clickbench': 'clickbench · hits'}

def write_corpus(entries, path):
    L = []
    both = sum(1 for e in entries if e['tag'] == 'both')
    nq = sum(len(e['members']) for e in entries)
    L.append("# WaveDB canonical query corpus\n")
    L.append(f"The unified benchmark corpus across two datasets. {nq} concrete queries "
             f"(30 from our TPC-H matrix, 43 from ClickBench) collapse to **{len(entries)} unique "
             f"shapes** at medium grain. A shape tagged **both** is exercised by both suites on "
             f"different data — the same operation proven on two schemas.\n")
    L.append("Generated by `bench/corpus.py` from `bench/catalog.py` (M#) and "
             "`bench/clickbench_queries.sql` (Q#). Provenance map: `docs/crosswalk.md`.\n")
    L.append(f"**{both} shapes are `both`** · {sum(1 for e in entries if e['tag']=='ours')} ours-only · "
             f"{sum(1 for e in entries if e['tag']=='clickbench')} clickbench-only.\n")
    cur = None
    for e in entries:
        if e['bucket'][1] != cur:
            cur = e['bucket'][1]; L.append(f"\n## {cur}\n")
        L.append(f"### {e['cid']} · {e['label']}")
        L.append(f"`benches: {e['tag']}`  ·  _{e['shape']}_\n")
        for m in e['members']:
            L.append(f"- **{m['id']}** ({SRC_LABEL[m['src']]})")
            L.append(f"  ```sql\n  {m['sql']}\n  ```")
        L.append("")
    open(path, 'w').write('\n'.join(L) + '\n')

def write_crosswalk(entries, path):
    L = []
    L.append("# Benchmark crosswalk\n")
    L.append("Maps canonical corpus shapes (`docs/corpus.md`) to their source benchmark "
             "queries. `M#` = our TPC-H matrix (`bench/catalog.py`). `Q#` = ClickBench "
             "(`bench/clickbench_queries.sql`, Q0..Q42).\n")
    L.append("## Canonical → source\n")
    L.append("| canonical | benches | shape | matrix | clickbench |")
    L.append("|---|---|---|---|---|")
    for e in entries:
        L.append(f"| {e['cid']} | {e['tag']} | {e['label']} | "
                 f"{', '.join(e['M']) or '—'} | {', '.join(e['Q']) or '—'} |")
    # reverse maps
    m2c = {}; q2c = {}
    for e in entries:
        for m in e['M']: m2c[m] = e['cid']
        for q in e['Q']: q2c[q] = e['cid']
    L.append("\n## Matrix (M#) → canonical\n")
    L.append("| " + " | ".join(f"M{i}" for i in range(1, len(QUERIES)+1)) + " |")
    L.append("|" + "---|" * len(QUERIES))
    L.append("| " + " | ".join(m2c.get(f'M{i}', '?') for i in range(1, len(QUERIES)+1)) + " |")
    cb_n = len(_clickbench())
    L.append("\n## ClickBench (Q#) → canonical\n")
    L.append("| " + " | ".join(f"Q{i}" for i in range(cb_n)) + " |")
    L.append("|" + "---|" * cb_n)
    L.append("| " + " | ".join(q2c.get(f'Q{i}', '?') for i in range(cb_n)) + " |")
    open(path, 'w').write('\n'.join(L) + '\n')

if __name__ == '__main__':
    entries = build()
    docs = os.path.join(HERE, '..', 'docs')
    write_corpus(entries, os.path.join(docs, 'corpus.md'))
    write_crosswalk(entries, os.path.join(docs, 'crosswalk.md'))
    print(f"wrote docs/corpus.md and docs/crosswalk.md  ({len(entries)} canonical shapes)")
