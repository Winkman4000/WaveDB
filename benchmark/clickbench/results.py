"""Assemble results/YYYYMMDD/<machine>.json in the ClickBench shape from the shared driver's log (the output of
benchmark.sh: "Load time: ...", one "[t1,t2,t3]," line per query, "Data size: ...", "Concurrent QPS: ...",
"Concurrent error ratio: ...") and template.json.
usage: python3 results.py BENCHMARK_LOG MACHINE [YYYY-MM-DD] > c6a.4xlarge.json"""
import sys, os, json, re, datetime

log = open(sys.argv[1]).read()
machine = sys.argv[2]
date = sys.argv[3] if len(sys.argv) > 3 else datetime.datetime.utcnow().date().isoformat()
tpl = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'template.json')))


def num(pat, cast):
    m = re.search(pat, log)
    return None if (m is None or m.group(1) == 'null') else cast(m.group(1))


rows = [json.loads(m.group(1)) for m in re.finditer(r'^(\[[^\]\n]*\]),\s*$', log, re.M)]
assert len(rows) == 43, ('expected 43 query rows in the log, found', len(rows))
res = {'system': tpl['system'], 'date': date, 'machine': machine, 'cluster_size': 1,
       'proprietary': tpl['proprietary'], 'hardware': tpl['hardware'], 'tuned': tpl['tuned'], 'tags': tpl['tags'],
       'load_time': num(r'Load time: ([0-9.]+)', lambda s: round(float(s))),
       'data_size': num(r'Data size: ([0-9]+)', int),
       'concurrent_qps': num(r'Concurrent QPS: ([0-9.]+|null)', float),
       'concurrent_error_ratio': num(r'Concurrent error ratio: ([0-9.]+|null)', float),
       'result': rows}
out = json.dumps(res, indent=4)
out = re.sub(r'\[\s+([0-9.]+|null),\s+([0-9.]+|null),\s+([0-9.]+|null)\s+\]', r'[\1, \2, \3]', out)   # a row per line
print(out)
