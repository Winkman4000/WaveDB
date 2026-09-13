"""Assemble results.json in the ClickBench shape from load.sh and run.sh output.
usage: python3 benchmark/clickbench/results.py LOAD_LOG RUN_LOG [machine] > results.json"""
import sys, json, re, datetime
load = open(sys.argv[1]).read(); run = open(sys.argv[2]).read().strip()
lt = re.search(r'load time: ([0-9.]+)', load); ds = re.search(r'storage bytes: ([0-9]+)', load)
res = {
    "system": "WaveDB", "date": datetime.date.today().isoformat(),
    "machine": sys.argv[3] if len(sys.argv) > 3 else "unknown",
    "cluster_size": 1, "proprietary": "no", "tuned": "no",
    "comment": "dictionary-coded columnar store; queries via the wdb server + thin client; time-clustered load with declared casts",
    "tags": ["Python", "column-oriented", "embedded"],
    "load_time": float(lt.group(1)) if lt else None,
    "data_size": int(ds.group(1)) if ds else None,
    "result": json.loads(run),
}
print(json.dumps(res, indent=2))
