s = open('/home/jack/WaveDB/src/wdb_join.py').read()
A1 = "            conj9 = _flat9(where.this)\n"
i0 = s.index(A1) + len(A1)
iB = s.index("print('JOIN BILL: PARTITION pre-pass")
i1 = s.index("\n", iB) + 1
seg = s[i0:i1]
# indentation: seg begins inside "if where...try:" (12 sp) but the bill print
# tail sits at 8 sp (outside the try) -- check and normalize: find the tail
tail_anchor = "            if _bill9 is not None:\n                print('JOIN BILL: PARTITION"
assert tail_anchor in seg
# delete the regather block (keys will be BORN at survivor scale post-hoist)
r0 = seg.index("                    # keys were EXTRACTED full-width")
r1 = seg.index("gk9h['full'])[rows9]\n") + len("gk9h['full'])[rows9]\n")
assert r0 < r1
seg = seg[:r0] + seg[r1:]
# lift the env gate
seg = seg.replace("""                    and not _where_spent
                    and os.environ.get('WDB_SURVIVOR_HANDOFF')):""",
                  "                    and not _where_spent):", 1)
assert "WDB_SURVIVOR_HANDOFF" not in seg
# original site: consume hoisted residuals
repl_orig = "            if _resid_c9X is not None:\n                conj9 = _resid_c9X\n"
s = s[:i0] + repl_orig + s[i1:]
# subsumed line at original site
s = s.replace("""            if _plane_mask9 is not None or rows9 is not None:
                conj9 = _resid_c9
""", "", 1)
# insert the hoisted block before _rw9
H = "    def _rw9(a):\n"
assert s.count(H) == 1
hoist = (
    "    _plane_mask9 = None\n"
    "    _resid_c9X = None\n"
    "    # THE HOISTED PARTITION (Jackson's structural ruling): serves and the\n"
    "    # survivor handoff run BEFORE anything row-aligned exists, so keys,\n"
    "    # slots and roads are BORN at survivor scale -- no retrofits.\n"
    "    if where is not None and not _where_spent and rows9 is None:\n"
    "        try:\n"
    "            def _flatH(nH):\n"
    "                if isinstance(nH, E.Paren): return _flatH(nH.this)\n"
    "                if isinstance(nH, E.And):\n"
    "                    return _flatH(nH.this) + _flatH(nH.expression)\n"
    "                return [nH]\n"
    "            conj9 = _flatH(where.this)\n"
    + seg +
    "            _resid_c9X = _resid_c9\n"
    "        except Exception:\n"
    "            _plane_mask9 = None; _resid_c9X = None; rows9 = None\n"
)
s = s.replace(H, hoist + H, 1)
open('/home/jack/WaveDB/src/wdb_join.py', 'w').write(s)
print('HOISTED')
