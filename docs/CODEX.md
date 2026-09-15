# THE ENGINE CODEX (seed — populated from code, never from memory)
Jackson's ruling: every part of the codebase accounted for and known for
where it adds contribution, so changes target a NAMED LAYER and tests
reflect consequences of that layer.

## THE LAYERS (taxonomy to populate, one module at a time, by READING each)
1. ENCODING LAYER — wdb_encode: modes (0..6+), dresses (enc-0..13), elections, seals, frame law
2. STORAGE/METADATA — wdb_engine.Segment (passport: V, N, BR, dt, stairs), wdb_catalog, sidecars (plists, jptr pointers, gbc/gbp)
3. READS LAYER — read_methods.py (named!), engine reads: codes/codes_band/codes_at/_raw_codes/_e13_band, blockstats
4. QUERY IDENTIFICATION / ROUTING — controller.py (the consult chain), commands.py
5. PLANNING / COURTS — per-shape lanes: funnel, join (chains/hash/sidecars/potency), lenagg, sampletop, compound (SURVIVOR-SET — already exists!), coscan, countpos, clustertopk, bsi, affine*, cdgroup, cte...
6. EMIT/RESULTS — colresult, sql tail (_apply_order/having), top-k emits
7. REFEREE/GATES — tests/ (1692), megaboard, board_clickbench, board_tpch, duck referees

## KERNELS ARE NOT A LAYER (Jackson's correction)
Kernels are a SPEED CLASS, not a place: a kernel can be a read, a fold,
a sort -- it belongs to whatever layer's stage it compiles. wdb_kernels,
wdb_exprjit, wdb_radix, bsi_kernels are SHARED ARMORIES (cross-cutting
infrastructure, accounted for in the census but not pipeline rungs).
Every organ's codex entry must state WHICH of its stages run at kernel
speed and which run in interpreted glue -- that attribute is the era's
recurring diagnosis (lenagg: 4ms fold in 320ms glue; Q3: 62ms kernel in
2000ms glue). A new kernel is never an architectural change by itself;
the architectural fact is which layer's stage got compiled.

## MEASURED-VERDICTS TABLE (append-only; warm wall, commit, date)
| query | before | after | commit | note |
|---|---|---|---|---|
| TPCH Q3 | 23.0s | 2.00s | 202c59b | arc: uniq-meta, jptr sidecars, sorted-run, mono-acc; kernel=62ms; REMAINING: emit 475 (topk gate), assemble 360, prework 133, ~950ms UNMEASURED in db.run dispatch |
| TPCH Q3 | 2.00s | 1.34s | 63441ba | BALANCED BILL era: setup bracket found 885ms never measured; leading-run court killed the 60M co-key gathers (2003->1820); numeric-emit + top-k gate fired, emit 473->32 (1820->1340). Bill now: setup 733, assemble 350, kernel 60, emit 32, prework 131. Next: sub-bracket setup; assemble LUTs. |

## STANDING RULES (the anti-slip laws)
- BALANCE LAW: no strike while the bill's stages don't sum to ~the wall.
- SCOPE LAW: a principle names WHICH LAYER it restructures before code is touched ("changes which rows exist" vs "which test runs first").
- CENSUS LAW: scope existing organs (this codex + census) before building anything.
- VERDICT LAW: the only progress number is warm wall before vs after, written here.


## VERDICTS 2026-08-24 (the cascade session)
| query | before | after | commit | note |
|---|---|---|---|---|
| TPCH Q3 | 1.34s | 1.004s | 30b34bf | THE SURVIVOR CASCADE LANDED: 59,986,052 -> 302,114 rows printed on the bill (duck-identical survivor count); kernel 2ms, emit 21ms. Remaining: cascade passes 540 (kernelize first full-width LUT pass), setup 180, pre-work 130. |
| megaboard | 3.57x | 3.63x RECORD | 30b34bf | 103/103, zero wrong, 89 wins, cascade aboard |
| TPCH board | 1 wrong | ZERO WRONG | 30b34bf | Q18 fail-open FIXED (3 sites): subquery-IN became isin([])=all-False=silent empty. LAW ENGRAVED: fail closed means fail LOUD, never empty. Q18 now an honest hole. |
| TPCH Q3 | 1.004s | 0.819s | 33db9e3 | THE HOME-TABLE LAW: predicates judged at home in PARALLEL, verdicts flow FK roads down, fact pays one gather per road. Cascade 540->360. Remaining: cascade 360 (fuse fact pass), setup 178, assemble 143, prework 130. |
| REGRESSION FLAG | - | - | - | Q1 0.44->0.83s and Q6 0.31->0.50 vs board v1 (different pod; duck moved 0.44->0.37 only). INVESTIGATE next session: single-table fused_agg era changes. |

## NEXT SESSION — THE SURVIVOR CASCADE (single task, Jackson-ratified — DONE, kept for history)
SCOPE: restructures the join court pipeline — changes WHICH ROWS EXIST
after the filter stage. Not a trim.
1. FILTER FIRST, code space, potency-descending: shipdate keep-LUT over
   sequential codes -> gather pointers ONLY for survivors -> orderdate
   LUT -> 2-hop mktsegment -> ~3M survivor indices. No values decoded.
2. THEN setup at survivor scale: group-key codes sliced at survivors,
   gids on survivors, slots as V-LUTs (decode spec).
3. Kernel on survivors, pred spent. Emit decodes k.
Gates: suite + megaboard + TPCH board, exactness vs duck, warm-wall
verdict vs 1.34s written to the table. Projected band: 300-500ms.


## NEXT SESSION DOCKET (post home-table-law, Jackson-ratified)
1. THE RUN-ROAD LAW: monotone pointer roads read as RUNS, not
   addresses -- the cascade fact pass becomes repeat(verdicts,
   run_lengths): zero gathers, zero pointer bytes. (~90ms of Q3)
2. POINTER WIDTH ELECTION: u16/u32 by parent size at sidecar birth
   (naked int64 = 2-3x bytes); elegant end-state: .jptr as a
   dressed wdb column under the encoder own elections.
3. setup 178 + assemble 143 sub-bills; prework 130.
4. REGRESSION FLAG (standing): Q1 0.44->0.83, Q6 0.31->0.50 vs
   board v1 -- bill single-table fused_agg BEFORE new blades.
VERDICT 2026-08-25: Q1 regression CURED 869->503 exact (643f0e3, the
marginal-bound gate: exact census product decides cascade firing);
Q3 788 unharmed.
Baselines: Q3 0.819s (33db9e3) | megaboard 90 wins zero-wrong |
TPCH zero-wrong, 3 ok, 11 honest holes.


## VERDICTS 2026-08-25 (morning)
| item | verdict |
|---|---|
| Q1 regression | CURED 869->503 exact (643f0e3): THE MARGINAL-BOUND GATE -- expected survivors from exact census product decides cascade firing (fires iff <=50% survive). Q3 unharmed. |
| run-road law | TRIED 3 FORMS, MEASURED 819 -> 1002/883/1042, REVERTED (e35240a). At Q3 shape the 60M-wide verdict-repeat loses to the 32M survivor-scale gather. Shelved with numbers; revisit only for multi-road low-survivor shapes. |
| width election | REVERTED (8f0112b): numpy indexes with intp -- narrow pointer arrays pay a cast-copy per gather (+50ms/query for 240MB disk). Index arrays are intp by law. |
Baselines now: Q3 0.832s | Q1 0.503s | both exact.


## VERDICTS 2026-08-25 (afternoon -- the stamp session)
| item | verdict |
|---|---|
| SURVIVOR-SCALE READS (52b8c96) | Q3 832 -> 699ms EXACT, REALM RECORD. Within-home potency cascade (first conjunct sequential, later ones codes_at at survivors); gkeys+slots fetch at survivor scale; K dict-wide from passport. Setup 178->38 (plain), ->1ms (stamped). |
| THE 255-SLICE SUM STAMP (Jackson) | BUILT+PROVEN: per-order sums, 255 quantile slices, u8 stamp = 25MB on disk (0.4B/row). Stamped top-10 == duck EXACT; STOP RULE PROVEN on real data (#10 rev 415,367 >= ceiling[254] 391,935 -- one slice provably suffices). Cascade reaches 1,960 rows, setup 1ms, kernel 0ms. |
| stamp speed, honest | 808 vs plain 699: the stamp own 60M pass (~150-200ms u8 decode) outweighs savings while downstream is already cheap. PAYS WHEN: (a) u8 scan kernelized (~40ms), (b) the DESCENT COURT wired (auto stamp injection + stop rule + slice descent), (c) month-prune shrinks the fine pass. All docketed. |
Baselines: Q3 0.699 EXACT (record) | Q1 0.503 | stamp aboard lineitem, ceilings in lineitem.sumslice.json, backup lineitem_0.wdb.bak intact.
NEXT: descent court + stamp-scan kernel + month-prune; boards re-gate first (launched at session end -- read /tmp/megaboard_v4_6.log + /tmp/tpchboard8.log).


## THE POTENCY LAW, COMPLETED (Jackson, 2026-08-25)
The aggregation is always the filter target. A filter has potency
only while (downstream work saved) > (its own cost) -- prune rate
alone is meaningless. When the protected work is already cheap,
NO filter earns a place. Cascade MEMBERSHIP and ORDER both follow.
Today proofs: 13ms stamp = potent vs an 18ms sum; 150ms stamp
pass = cargo; run-road repeat = negative potency; marginal-bound
gate = this law at the cascade gate.

## NEXT SESSION (single task): THE DESCENT COURT
Wire the measured 129ms floor as the join court lane: stamp scan
(13) -> confirm at survivors (98) -> survivor sum + top-k (18) ->
STOP-RULE legality check (descend a slice on failure). Engine
target 180-250 vs duck 265. Floor probe + numbers above. Gates:
boards zero-wrong first (logs /tmp/megaboard_v4_6.log,
/tmp/tpchboard8.log from launched runs).


## VERDICTS 2026-08-26 (the descent court session)
| item | verdict |
|---|---|
| THE DESCENT COURT (e3cb45f) | LIVE AND EXACT in db.run: DESCENT slice>=255, kth 415367 > ceil 391935, STOP every run; cascade 60M->1,960; falls to plain path on any mismatch. Root-caused two silent declines on the way: (a) court must consult BEFORE join_query doors (chain build MUTATES the tree, popping join eqs); (b) _FastUnsupported confessions now print under the bill flag. |
| Q3 stamped | 758ms exact vs plain 699, duck 271, floor 129. The court wins nothing YET because cascade=490 is unexplained at the stamped shape -- the floor proves ~130 is physical. |
| band scan | TRIED (f2cf454), 758->792, REVERTED. First-pass decode is NOT the 490. |
| NEXT (single task) | SUB-BILL THE STAMPED CASCADE 490 line by line (balance law) -- stamp pass / codes_at fallthrough at 405K / roads / flatnonzero / qmem re-reads -- THEN strike the named line. Also: court avoids double chain build; assemble 150 dict copies. |
Boards: megaboard_v4_7 + tpchboard9 grinding as the court gate.


## SESSION CLOSE 2026-08-26
Jackson pipeline IS the descent executor (exact, gated, fallback
safe). Numbers: executor 930 | general court 758 | plain 699 |
LAWFUL FLOOR measured piecewise ~230 (stamp 110 + confirm ~100 +
sum/topk 20) | duck 271. THE GAP IS THREE DELETIONS, named:
1. chain rebuild per query (~130) -> cache per (sql-shape, epoch)
2. shipdate confirm reads FULL column when 405K survivors scatter
   (~110+) -> confirm cheapest-per-survivor first (parent verdicts
   via pointer ~free) so shipdate confirms at ~20K not 405K
3. stamp decode 95 -> scan-class columns elect the RAW dress
   (59MB mmap, scan 6ms measured at 10.6GB/s)
Boards cancelled by order this session; next session MUST gate
(megaboard + tpch) before any further push rides on the court.


## THE ISO-FLOOR LAW (Jackson, 2026-08-26 -- binding)
The isolated cost of a pipeline IS its engine budget. If the steps
cost 230ms measured alone, the engine serving them above ~230ms is
DEFECTIVE CODE, not overhead -- there is no legitimate category of
engine time that performs none of the pipeline steps. Deviation
from the iso floor is a defect to be deleted, and the deletion is
verified only by db.run before/after. FIRST ACT NEXT SESSION: the
descent executor body becomes the probe code LITERALLY (it exists,
it measured 230-class); db.run must then print ~230-280 or the
diff gets billed line by line until it sums. No other work first.


## 2026-08-26 LATE: THE NEEDLE MOVED — ENGINE UNDER DUCK
| step | number |
|---|---|
| morning state | 769ms (fallback band, executor silently gated) |
| THE CENSUS SIDECAR (06b6ac1) | 769 -> 547. code_counts persisted as file facts; PEMDAS scoring 313 -> 15ms. Jackson diagnosed it (metadata evaluation) and split it (facts cached, decisions computed). |
| EXECUTOR UNBLOCKED | 547 -> 256 EXACT, UNDER DUCK 271. Gate G3: join equalities are ROADS not filters; skipped, edges honor them. All 16 exec gates now confess under the bill flag. |
EXEC bill: ctx+stamp 47, confirm ~100, roads 20, sum+topk+emit 97.
Iso floor 230; engine 256; remainder ~26ms = court wrapper + run dispatch.
STANDING DEBT: boards NOT re-gated today (cancelled by order) — megaboard + tpch MUST gate next session before anything rides further. Running-rule + census + executor all touch every agg join.


## BOARDS 2026-08-26 (post census+running-rule+executor) — GATES CLEAN, HISTORY MADE
MEGABOARD: 103/103, zero wrong, 89 wins, median 3.73x NEW RECORD.
TPCH: zero wrong, FIRST WINS EVER: Q1 0.36 vs duck 0.39 (1.06x),
Q3 0.28 vs duck 0.29 (1.03x). Q6 0.24 vs 0.15 next target.
Baselines: Q3 0.256-0.28 exact | Q1 0.36 | boards clean.
NEXT DOCKET: Q6 face; the 26ms wrapper; ClickBench fronts (Q30
enc-13 tenant, Q39 group-stage); TPCH holes (FROM-subquery forge
first: Q7/Q13/Q22); stamp forges for other fact tables.


## 2026-08-27: Q6 CAMPAIGN — measured verdicts
| lever | verdict |
|---|---|
| byte/bit planes | LOSES in sim (230/362ms vs 154/80): low planes are noise, unpackbits costly. |
| cross-column parallel decode | FLAT (+17ms real; -70 projection wrong: each column already runs 8 internal zstd lanes). Reverted. |
| BAND COMPARES in the jit kernel | 229 -> 245-263, REVERTED. Kernel dicts are L1-resident; b[c[i]] already compare-cheap. THE YEAR LAW LIVES WHERE MASKS MATERIALIZE FULL-WIDTH: the cascade kx[codes] passes (18 vs 70 measured there) — future strike site, not the kernel. |
Q6 stands 229 vs duck 207 this pod (0.89x). Honest gap ~22ms = decode 151 (dress-bound) + kernel 55 + dispatch.


## CORRECTION (Jackson-demanded A/B, 2026-08-27)
Q6 0.63x -> 0.89x was NOT pod ambient. Same-pod A/B: pre-census
code 574ms vs current 229ms. THE CENSUS SIDECAR (Jackson's
facts/decisions split) cut Q6 by 2.2x -- its pred-potency ordering
had been re-scanning 60M-row censuses per query. Claude attributed
the win to pod weather without testing; the untested attribution
was false. LAW REINFORCED: cross-pod deltas get an A/B before any
ambient claim.


## THE FIELD-PLANE DRESS (Jackson's design, 2026-08-27) — MEASURED CONTRACT
Dates decompose to y/m/d u8 planes (widths MEASURED from range, not
assumed: 210y fits u8, else u16). Real numbers, l_shipdate 60M:
planes 16.4+29.0+37.3 = 82.7MB vs enc-3 94.1 (SMALLER — the calendar
correlation beats the naive 85MB entropy bound); year-plane predicate
~14ms laned vs 54 today (4x); full-decode CORRECTION found mid-build:
reconstruction to DICT CODES adds an inverse-LUT gather ~+35 laned
=> ~75 vs 44 (WORSE) — the probe priced days, not codes.
THE CORRECT SHAPE: primary consumer = the PLANE-TEST READ METHOD
(year/month bands served maskless from planes, zero reconstruction);
codes() reconstruction is the fallback. ELECTION RULE (ratified):
types NOMINATE candidates (dt==3), measurements SIZE the planes,
the size election alone ELECTS; structural recognition later as
cheaper candidate pruning (lossless reconstruction is the only bar).
BUILD ORDER (one session, gated): enc-14 encoder candidate + reader
fallback; plane_test read method wired to pred/cascade for
year-aligned bands; suite+boards; lineitem re-encode last; verdicts
vs: Q6 278 baseline, year-test 14-vs-54 contract, full-decode
regression must stay confined to plane-elected columns.


## 2026-08-27/28: ENC-14 CAMPAIGN — full honest ledger
BUILT AND BANKED (all suite-green, all pushed): enc-14 framed field
planes (elections: 3 lineitem dates, -47MB, everything else held);
reconstruction kernel; THE PLANE-TEST READ (lex calendar band, no
reconstruction) at both consumers; the YEAR FAST PATH (y-plane
alone decides year-aligned bands); fork-safe kernels+cube pools;
retype tag; AND the mask-always fix -- the kernel wrappers DROPPED
a materialised mask whenever a fused pred existed, a wrongness
class the suite caught live.
BUGS KILLED EN ROUTE: thread-shared dctx corruption; fork-after-
OpenMP pool death; two inherited silent excepts now confess.
VERDICTS (the law rules): pre-swap Q6 229 / Q3 256-CROWN; with
enc-14 elected on lineitem: Q6 350 / Q3 314 -- the dress costs the
flagships more than its 47MB pays AS CONSUMED TODAY. POD DATA
RESTORED to the crowned state (lineitem_0.wdb.e14v2 kept for A/B).
STANDING CONTRACT before lineitem re-elects: bill the enc-14
remainder line by line (per-query plane decompress ~30-50; the
mask fires full-width OUTSIDE the running rule -- mask-vs-pred
order ungoverned; codes_at at survivors pays full reconstruct),
strike the named lines, re-verdict. The organs are general and
stay; the ELECTION waits for the bill to balance.
Boards megaboard_v4_8 + tpchboard10 grinding as the code gate.


## BOARDS + A/B CLOSE 2026-08-28
Boards ZERO-WRONG both (megaboard 103/103, 90 wins; tpch 3/14 ok).
Same-pod same-minute A/B (record code eaea2e3 vs HEAD 005630c):
Q1 0.39-WIN vs 0.45 = CODE REGRESSION ~+60ms from today's emitter/
partition changes, line unidentified; Q3 0.32 vs 0.33 and Q6 0.33
vs 0.35 = POD AMBIENT (proven by A/B, duck drifted equally).
NEXT SESSION OPENS WITH: bill Q1 under both heads until the 60ms
names itself (suspects: pred-partition pre-pass per query; emitter
source changes; mask plumbing). Then the enc-14 remainder bill per
the standing contract. Crowned data state intact on pod.


## 2026-08-28 LATE: THE DOUBLE-SCAN DAY (Jackson's session)
Jackson's shot in the dark, caught in the act by door instruments:
a fully-plane-served WHERE re-evaluated itself via get_mask (full
column reconstruction to re-answer an answered question) AND twin
half-bands on one column each paid 3-plane tests instead of fusing
into their year-aligned interval. Both killed (f637604): Q6 planes
426->321, Q1 planes 789->555. SAME-MINUTE TRIANGLE: Q6 enc-3 331 /
enc-14 273 / duck 231 -- THE DRESS NOW WINS ITS ORIGIN QUERY by
58ms same-metal (~0.87x vs duck; ~185ms-equivalent on the 207-duck
day). Fused year mask (test-at-decompress, Jackson's shape,
4428c76): flat within today's +/-35ms pod noise, kept for shape.
STANDING: (1) serve governance -- census-gate the plane serve
(Q1's 98%-keep mask costs 58 for nothing; -15 residue); (2) Q1's
code-only +100 (A/B-proven, stage-smeared, needs alternating-min
protocol); (3) mask delivery fusion (60M bool round-trip, the
architecture question); (4) ELECTION RE-VERDICT + boards before
lineitem re-elects; (5) pod noise now exceeds strike sizes --
repeat-median protocol required for further Q6 verdicts.


## Q1 CROWN RECOVERED (2026-08-28, 4712f6d)
The +60-100ms code regression = THE WORKQUEUE TAX: the fork-safe
numba layer taxed EVERY prange kernel (kernel, comp, keys), which
is why it smeared across stages. Repealed; fork safety kept at the
source (cube + prewarm pools on spawn; regexgroup COW fork keeps
its graceful fallback). Alternating drift-proof A/B: REC 431/458
vs HEAD 358/425 -- Q1 1.04-1.16x, BEATS DUCK AND THE RECORD.
Same-day scoreboard: Q6 0.87x in the field-plane dress (its best
ever), Q1 crowned again. LAW LEARNED: environment-level knobs are
engine-wide taxes -- price them on a flagship before adoption.


## THE ELECTION BANKED 2026-08-29
Median protocol (5 boards per arm, zero wrong on all 10): PLANES
0.98/1.00/0.77 vs ENC-3 0.95/0.94/0.69 on Q1/Q3/Q6 -- the dress
wins or ties every face, -47MB disk. lineitem WEARS THE FIELD
PLANES PERMANENTLY (enc-3 retired to .enc3 backup). Jackson's
y/m/d design, elected by the rule he ratified: types nominate,
measurements size, the election elects.


## Q4: HOLE -> FACE (2026-08-29, Jackson's walk end to end)
The walk: quarter the orders, ask each surviving order ONE BIT via
its line items, stop at the first late line, count by priority.
Landed as: matcher sees bare correlations (schema truth); THE
SCATTER FORM -- child residual pred in arrays, idempotent bool
scatter through the road sidecar (which now BIRTHS on first ask),
parent rows ARE codes on mode-4 keys, straight into the _codes
sentinel; the In-loop honors pre-answered sentinels. Numbers:
HOLE -> 14,570ms (value-path v1) -> 1,413ms EXACT (duck 234).
Five silent gates convicted en route incl. wdb_subquery lacking
module-level numpy and my unique-sort of 38M rows (9 of the 10s).
NEXT STRIKES, named: plane-lex col-vs-col compare (kills 2x
reconstruct + 1GB of value gathers -> ~450 est); quarter-first
pruning through the road (Jackson's original order) -> ~duck.
STANDING: boards re-gate (subquery machinery touched); Q22 next
on this matcher + the FROM-door; holes now 10.


## ENC-15 THE CLOCK DRESS: FORGED, CORRECT, ELECTION DEFERRED (2026-08-29 late)
Jackson designed the dial live: date pair = anchor(min planes) +
arm separation(u8 delta) + one orientation bit. Probe: 3-way sizing
beat enc-14 by 14pc on commit+receipt; sorted-pair LOST (pigeonhole
honored). Forged whole: pair nomination by measured property, size
election vs BOTH standalone sections, tag-15 anchor + tag-16 stub,
either column reconstructs its own codes (nogil t-eyes), no-FD fast
path two-phased so cross-column elections can see. THE ELECTION
OUT-DESIGNED US: it paired ship+receipt (delta 1-30d, tightest) not
commit+receipt -- the law measuring better than intuition. Result:
file 1,075->1,034.6MB (-40.6), ALL EXACT (Q4 1,427 in-dress), suite
1692/0 -- but crowns taxed (Q6 0.50x, Q3 0.88x) because plane_test/
year-path/exec-confirm gate ==14 and shipdate now wears 15. BY THE
VERDICT LAW: NOT BANKED; pod restored to crowned e14; .e15 file
kept for re-verdict. NEXT-SESSION FORGE, shaped exactly: plane_test
-15 (shipdate = anchor + delta*(1-bit): band test = one fused pass
over 5 planes; year path rides the anchor) -> the clock keeps its
-40MB AND the crowns. Same debt enc-14 paid in one session.
ALSO THIS SESSION: Q4 hole->face (14,570 -> 1,413, Jackson walk),
elections banked for enc-14 by medians, Q3 crown retaken dressed,
serve governance + Q1 +100 hunt closed (workqueue tax repealed).


## PLANE_TEST-15 BUILT; THE LAST KERNEL NAMED (session end)
The clock now answers band tests (fused 5-stream kernel, both pair
columns, all three consumer gates widened; suite 1692/0, pushed).
Measured: the serve costs 152ms/query (full civil math) vs enc-14
year path ~20 -> crowns still taxed (Q6 0.55, Q3 0.82 in-dress).
THE FAST FORM, shaped by the data itself: bit(ship<=receipt) is
~all-ones, so shipdate==anchor a.e. -> per-frame: if the bit chunk
is solid ones, skip delta AND civil math, pure LEX tuple test on
anchor y/m/d (the e14 trick verbatim); mixed frames take the civil
path per mismatched row only. Est: e14 parity, clock keeps -40MB.
NEXT SESSION OPENS HERE: that kernel (e15_band_lex_chunk), then
verdict trio, then the median election protocol for the clock.
Pod .132:11381 holds lineitem_0.wdb(=e14, crowned), .e15, .enc3.
Then: Q22 on the widened matcher; Q12/Q14/Q10 guard bugs; boards.


## THE DECLARED CLOCK BANKS (2026-08-29 night)
Jackson ruled: clock pairs are OPERATOR-DECLARED (semantics are
workload knowledge; the engine never guesses) -- date_pairs on
encode, physical property verified, fails loud. Declared commit+
receipt on lineitem: ship KEEPS planes by construction, crowns
verified IN the new file (Q6 199ms 0.96x BEST EVER, Q3 298ms 1.01x
crown held), Q4 1,515 exact, file 1,054.4MB (-20.9 vs e14). Median
protocol still gates the formal election; trio is unambiguous.
NEXT OPENERS: (1) Q4/Q12 bit consumer -- the scatter evaluator
answers commit<receipt by READING THE BIT (est. Q4 ~350-450);
(2) median protocol to formalize; (3) Q22 on the widened matcher;
(4) guard bugs Q12/Q14/Q10. Pod .132:11381: lineitem_0.wdb =
DECLARED CLOCK (live), .e14, .e15 (auto-pair), .enc3 backups.


## THE BIT ANSWERS (session close): Q4 1,515 -> 952 EXACT (-563).
pair_bits reads delta+orientation only (~28MB laned); the scatter
serves declared-pair compares from the operator table. Remaining
~700ms named: the 38M-parent road scatter fires BEFORE the outer
quarter -- Jackson's quarter-first ordering is the next strike
(est ~400ms), then Q12 consumes the same bit on its guard fix.


## Q4 ENDGAME (session true close): 952 -> 778 -> 354 -> 270ms EXACT
(duck 234, 0.87x). Line-item hunt, Jackson-driven: quarter-first
(-174); THE REBIRTH BUG (-424): fk_pointer used a stale .fkptr
naming that never matched the .jptr birthmark regime, so the road
re-birthed EVERY query (470ms) -- exists-road now rides the
canonical _hash_pointer; conjunct value gathers dedup; then his
three-scans glue FUSED (-70): child verdict AND parent gate AND
scatter in one prange pass. Remaining vs duck ~35: outer exec
assemble ~66 on a 3-row answer + okeep gathers (kx-LUT instead of
td-gather, est -20) -- named, unstruck. NEXT OPENERS: Q12 guard
fix (+ free bit consumer), Q22, median protocol, boards re-gate
(subquery+kernels touched since last gate). fk_pointer/path_for
stale API flagged for retirement.


## THE DEAL LAW DAY (2026-08-30): ENC-17 BANKED, TWO CROWNS AT ONCE
Jackson could not phrase the intuition at breakfast; by evening it
was statute: every compression is a DEAL -- symmetric shrink S vs
symmetric slowdown T, zstd keeps a column iff S > T. Measured
genesis: the deals tables (per-op, engine-true strategies after
his fairness objection killed my bogus lookup numbers) showed a
cliff -- floor-entropy columns paying 12-39ms/MB tolls vs sumslice
earning 58 percent. ENC-17 raw packed codes (LE bitstream, mmap
1:1, one unpack kernel). THREE ELECTION BUGS caught by file-size
confessions across four grinds: inverted S, a bare-sum op model,
then 1T-bincount overshoot -- T is now a CALIBRATED CONSTANT
(0.17 from the engine-true bench; per-column micro-timing lied
both directions). Verdicts: ship/qty/disc/tax -> 17; retflag/
sumslice keep earned zstd; binaries excluded (Jackson). File
1,066MB (+12 buying the tolls out). FLAGSHIPS: Q6 182ms 0.95x
best-ever; Q1 339ms 1.07x CROWN. FRESH GATES: megaboard 103/103
zero-wrong 3.68x; tpch TWO SIMULTANEOUS WINS (Q1 1.03, Q3 1.09
BEST EVER) -- campaign first. Also today: Q12+Q14 holes->faces
(pandas catch-all taught col-vs-col, CASE-in-agg, post-agg
compose, arrow-LIKE sidestep); holes 10->8.
SPEED WAR NEXT: Q12 4.0s / Q14 2.2s on pandas -- fast-path homes
(CASE in the fused kernel; Q12 consumes the clock bit free).
Enc-14 nomination gate over-wide on small ints (wart, docketed).


## THE ADAPTIVE CASCADE DAY (2026-08-31): Q6 CROWNED — CAMPAIGN FIRST
Jackson's session, three rulings forged into the pred door:
(1) COUNT/GROUP-SURVIVORS is its own query class -- every conjunct
must run; potency ORDERS work, it cannot skip it; (2) potency
recalculates per stage (conjuncts correlate) with a TWO-TERM cost
model -- fixed-cost serves cannot be reduced so they go FIRST;
(3) SHADOW SCHEDULING: the clock band fans out only 5.4x of 16
cores, so cheap serves run CONCURRENTLY inside its shadow
(measured: 3 serves, 300ms serial -> 164 concurrent). Plus CASE in
the fused kernel (string conds ride _code_lut as 0/1 slots) and
the pair bit serving in the pred partition (third bit consumer).
Q12: 4,763 -> 972 exact. AND THE CROWNS: Q6 148ms 1.11x -- THE
ORIGIN QUERY BEATS DUCK, first time ever; Q3 250ms 1.13x best
ever. MEASURED EN ROUTE: receipt-year on the clock is STREAM-BOUND
(~182 floor in-dress; the byte-scan year exists only where a
column owns its y-plane); survivor-scale clock read docketed.
DOCKETED: the JOIN-shape survivor handoff (group scramble; suite
q12-clause is its gate); a declared-pair col-vs-col suite test
(the mask-drop passed the suite, caught only by pod exactness);
sibling fusion (bit+band share streams, ~45ms); boards re-gate.


## THE TWO-CASCADE SESSION (2026-08-31 pm): LAWS BANKED, HOIST REVERTED
Jackson's session. BANKED SUITE-GREEN: THE SURVIVOR READ (codes_at
on enc-14/15/16 = streams-once + math-at-rows; every consumer
silently faster) · THE COST CURVE (cost_of(nm,n)=a+b*n from stored
bytes x calibrated family rates; zero-ms plans) · served-conjunct
gate fix · concurrent serves live. LAWS RATIFIED, AWAITING THE
STRUCTURE: the SHADOW ADMISSION LAW (join shadow iff max(0,F-W)<V)
and TWO FORMAL CASCADES (selection vs count/group-survivors).
THE HOIST (attempt 1, tools/hoist_attempt1.py): partition+handoff
moved before gkeys so keys are BORN at survivor scale; q12-clause
PASSED on joins; REVERTED because Q6 (scalar, single-table) went
exact=False -- the scalar flow disturbed. CLUE: suite passed with
the wrongness; scalar+plane-serve exactness is coverage gap #2
(gap #1: declared-pair col-vs-col). NEXT SESSION OPENER: two suite
exactness tests FIRST, then hoist attempt 2 hunting the scalar
disturbance, then the admission scheduler on the curves.
Q12 floor ~210 stands: needs hoisted handoff + sibling fusion.


## THE HUNT (2026-08-30, session 3): Q12 4,763 -> 300, Q4 CROWNED
Jackson's orders: harden the suite, understand before abandoning,
never trade correctness -- and fix it. THE EXACTNESS GATES (four
new walls, tests/test_exactness_gates.py) caught, in order: the
hoist's Q6 breaker (the pred block's _plane_mask9 re-init ERASING
hoisted serves -- the WHY of attempt 1, one line), dt3 numeric
literals silently emptying bands (now native day units), ORDER
KEYS dressed as dates (nomination gate: u[0]>=366, V<=20000), the
LUT band's 1-based month table. THE HOIST LANDED (keys born at
survivor scale). CONVICTIONS by absolute-clock brackets: survivor
slots full-decoding first (raw_dict_col want_codes); clock streams
per-consumer (SIBLING SHARE: one locked loader feeds band + bit +
survivor reads); an orphan birthmark hiding a per-query 60M rehash
(fail-loud now); roads re-paging off the network mount (RAM-pinned
int32); and the deepest -- wdb_qmem flushing _ptr_cache every query
made the legacy .fkptr road a 480ms decompress+cumsum tax on EVERY
join: the chain now rides the canonical .jptr road first (550ms
back across Q12 and Q4). TABLE-DRIVEN CLOCK BAND (ystart+mcum,
planes M/D 0-based). Loaders fan to the box. Suite runs ON THE POD
(local /tmp leak fixed: suite temps cleaned per process).
NUMBERS: Q12 300ms exact (0.78-1.04x, crosses duck once) · Q4 224
1.13x CROWNED · Q3 1.20x best ever · Q6 1.10x · Q1 holds. Suite
1696/0. THE LAST 90 TO 210, measured: dress floor 84 (streams 42 +
LUT 42); partition wall ~175 = warm-loads contending with the band
inside the shadow; gkeys 44 + assemble 110 serial after survivors
(~15ms real per-row work, rest glue). NEXT OPENERS: warm-pool
contention (one unified scheduler, the admission law on the cost
curves), the serial post-survivor glue, then the boards re-gate on
everything banked today. Q14 fast path still docketed.


## THE TREE & THE DEAD GATE (2026-08-31): board ok=8, WRONG=0
Q5 HOLE -> FACE ON THE FAST PATH: THE CHAIN IS A TREE (edges a
multimap child->parents; BFS routes; pruning keeps fact->referenced
paths); parent-vs-parent equalities ride WHERE as pred conjuncts
(two pointer-slot VALUE streams, dict-independent; the old blanket
decline predated that). Plan attempts ISOLATED (each _build_chain
gets its own tree copy; strip only on success), uncomposed alias
declines the plan, declined stored-only chain releases for the
full-tree retry. 525ms exact; probe floor 247 (order band at
parent scale DOCKETED). Q10 HOLE -> FACE, full-row exact: pandas
tail skips output aliases and gathers non-WHERE columns AT
SURVIVORS; values_at takes DICT CODES never rows (my violation --
mode-5 sorted dict punished it; engine/encoder/data all vindicated,
no re-encode). ~23s face: WHERE phase decodes fact-scale strings;
speed home = fast-path mode-5 gkeys / descent top-k (DOCKETED).
THE SILENT GATE IS DEAD: a failed pred build declines the plan
instead of proceeding predicate-less (Q21 had shipped a confident
wrong top-100). Its death exposed accidents the swallow served, now
explicit: IS [NOT] NULL folds on no-null columns; MODE-4 sequence
slots (raw_dict_col serves values as identity-coded base); THE MASK
HANDSHAKE -- a pred declining ONLY on subquery-resolved INs (codes
attached) ANDs the full WHERE mask into the plan (Q4 had been exact
by accident via the numpy specs path; now a contract). Integer
aggregates emit BIGINT (Q12's WRONG was 62071.0 vs 62071; the
one-column qx check is retired -- the board referee is law).
BOARD: 14 attempted | ok=8 (Q1,3,4,5,6,10,12,14) | holes=6 (Q7/13/22
FROM-door, Q18 IN-subquery, Q19 compound ON, Q21 declines honestly)
| wins=4 | WRONG=0. Suite 1696/0 on the pod at every push.
NEXT OPENERS: Q10 speed (23s: mode-5 gkeys or descent top-20-by-
SUM); Q5 parent-scale serve (525->~300); the FROM-subquery door
(three holes, one organ); Q12's last 60 to 210; Q14 fast path.


## THE COMPOSE (2026-09-01): board wins=6, wave total 2.3s == duck
Q10 23s -> 198ms (1.71x) and Q5 519 -> 183ms (1.78x) from ONE
conviction chain. Probe first (Jackson's law): group by the customer
pointer, top-k, decode attributes for k rows -- 164ms exact; the
census-descent (Threshold Algorithm) belongs to the UNFILTERED top-k
lane (docketed): under filters the filtered counts cost the same
bincount as the sums. fpa's native topk-attr door already served the
shape once the cascade stopped declining; the attribute court I wrote
was dead code and came out. CASCADE cuts riding every join: native
keeps via parallel LUT kernel (128->47); running rule enforced -- a
depth-1 parent keep applies AT the fact keep's survivors (5ms), never
flowed to fact scale (60M gather+AND, 120ms). raw_dict_col caches the
typed base per segment (1M-entry Python-list rebuild per slot, 35ms).
THE COMPOSE: the full-tree chain composed grandparent roads at FACT
SCALE with serial numpy (two 60M gathers, ~237ms per query) -> parallel
compose kernel (pre-work 247->53). Early decline for unreachable
referenced aliases at chain-build time. COLD-START named: codes_at on a
mode-5 column factorizes the whole column (2s/column) -- attribute
readers at rows must use values_at_rows; the rest is road pinning off
the network mount (docketed). BOARD: 14 attempted | ok=8 | holes=6 |
wins=6 (Q1 1.02, Q3 1.06, Q5 1.87, Q6 1.45, Q10 1.70, Q12 1.10; Q4 0.98
breathing; Q14 0.41) | WRONG=0 | wave 2.3s duck 2.3s. Suite 1696/0.
NEXT OPENERS: Q14 fast path (Div-of-Sums, the only sub-1x face); the
FROM-subquery door (Q7/Q13/Q22; Q7 is tree-shaped inside it); Q4's
last breath; cold-start road pinning; the unfiltered-top-k descent
with Jackson's census bound.


## THE DISSOLUTIONS (2026-09-02): THE BOARD IS FULL -- 14/14, 0 holes, 0 wrong
SIX HOLES FELL IN ONE SESSION, every one to a dissolution, not an
organ. Q7: THE FROM DOOR + THE FLATTEN (aggregate-less inner folds
into the outer) + EXPRESSION GROUP KEYS RIDE THE DICT (transform V
values, collapse, remap -- O(V) never O(N)) + alias-identity key
matching (n1/n2). Q13 (CROWN 1.46x): Jackson saw the LEFT JOIN
fetches NO foreign data -- domain=all customers, absent=zero: byte-
kernel NOT LIKE over the front-coded dict (plike_fc: restart blocks,
prefix carry; 118ms vs 15s of Python) + two bincounts. Q22: prefixes
read as TWO BYTES (pprefix2), scalar AVG recurses over the same
masks, NOT EXISTS = THE CENSUS SERVE (empty slot in the fk bincount).
Q18: THE WEIGHTED CENSUS SERVE -- IN(GROUP BY key HAVING agg cmp lit)
keeps parents whose weighted-bincount slot clears the bar; resolved
to literal keys; census cached per parent segment; the topk-attr door
learned ORDER BY ATTRIBUTES. Q19: SCHOOLBOOK ALGEBRA -- (E and A) or
(E and B) = E and (A or B); common conjuncts hoist from OR branches;
ZERO new organs; 1.12x on arrival. Q21 (the last): THE LONELY REWRITE
-- the EXISTS pair is two parent keeps: per-order distinct-supplier
censuses over the sorted road (pruns_distinct); keep = nsupp>=2 and
late-distinct==1 injected as an In carrying _codes; the mask
handshake accepts _codes INs; THE DECLARED CLOCK serves strict pair
compares in the mask layer. sqlglot notes: from_ key, Like(negate).
BOARD: ok=14 | holes=0 | wins=9 (Q5 1.87, Q10 1.68, Q6 1.67, Q13
1.46, Q1 1.28, Q12 1.24, Q19 1.12, Q4 1.06, Q3 1.03) | WRONG=0.
Suite 1696/0 at every push. SPEED WARS, probe floors proven: Q18
296ms/1.67x (cascade declines parent literal-IN -- convict next);
Q21 786ms/1.11x (same escape to pandas); Q7 2.1s (OR-pred at fact
scale); Q22 census sidecar; Q14 Div-of-Sums. The campaign's law
held: probe first, dissolve before building, measure before
theorizing, never ship wrong.


## THE CROSSINGS (2026-09-03): wins=12/14, wave 3.8s vs duck 4.9s
FOUR QUERIES CROSSED DUCK IN ONE DAY. Q7 2,080->312 (1.05x): the
survivor handoff PRICES the key (dressed keys hand off at keep<60%);
MULTI_GROUP_CEIL 2^18->2^21; THE IMPLIED KEEP (an OR pinned on the
same columns in every branch implies col IN (union)); the algebra pass
on every OR conjunct (a stale duplicate def was shadowing it);
cascade LEFTOVERS under THE ARBITER (potent parent keep <5% AND heavy
keys, or parent keep <1%); tiny mode-5 dims keep by rank; flow to
FIXPOINT (a snapshot loop dropped keeps past one hop); conjunct
fusion PER ALIAS. Q18 2,200->273 (1.97x): mode-4 sequence keys in the
cascade (cached array, never list() of 15M); the door picks THE
ROUTABLE KEY; gate accepts ORDER BY attributes. Q21 2,200->749 (1.18x):
cascade serves _codes INs; the door honors LEFTOVERS AT ROWS (it had
consumed rows9 raw -- the referee caught it in one run). Q14 580->160
(1.44x): AGGREGATE ARITHMETIC -- hidden aggregate aliases through the
scalar engine, arithmetic on the results. Q5 rode the cascade laws to
1.81x untouched. BOARD: ok=14 | holes=0 | wins=12 (Q3 0.99 at the
line; Q22 0.77 the last under) | WRONG=0 | wave 3.8s duck 4.9s. Suite
1696/0 at every push. Housekeeping: the remote URL had a baked-in
token; now clean, helper=store. NEXT OPENERS: Q22 census sidecar
(.cnt.npy law on fk columns); Q3 the last breath; the ARBITER as a
true cost curve; Q21 786ms probe floor (the lonely censuses cached);
cold-start road pinning.


## THE LINE (2026-09-03 evening): wins=13/14, wave 3.5s vs duck 4.9s
Q22 (1.79x) and Q3 (1.24x) were already won on a fresh pod -- the
earlier scores were CPU breathing, not engine debt. Q19 275->132
(2.16x): THE CASCADE SERVES SCALAR QUERIES (its gate demanded a group
key); the arbiter prices TOTAL expected keep for scalars; the ZERO-
SURVIVOR LAW held (a scalar emits exactly one row over an empty
cascade -- five matrix tests caught the []). BOARD: ok=14 | holes=0 |
wins=13 (Q7 0.98-1.04 pod-to-pod: a coin flip, needs ~15% margin) |
WRONG=0 | wave 3.5s duck 4.9s. Suite 1696/0. NEXT STAGE: generality --
run every board (ClickBench, mega, old TPC, this one), then hunt a
dataset that exposes the engine's assumptions (FK roads, sorted
child runs, dict-space serves).


## THE GENERALITY STAGE (2026-09-04): all boards clean, a new realm
ALL BOARDS: TPC-H 14/14 exact wins 13 (wave 3.5s vs duck 4.9s);
Megaboard 103/103 (wave faster on 88, median 3.72x); ClickBench 42/43
ok, false=0, Q27 timeout known. Two generality defects found by the
existing boards before any new data: (1) THE ID-SPACE BIRTHMARK -- the
group-distinct shelf validated only n==N; a July RegionID shelf built
on raw values, decoded today as dict codes, gave right counts with
WRONG labels at 4ms (ClickBench Q08/Q09). Sidecars carry their id
space; a mismatch is refused loudly and reborn. (2) THE SELF-EXISTS
CODE SET (megaboard cq-notexists): same-table EXISTS with an inner
filter is a code set in the outer column's own space; and the
fallback read door re-parsed SQL text, losing resolved _codes (now
takes the tree). NEW REALM: H2O.ai db-benchmark groupby (50M rows,
K=100, 500K-distinct string and int keys; bench/h2o_groupby.py with a
numpy generator and the referee-law comparator). Findings: the single-
table executor treated any unrecognized projection as a GROUP KEY and
crashed on a positional fallback -> aggregate arithmetic routes to the
engine that rewrites it; MEDIAN/STDDEV/CORR and window functions
decline loudly; MIN/MAX emit integers like SUM. BOARD: ok=6/7
attempted, wins=5 (q1 6.2x, q2 6.4x, q3 2.3x, q4 2.3x, q5 3.0x; q7
0.83x), WRONG=0, three loud holes (median/stddev, window, corr). THE
CARDINALITY WAR: q10 GROUP BY id1..id6 = 50M groups (V==N; duck 49s
incl. fetchall) -- the composite overflow now declines loudly in 1.7s
instead of a 37GB pandas hang; THE HASHED-COMPOSITE ORGAN for the
moderate range (2^21..~20M groups) is the next real piece. Speed wars
docketed: ClickBench Q09 (1.1s, COUNT DISTINCT in a multi-aggregate),
cq-notexists (5.3s code walk), Q7 margin, H2O q7 (0.83x). Suite
1696/0 at every push. NEXT: hashed-composite organ; MEDIAN/STDDEV/CORR
aggregates; the join realm (JOB / H2O joins).


## THE CARDINALITY REALM, COMPLETE (2026-09-04, night): H2O groupby 10/10 exact
THE HASHED COMPOSITE (composite key spaces past int64 pack by bit width
into two words; open-addressing group ids at survivor scale; key words
kept for emission) -- q10 50M groups exact, organ 2.4s. EMISSION AT C
SPEED (dictionaries decode once per V, labels as object arrays, numeric
cells skip _pyval) -- 57s -> 27.8s on 50M x 8 cells; lifted every high-
cardinality query 50-70%. MOMENTS AS ALGEBRA (STDDEV/VARIANCE/CORR over
hidden SUM/COUNT; q9 3.9x). THE ORDER-STATISTIC SIDE PASS (MEDIAN:
counting scatter + per-group parallel sorts; q6 20.5s -> 1.85s). THE
TOP-K-PER-GROUP DOOR (ROW_NUMBER OVER PARTITION/ORDER + rn<=k: scatter
row indices, per-group sorts, gather; q8 exact, 1M rows). BOARD: q1
7.4x, q2 7.8x, q3 4.3x, q4 2.5x, q5 5.0x, q9 3.9x, q10 ~1.6x (50M
output rows; duck 49s incl. fetchall); faces under: q6 0.53, q7 0.96,
q8 0.27 -- wars named (side-pass cost; sequential scatter + per-group
allocations). WRONG=0 throughout; suite 1696/0 at every push. LAW
HELD: probe first, dissolve before building, unknown shapes decline
LOUDLY (never a crash, never a hang). NEXT: the join realm -- H2O
joins (small/medium/big, the big-big 1:1 with no road), then JOB.


## THE JOIN REALM (2026-09-05/06): H2O joins 8/8 exact
New realm + board (bench/h2o_join.py: x 10M, small 10, medium 10K, big
10M; five official SELECT-* joins + aggregate forms; FULL-ROW comparator
-- x.id3 is not unique, and three 'wrongs' in this realm were the
comparator, never the engine). THE ROAD JOIN: a row-emitting equi-join
to a unique key is a POINTER, not a merge (_join_pointer keeps -1 for
misses; INNER drops, LEFT emits NULLs; columns gather at rows / at the
pointer; mode-aware reads) -- served j1..j5 INCLUDING the big-big 1:1
(j5), which is just another road. The step-1 joiner accepts INNER/LEFT/
RIGHT; the chain builder accepts explicit INNER ... ON (one gate hid a
4-12x lead: a1 3.7x, a2 3.9x, a5 big-big aggregate 12.6x). THE DECODED-
DICT SHELF consulted before the typed dict (the flush drops _tdict; a
6.3M-string dict re-decoded per query: 4.0s -> 0.88s). Route trace:
WDB_ROUTE_DEBUG. BOARD (clean morning): 5/5 SELECT-* exact at 0.58-
0.72x (THE EMISSION WAR: 10M x 9-13 Python cells on both sides; floor
~6-8s vs duck ~11.5); 3/3 aggregate forms winning. LAW HELD: A/B on the
same pod before attribution -- a '63s regression' was host contention
(load 25 on a shared box; duck stalled too); the committed source
measured identically. NEXT: re-measure the join board on a quiet pod;
the emission floor for SELECT-*; then JOB (junction-table joins --
the first realm where the join truly fetches foreign data both ways).


## THE JOIN REALM, SWEPT (2026-09-07): H2O joins 8/8 exact, 8/8 wins
Wave 45.4s vs duck 69.2s. j1 1.42x, j2 1.53x, j3 LEFT 1.54x, j4 string
key 1.45x, j5 big-big 1:1 1.54x; aggregate forms a1 4.5x, a2 4.2x, a5
big-big 13.5x. Convictions: (1) the 63s 'regression' was HALF mine --
host contention stalled duck too, but beneath it the chain's new INNER
acceptance let SELECT-* joins pass the lazy chain, get declined by fpa,
and fall to the pandas tail with the road-join door below: THE ROAD
JOIN NOW RUNS BEFORE THE LAZY CHAIN (63.5s -> 11.5s). (2) The decoded-
dict shelf delivered its isolated bill once routing stopped hiding it.
(3) A ten-row dim's mode-5 column sliced per row 10M times (3.9s):
SMALL MODE-5 SIDES DECODE ONCE AND GATHER (138ms); j1 11.8 -> 7.9s.
(4) Parallel column reads measured, no gain (GIL-bound) -- removed;
per-column bill under WDB_JOIN_BILL. The honest floor: ~3.4s of zip
building 10M Python tuples, the cost duck pays in its own fetchall.
BOARD OF BOARDS: TPC-H 14/14 (13 wins) | Megaboard 103/103 |
ClickBench 42/43, false=0 | H2O groupby 10/10 (7 wins) | H2O joins 8/8
(8 wins) | suite 1696/0 at every push. FIVE REALMS, ZERO LIES. NEXT
SESSION: plan from here -- candidates: JOB (junction-table joins; the
real hash-join question), the groupby wars (q6 side pass, q8 window
scatter), old speed wars (ClickBench Q09, cq-notexists), Q7 margin.


## THE SCOPE STAGE (2026-09-07/08): 100/100 SQL constructs exact
bench/sql_scope.py: 100 constructs vs duck on a 200K realm with nulls,
booleans, dates -- classified OK / WRONG / HOLE (loud) / CRASH. Sunday
morning: OK=51 HOLE=31 CRASH=13 WRONG=4. Monday night: OK=100. THE
WRONGS (four, the class that matters): QUALIFY and DISTINCT ON were
silently IGNORED (now rewrites into the top-k door); booleans emitted as
strings (THE BOOL MARKER, aux=9, decoded at the typed dict and the point
read); integer sums as floats (INTEGER EMISSION in one post-pass, TYPE-
based: FLOOR sums to a double, an integer expression to a BIGINT). THE
CRASHES (13): the single-table planner declines BY NAME anything it
cannot classify. THE FAMILIES: expressions (the row evaluator learned
the function vocabulary; aggregate arguments and predicates fall back
to it), windows (THE WINDOW DOOR: one lexsort by (partition, order),
every function as arithmetic over the sorted order; QUALIFY evaluated
over its own output), joins (the step-1 joiner general: USING, multi-
column ON, computed keys, one-sided extras filter their side BEFORE
the merge, CROSS/non-equi bounded, FULL OUTER with extras = INNER under
every conjunct + both sides unmatched), aggregates (THE ROW-AGGREGATE
family over per-group slices: QUANTILE/MODE/BOOL_AND/STRING_AGG/
ANY_VALUE/FILTER/DISTINCT-and-expression args), subqueries (the
correlated-scalar decorrelation accepts arithmetic around the
aggregate; ANY/ALL, SELECT-list scalars and nested-aggregation CTEs are
re-entries BEFORE routing), literal relations (wdb_literal.py on
sqlglot's executor, gated by the LIVE catalog -- the suite caught the
first gate). LAWS HELD: rewrites re-enter before routing or the router
chooses from the old text; a decline is a raise, never a None; LAG and
STRING_AGG on tied keys are legally ambiguous -- the probe orders
deterministically. Honest note: the new families are correctness
faces (row evaluator, pandas joiner); the dict-space transform is the
speed home when a board asks. Suite 1696/0 at every push.


## THE SCOPE SPEED STAGE (2026-09-08): the faces billed and struck
bench/board_scope.py: all 100 constructs at 10M rows, wave vs duck,
exact-checked -- a permanent third eye beside the realms. Morning:
wave 140.7s vs duck 15.8s, 52 wins, and TWO WRONGS at scale the 200K
probe never saw. Evening: wave ~29s, 55+ wins, median 1.5x, 100/100.
CONVICTIONS: PREDICATE PUSHDOWN in the step-1 joiner (a 10M x 10M self-
join built ten billion pairs before its WHERE); string literals stay
strings in _scalar_cmp ('01' is not 1); NULLIF/COALESCE typed for
integer emission; THE BARE-COLUMN LAW twice -- the cascade's mask leaf
and its plan-build accepted a function LHS and resolved it through
.name to the inner column (SUBSTRING(id3,3,2) = '10' judged as id3 =
'10': a SILENT ZERO, only on front-coded columns at scale). ORGANS:
THE DICTIONARY MAP (a function over a dictionary column evaluates once
per distinct value and gathers through the codes; DISTINCT aggregates
over expressions on distinct code tuples; MIN/MAX of strings from the
extreme present code) -- concat 21.4s -> 0.30, regexp 10.6 -> 0.29,
like 5.6 -> 0.37, substring 4.0 -> 0.37, string_agg 11.9 -> 1.8; THE
TOP-K ROWS DOOR (argpartition with ties, codes as keys for sorted
dictionaries, yields to clustertopk) 26.9s -> 0.66; SET SEMANTICS AT THE
LEAVES (INTERSECT/EXCEPT/UNION leaves DISTINCT at V-scale) 7.4 -> 0.03,
6.5 -> 0.04; the window door filters on arrays before materialising
and counting-scatters partition-only windows (scalar_corr 11.6 -> 1.6);
one-column expression group keys in code space (case_multi 4.0 ->
0.06). PROBE LAWS: ANY_VALUE is legally ambiguous per group; float sums
differ in the 14th digit -- the board's %.6g comparator is the referee.
LEFT: small row-aggregates ~0.8-1.0s (count_if, bool_and, coalesce,
nullif: one column read plus Python -- a composite map or a kernel),
using 0.8s (a small dim through the step-1 joiner; the road organ),
min_max_str 1.6s. Suite 1696/0 at every push.


## THE JOB REALM (2026-09-09): the semi-join fixpoint and the reverse road
IMDB (21 tables; cast_info 36M, movie_info 15M, name 4M) pulled and built
in fifteen minutes: duckdb referee, encodes, catalog (bench/job_realm.py),
the 113 Join Order Benchmark queries as a board (bench/board_job.py; the
'at' alias is a reserved word in duck's parser -- renamed both sides).
First board: 109/109 named holes, ONE family -- two facts meeting
through a shared parent, which the one-fact road tree declines by name.
THE SEMI-JOIN FIXPOINT (src/wdb_semijoin.py): every JOB query projects
only MIN/MAX, and under MIN/MAX row multiplication is irrelevant -- a
table contributes the extreme over its rows that PARTICIPATE. Local
filters seed each table's keep, every equality edge prunes both sides
to the other's surviving keys, iterate to stability, MIN by the extreme
present dictionary code. No hash join, no row explosion. 113/113 EXACT
on the first full board (wave 592s vs duck 10.6s). THE REVERSE ROAD:
an inverted index per join-key column (sorted uniques, offsets, row
order, and the inverse permutation RANK) in four mmap'd .npy sidecars,
born once -- serves prunes from a small keep AND point reads without
decompression (u[searchsorted(offs, rank[row])]). LAWS: three-valued
logic under NOT/LIKE (NULL notes leaked through NOT LIKE in 1b); npz
members are not mmaps; counts tracked, never re-summed over 36M bools;
a fixed-width S-array is V x max length (THE WIDTH LAW); prefix LIKE on
a sorted dictionary is a contiguous code range; string IN always in
code space; IS [NOT] NULL on a null-free column is a constant; a
dictionary is V-scale vocabulary and survives the flush (THE
DICTIONARY SHELF, budgeted, size estimated ONCE). THE PATTERN OF THE
DAY: every JOB cost that fell was a per-row pass over N hiding behind
a shape with a V-scale or survivor-scale answer. Duck wins JOB by brute
vectorised scanning; we win, when we do, by never scanning. 6a 4.3 ->
0.49s, 15b 385 -> 1.8s, 29a 14.4 -> 2.7s. NEXT: the fixpoint should
start from the most selective table and never materialise an all-true
keep; the board twice in one process for the honest warm total.


## THE BOARD CENSUS (2026-09-09, evening): every realm re-run, with peak RSS
TPC-H 14/14 (12 wins, 3.2s vs 4.4s, 8.9GB) | Megaboard 98/103 -> the 5
FALSE were MINE | ClickBench 42/43 false=0 but median 1.17x (was 3.10x)
-> Q25/Q35 were MINE | H2O groupby 10/10 (7 wins, 55.5GB peak) | H2O
joins 8/8 (8 wins, 21.8GB) | Scope 100/100 (71 wins, median 2.26x,
16.8s vs 14.6s -- ahead in total) | JOB 113/113 (143s vs 12s, 11GB).
THE DISEASE, twice: new doors pre-empting older, faster, exact ones.
(1) has_agg_arith saw the SUM inside SUM(x) OVER and sent windows to
the join engine, whose window door ignored frame specs and RANGE peer
semantics -- five megaboard windows FALSE. Fixed and banked: windows
own their aggregates; THE FRAME LAW; the join engine's window door
last. (2) The top-k rows door pre-empted the sorted-projection door
(Q25 40ms -> 615ms) and has_expr_group pre-empted the affine-group door
(Q35 95ms -> 13.3s). Fixed with THE PRECEDENCE LAW (controller first,
new doors answer only declines) -- exact, but it OVER-YIELDS: the old
path serves top_no_group and case_multi slowly instead of declining
(scope board 16.8s -> 56.6s, still 100/100). NEXT: narrow the rule --
yield only to the specialised fast doors (sorted projection, cluster
top-k, affine group), never to the general scan. ALSO DOCKETED: the old
window door accumulates integer running sums in float64 (w-runsum: 105
boundary rows differ from duck's exact BIGINT); RSS peaks of 65.7GB
(megaboard) and 55.5GB (H2O groupby) are emission/window
materialisations -- the shelf ceiling must govern them. THE PLANS
AGREED WITH JACKSON: one sidecar registry (manifest, birthmark
validated on every load, births under the BIRTH GATE, disk budget with
named declines, vacuum) and one process-wide shelf (byte ceiling as a
fraction of RAM, LRU, named declines instead of OOM, one representation
of mode-5 text as an mmap'd sidecar, db.stats()). JOIN, restated in
Jackson's terms: a join declares a shared value space; the work is
aligning dictionaries; rows follow values -- the key-space fixpoint and,
further out, columns as sparse 0/1 matrices over one V-space.


## THE DISCIPLINE STAGE (2026-09-10): the precedence law, the registry, the shelf
THE PRECEDENCE LAW, narrowed: a new-door shape (aggregate arithmetic,
expression group key, window, top-k rows) goes to the scope-stage doors
FIRST unless a SPECIALISED fast door claims it -- asked through the
controller's own detect() (sorted projection, cluster/value top-k,
first-k, distinct-limit, affine group/sum, the old window door,
heavypair, sumtopk, gridwalk, smallk) -- and never yields to the general
scan; and LAST for whatever the controller declines by name. 'Old
first' is as wrong as 'new first': precedence is decided by what a
door IS. ClickBench Q25 51ms, Q35 95ms; scope 100/100 at 17.3s vs duck
17.5s. INTEGER ACCUMULATION in the old window door (w-runsum exact, 58.3M
rows, int64 end to end; two literal 0.0s had promoted the cumsum).
COUNT(DISTINCT col) back on its group-distinct door (Q09 53s -> 1.15s).
THE REGISTRY (wdb_sidecar.py): twelve families known by suffix; scan /
stats / manifest / vacuum; a sidecar older than its segment is false by
construction; may_birth() refuses births past WDB_SIDECAR_GB by name.
Census: 350 sidecars, 7.7GB, all fresh. THE SHELF (wdb_shelf.py): one
process-wide LRU with a byte ceiling (25% of RAM or of the cgroup
limit), refusals by name; dictionaries, inline text (ONE representation:
object array + joined buffer), reverse-road uniques, the string caches;
db.stats() prints shelf and registry. WHY: this pod has a 128GB cgroup
ceiling; two ClickBench boards died oom_kill because the unbounded
inline-text shelves accumulated hits.URL/SearchPhrase at 100M rows in
three copies (Q16 alone: 5.3GB). Under the shelf: ClickBench 42/43,
false=0, faster 24, median 3.35x (was 3.10x). LESSONS: never rsync src
while a board runs (lazy imports); pgrep -f with '\|' is a literal in
ERE -- it reported 0 while nine boards ran concurrently. OPEN: transient
working sets (emission of 50M-row results, giant group-bys: mega peak
65.7GB, H2O groupby 55.5GB) are not shelf objects -- the ceiling must
learn to govern them too (chunked emission, spill, or named declines).


## THE COUNTING FIXPOINT (2026-09-11): joins that count
The biggest generality gap named on the state-of-the-engine map: the
MIN/MAX fixpoint was only legal without multiplicity. Now, after the
semi-join reduction, the join hypergraph's VALUE CLASSES form a join
tree; rooted at the aggregate's table, every surviving row's WEIGHT is
the product over its classes (except the one to its parent) of the
partner weight sums in the child subtrees -- COUNT(*) = sum of root
weights, SUM(T.x) = weighted sum, AVG = ratio, GROUP BY on root keys =
weights per key. No pair is ever built (Yannakakis). Routing: MIN/MAX
fixpoint first; the counting fixpoint LAST, only when the road engine
declines (TPC-H stays on the roads). JOB-COUNT: 113/113 EXACT (wave
149s vs duck 11s -- the fixpoint's speed). Also today: the set-op leaf
cost convicted (fjdb encodes AdvEngineID as a mode-4 sequence, V == N;
the .bst.npz is blockstats, not a bitset; np.isin -> direct compare);
THE STATE OF THE ENGINE map: structural gaps in order -- (1) joins that
count [DONE], (2) multi-segment tables in the join engine (19 solo-
segment gates), (3) transient working-set RAM (emission, giant group-
bys), (4) date/time functions, (5) outer joins and frames in the join
engine, (6) the encoder's mode judgement; product gaps -- DML meets
sidecars/shelves, concurrency, a wire, EXPLAIN/cost model, bad-input
robustness; research -- the key-space fixpoint, the join family as
sparse matrices over one V-space.


## THE STRUCTURAL DAY (2026-09-11): six gaps on the state-of-the-engine map
(1) JOINS THAT COUNT -- THE COUNTING FIXPOINT: weights on the join tree
after the semi-join reduction; COUNT/SUM/AVG/GROUP BY through junction
tables without a pair; JOB-COUNT 113/113 exact. (3) TRANSIENT RAM -- THE
WORKING-SET GOVERNOR (WDB_WORK_MB, 25% of RAM/cgroup): an N-scale Python
result over budget declines BY NAME; db.stream(sql) yields blocks built
from columnar arrays and freed behind (50M rows: 55GB -> 14.7GB peak);
STREAM-AND-CLEAN (Jackson's rule): the row evaluator walks big columns
in WDB_BLOCK_ROWS blocks and keeps only each block's decision. (2)
MULTI-SEGMENT JOINS -- SEGMENT PARTIALS: a catalog override pins the
multi-segment fact to one segment per pass, every door sees a solo
segment, partials merge by wdb_merge's law (one multi-segment table per
join; the merged-dictionary view is the next step). (4) THE DATE FAMILY
-- a 2M-row dates realm; datetime64 emits as date/datetime; EXTRACT /
DATE_TRUNC / INTERVAL (calendar clamp) / DATEDIFF / LAST_DAY / STRFTIME /
EPOCH / casts; 35/35. THE CAST LAW: a TYPE-CHANGING CAST is not
transparent -- three compilers stripped CAST(ts AS DATE) to ts and
compared microseconds to days, silently. (5) OUTER JOINS AND FRAMES --
RIGHT and FULL on the road (unmatched parents once, with NULLs), WHERE
on the road (single-sided conjuncts filter their side; IS NULL is the
only truth on a NULL side); the window door serves ROWS k PRECEDING,
the default RANGE frame (peers share the LAST peer), RANGE UNBOUNDED;
five constructs exact, 2-3x ahead. (6) THE MODE AUDIT: the registry
names storage misfits; fjdb's hits (a symlink to ClickBench's July
segment, pre-guard) carries six narrow mode-4 columns -- every predicate
on them decodes 100M rows; the encoder's judgement itself was already
right; re-encode scheduled. Suite 1696/0 at every push (seven today).


## THE FIXTURE REPAIR AND THE LITERAL-VALUE LAW (2026-09-12)
fjdb's hits_0.wdb was a symlink to ClickBench's July segment, pre-guard
(six flag columns as 100M-value sequences). A full re-encode is not
viable on this pod: the streaming encoder held 37GB and managed 4 of 105
columns in 83 minutes -- THE ENCODER NEEDS THE GOVERNOR (encode a column,
write it, free it; per-column parallelism under a byte budget). The
right road: repoint to hits_14 (same rows, post-guard dictionaries),
remove every hits_0 sidecar BY NAME (dictionary codes differ between
encodes; mtime cannot see it), drop the plan-cache ledger. Within the
hour the new fixture exposed a July silent wrong: _dict_eq_mask's
presence-bitmap shortcut (enc-8 columns) passed the literal NODE to
_code_of -- str(Literal) is the quoted SQL, so = '' matched nothing
and <> '' everything (sq-notin 99.6M vs 12.8M). THE LITERAL-VALUE
LAW: the value, never the node. Megaboard on the healthy fixture: 102/
103, faster on 91, median 4.69x -- A NEW RECORD (was 89 / 3.73x); t-null
0.62s -> 0.00s, u-unionall 0.85s -> 0.14s. OPEN: g-substr WAVE-ERROR
(ascii decode of Cyrillic in a SUBSTR path). EXPLAIN: db.explain(sql) --
which door served, every bill, rows, wall, peak RSS, shelf and governor.
DML MEETS THE DOORS (local, uncommitted, untested): the truth test on a
parquet-encoded realm found DELETE silently deleting zero rows (the
segment class expects a canonical buffer) and INSERT crashing on the
type name str -- and NO controller door consults presence or overrides:
eight doors confidently wrong after one DELETE. Fixed locally: DELETE
tombstones parquet realms; type names; THE PRESENCE GATE (a segment
with tombstones or overrides is served by the general scan only until
compaction). NEXT: sync, rerun /tmp/dml.py, suite, commit; db.compact();
convict g-substr. TWO LESSONS: a second fixture finds what the first
cannot -- realm diversity is a correctness instrument; a fast door that
cannot see a tombstone is a wrong answer waiting for the first DELETE.


## DML MEETS THE DOORS (2026-09-13): the truth test
A DML path never run against a referee on a real realm is a hypothesis.
The truth test (a parquet-encoded copy of the scope realm; eight doors;
DELETE, INSERT, flush, compact; vs duck) found in its first minutes:
DELETE silently deleting ZERO rows (the segment class wanted a canonical
buffer a parquet realm never has); INSERT replacing a 200,000-row
segment with a 1-row one (re-encoding from a buffer holding only the
new row -- DATA LOSS, caught by the presence sidecar's count); and NO
controller door consulting presence or overrides -- eight doors
confidently wrong after one DELETE. LAWS: DELETE tombstones cold
segments; a table with a cold segment and no canonical buffer appends
to the hot tier (a DML-born table keeps its buffer); THE PRESENCE GATE
-- a segment carrying tombstones or overrides is served by the general
scan only; COMPACTION GIVES THE DOORS BACK -- a single dirty segment is
rewritten clean and EVERY derived artefact of a removed segment goes
with it (codes and positions are reborn). Lifecycle: DELETE 41,722 ->
INSERT -> flush (x_1) -> compact (x_0 + x_1 -> x_2, 158,279 live):
before 0 / after 0 / insert 0 / compact 0 wrong, doors serving again.
Also: THE UTF-8 LAW (never astype(str) on bytes: ASCII; g-substr on
Cyrillic). Megaboard on the healthy fixture stands at 91 wins / 4.69x.
NEXT: the encoder under the governor (encode a column, write it, free
it; per-column parallelism under a byte budget); concurrency; a wire;
the string family's speed (SUBSTR over a 6M-entry dictionary: 31s).


## THE ENCODER UNDER THE GOVERNOR (2026-09-12): ingest for real-sized tables
The 100M-row ClickBench table went from 30 hours projected (OOM-killed
twice at 128 GB) to 635 seconds. THE LAWS, each measured: THE LAYOUT
LAW (Jackson) -- _code_section computed every candidate and kept the
smallest, so a two-valued flag paid a full zstd pass over 100M byte-
aligned codes to lose to a 250 KB bit-pack; narrow codes skip zstd and
lay the bits down. THE ARROW LAW -- the pandas string path costs ~450
B/row (45 GB, 387s for one column); strings dictionary-encode in arrow
memory, sort in arrow, remap in numpy. CHUNKED PACKING -- the true peak
was _pack_codes building an N x bits uint64 matrix (16 GB); chunked,
byte-identical: 45 GB -> 9 GB per string column. THE DRIVER -- one
process per column reading its own column; heavy-first with lighter
columns FILLING THE GAPS; a byte budget; every blob written the moment
it lands, in completion order (the reader keys by name). SELF-HEALING --
a pool the kernel kills halves and re-queues. THE COMPLETENESS LAW --
the retreats had silently lost URL and Referer and reported success
(the cause: pop before result). THE CARDINALITY LAW -- judge sequences
by distinct count, not span (CounterID). THE FRONT-CODER IN NUMBA (15s
-> 2.2s, byte-identical). THE CODE-STREAM LEVEL 19 -> 9 (7-13x for ~6%;
the inline decision stays at the archival level: the suite caught the
flip). THE ENCODER MEASURES ITSELF -- workers report peak RSS, a fresh
process per column (ru_maxrss is process-lifetime), bounded learning.
THE COMPRESSION POLICY -- trained dictionaries give nothing after front-
coding; zstd threads halve the code stream. On beating zstd at ground
level (Jackson's question): its internals are a decade of tuned C; the
frontier we own is ABOVE it -- the right representation per column
class reaches the entropy floor before compression matters; the layout
law is the proof. Runs: 1595s -> 980s -> 771s -> 635s, 7.02 GB, peak
57.7 GB under the 128 GB wall, 12/12 ClickBench queries exact vs the
reference encode. OPEN: EventDate/EventTime as dt=3 need the reference
encode's conversion (config, not a bug); point fjdb at the governed
segment; per-class levels; SELECT * ... ORDER BY LIMIT star expansion.


## THE ENGINE ON ITS OWN INGEST (2026-09-12/13): an encoder's choices are query-time laws
Pointing fjdb at a governed segment was the true test of the encoder.
Correct at once (102/103) and SLOW (72 wins vs 91): the reference's
clocks are STAIRCASES and ours were blocked frames, because a
staircase only exists when rows arrive time-sorted. LAWS: DECLARED
CASTS (the operator types day counts and epoch seconds); THE GRID LAW
(diskpair declines mismatched frame grids by name -- g-exyear2); THE
CLUSTER ORDER (cluster_by in the streaming encoder: one permutation,
every worker gathers through it); THE CLOCK LAW (a clock that repeats
is a key -- dictionary + staircase -- never a sequence; a clock that
never repeats is a genuine sequence). RESULT: hits_gov7, megaboard
103/103 -- every query exact on our own ingest for the first time --
faster on 80, median 2.66x; the window family back (w-runmin 2.06s vs
23s); cq-notexists 0.41s. ROW ORDER IS A COMPRESSION PARAMETER
(Jackson's reading): every compressor we have is a same-as-the-row-
above detector; sorting chooses the neighbours; there is one order for
the whole table and every column competes for it. Time order gave the
clocks their staircase and scattered the sessions the file order kept
local (+1.8 GB on the near-unique ids). The secondary key cannot buy
locality back at second resolution (~70 rows per tick); the choice is
which RESOLUTION comes first, or an order-independent representation
for bursty ids (the differentiator shelf). bench/order_lab.py is the
instrument. Also learned: cb25db's catalog had pointed at the July
pre-guard segment all along; ~65 GB of historical encodes cleared.


## SEAL 2026-09-13 (late): the honest standing of the governed encode
gov7 (time order, level 9, all 105 columns): on the 25 columns the
reference carries, OURS IS SMALLER (5.09 vs 5.37 GB); the other 3.68 GB
is 80 columns the boards never touch -- the earlier +1.8 GB claim
compared file order to time order on the bursty ids and was never a
comparison against the reference. gov7 IS single-coordinate time order
(EventDate is a function of EventTime; the second key reorders
nothing). Megaboard cold pass 80 wins / 2.66x, WARM pass 77 wins /
2.08x, 103/103 exact both times. The gap to the reference (91 / 4.69x)
is a family of SearchPhrase <> '' filters ~10x slower (f-neq 0.33s vs
0.02, g-where 0.36 vs 0.04) plus IsLink/IsDownload flags where the
layout law's pack lost to a zstd that catches time-ordered zero runs
(2.5 -> 10.2 MB): candidate-selection and door-level work, NOT order and
NOT sidecars. NEXT: per-query diff on the SearchPhrase family (which
door served on each segment), let zstd win narrow codes when it wins by
4x, then item 2 (concurrency).


## THE CHUNK LAW, and the engine on its own ingest -- settled (2026-09-13)
The SearchPhrase <> '' family was 7x slower on our segment with the
same door and the same encoding. PROFILE BEFORE THEORY: one zstd
decompress of 0.31s per query vs nine tiny ones -- the DICTIONARY, not
the bitmap; binary-search probes decompress the chunk holding each
value; the reference has 368 frames of 16,384, ours was one 6M-value
frame because the arrow string prep never copied _prep_column's chunk
flag. Every big string dictionary (URL, Referer, Title, SearchPhrase)
paid a full decompress on every lookup, LIKE and <> ''. Three
theories before the profile (the second coordinate, the row order,
cold sidecars) were each partly plausible and each wrong. RESULT on
gov8 (cluster_by=['EventTime'], chunked dictionaries): megaboard
103/103 both passes, faster on 86, median 3.44x, TOTAL WAVE 76.3s vs
the reference run's 79.9s -- a segment built in eleven minutes by our
own ingest now beats the borrowed one on wall clock. Remaining: nine
coin-flips within 0.05s of duck, and g-len (LENGTH() as a group key,
0.34s vs 0.04). NEXT: THE NARROW-CODES LAW (codes as uint32 at the
source -- the per-column peak the learner reports is the candidate
zoo over 800 MB arrays; halving the base unlocks full-width encodes),
then g-len, then concurrency.


## A: MAKE IT A DATABASE (2026-09-13): the wire
Decision (Jackson): A first -- concurrency, crash safety, a wire, errors --
because the goal is the official benchmarks and a submission is a
working database: install, load, run the 43 queries three times through
the system's own command line, report load time and storage. THE WIRE,
first form: bin/wdb (init, load with --cluster-by/--cast, sql/file/shell
in table|csv|tsv|json, explain, tables, stats, audit, compact, flush,
vacuum; exit codes 0/1/2/3); the lifecycle exact from the command line
on a database that did not exist a minute earlier. THE CLICKBENCH KIT
(benchmark/clickbench: install.sh, load.sh, run.sh, queries.sql).
Measured: a CLI query cost 3.5s before any work (imports + JIT loading)
-- the floor would swamp every sub-second query. THE WIRE, second form:
 keeps one warm engine (HTTP: POST /sql, /explain; GET
/health, /tables; DML; one query at a time under a lock -- no
concurrency claim yet);  is a stdlib-only client:
3.5s -> 0.078s per query. THE WATCHDOG: the engine runs in a child; an
OOM-killed or crashed child is restarted; the client retries; THE LEDGER
prints wall, RSS and SQL per query. RESULT: the full kit through the
wire, 43 x 3: 42 answered, hot total 10.0s, one null -- COUNT(DISTINCT
UserID) grouped by 6M SearchPhrase values, a transient intermediate the
governor does not bound (the in-process board has always errored there
too). NEXT: a bounded group-distinct organ for that query (the last
ClickBench null); crash safety (journal + recovery); a concurrency
harness; then the load step as the kit measures it, and the first
results.json.


## THE SERVER-KILLER, CONVICTED (2026-09-14): profile before theory, and when the profiler cannot see it, use the signal
The kit's engine child died idle, seconds after 'serving'. Three theories
(the second coordinate, cold sidecars, COUNT(DISTINCT)) fell to
measurement. The instruments that named it: the ledger printing the SQL
BEFORE a run; a cgroup monitor with top processes by RSS (8 -> 127 GB in
eight seconds on AVG(length(URL)) GROUP BY CounterID); a 30 GB address-
space limit (under it the query succeeded at 7.3 GB -- a different door
explodes, and its MemoryError reads as a decline); faulthandler on
SIGUSR1 fired by a shell watcher at 40 GB, the only thing that names a
frame inside a C call holding the GIL: the fused cascade's
np.asarray(typed_dict) with no dtype -- THE WIDTH LAW: a list of 6M
bytes becomes a fixed-width S<maxlen> array, tens of GB in one call.
Twenty-four such sites; the fix at the source: a string dictionary is
an object array, never a list. Also: the dictionary map made universal
in _eval_rows, cached once per query. RESULT: the first fully clean
ClickBench run through the wire -- 43/43, 0 nulls, 0 restarts, hot
total 10.4s, median 0.159s, slowest 0.89s; cold run 85.3s (births) --
the load step should absorb the births before submission. LESSON: the
kill pattern in the same command line as the launch matched itself,
twice; kill and launch in separate calls, permanently.


## A, COMPLETE (2026-09-13): make it a database
THE WIRE (bin/wdb; wdb serve + a stdlib client: 3.5s -> 0.078s per
query); THE WATCHDOG AND THE LEDGER; CRASH SAFETY (the rename law
everywhere, fsync, recovery on open, the crash harness: 3/3 survive
exact); CONCURRENCY PROVEN (the catalog is the truth -- a long-lived
engine re-reads it when its stamp moves; the swap window retries once;
a child never outlives its watchdog; 50,813 answers / 0 wrong / 0
errors / 0 restarts under DELETE->compact and racing births); ERROR
SURFACES (syntax / error / unsupported / timeout / failed with exit
codes); TIMEOUTS AND CANCEL (reply first, then a clean death; a lost
reply is never re-run); THE WARM STEP (births are part of load). THE
FIRST results.json -- the full ClickBench protocol end to end: load
866s (758 encode + 101 warm), storage 9.93 GB, 43/43, hot total 10.5s,
median 0.153s, cold run 1 67.7s. The four cold spikes (Q20, Q22, Q27,
Q28 at 11-16s -> 0.2-0.5s hot) are the LIKE-on-URL/Title queries paying
the joined text buffer, a RAM-only shelf a separate warm process cannot
hand to the server: PERSIST IT AS A SIDECAR under the registry and run
1 becomes cold I/O. Also convicted on the way: THE WIDTH LAW AT THE
SOURCE (a string dictionary is an object array) -- found with
faulthandler on a signal after three theories failed. NEXT: the joined-
buffer sidecar; the 12s AVG(length(URL)) general-scan face; then B --
multi-segment as the normal state.


## THE CLICKBENCH BOARD ON OUR OWN INGEST (2026-09-13): 43/43, faster on 42, median 9.45x
The referee comparison on the governed segment (hits_gov8: time-
clustered, chunked dictionaries, declared casts), cold pass with births:
43/43 exact, 0 false, FASTER ON 42 OF 43, median ratio 9.45x. The one
duck wins is Q40 (149ms vs 112ms, cold). The previous standing on the
borrowed segment was 42/43, faster on 24, median 3.35x. What changed:
THE CHUNK LAW (string lookups stopped decompressing 6M values), THE
WIDTH LAW at the source (the 43rd query answers), the universal
dictionary map, the cluster order with the clock staircases. The kit
run through the wire: hot 10.5s / median 0.153s / cold 67.7s; the
board says the engine underneath is 9.45x duck at the median. Board of
boards now: TPC-H 14/14 (12 wins) . Megaboard 103/103 (86 wins, 3.44x)
. ClickBench 43/43 (42 wins, 9.45x) . H2O groupby 10/10 (7) . H2O joins
8/8 (8) . Scope 100/100 (71, 2.47x) . JOB 113/113 (2 -- the one we lose)
. Dates 35/35.


## B, STEP 3 (2026-09-13): THE MERGED-DICTIONARY VIEW -- from 8.06x to 1.00x in one session
STEP 1, the instrument (bench/board_segments.py: the same rows as one
segment and as five, 100 constructs, exact vs duck, timed on both)
found SEVEN SILENT WRONGS on the shipped multi-segment path in five
minutes (has_agg knew five aggregate types; SUM(DISTINCT) as a plain
partial; lost integer emission) -- a second fixture finds what the first
cannot, third time this month. STEP 3, the union (src/wdb_union.py): K
sorted dictionaries merge once per column; a remap table per segment;
codes() as the concatenation of the remapped streams, int32, resident
across queries; everything else derives; what a union cannot serve
declines BY NAME. The truth test first (13/13 columns exact). Routing
in three moves, each visible on the board: the general scan (83 ok);
the controller with the precedence law (windows via the new door, set
ops 204x -> 0.91x); _solo_segment RETURNS THE UNION -- the nineteenth
gate opened for the join engine and every scope-stage door at once.
Then the last 1.57x: point reads instead of full-column materialisation
per group (9.5s -> 0.8s), and the union serving values_at and
resident_values. RESULT: 99 ok / 1 hole / 0 wrong, 16.5s -> 16.4s,
1.00x. Two laws from the suite: A VIRTUAL TABLE NEEDS A BIRTHMARK (a
sidecar born under one union must never be read by another: hash of
paths, mtimes, sizes in the name), and A NAMED DECLINE CAUGHT BY
 BECOMES A WRONG ANSWER (_code_of_literal turned the
union's decline into 'literal absent -> 0 rows'); grep for that pattern.
Jackson's observation that segmented ran FASTER in four families was
the union's resident int32 code streams, not the CPU: the single-segment
path should keep decoded codes resident under the shelf too. LEFT: one
hole (both tables multi-segment), the registry naming rule for union
sidecars, then steps 2, 4, 5.


## SEAL 2026-09-13 (late): B COMPLETE, and the like-for-like DuckDB reference
B, all five steps, in one day: the instrument (7 silent wrongs found in
five minutes); tiered compaction; the union (8.06x -> 1.00x); the last
hole and the union in the registry; the harnesses on the five-segment
realm -- which found A REAL ATOMICITY GAP (a DELETE across five segments
was five atomic writes; readers between them saw partial deletes) and
produced THE TABLE-LEVEL PRESENCE FILE (one file per table, one rename
per statement). Eight silent wrongs surfaced by the multi-segment realm
today, every one in code a fast door had hidden. THE DUCKDB REFERENCE
(benchmark/clickbench/duck_reference.sh): DuckDB run the way ClickBench
runs it -- native load with their conversions, 43 x 3, fresh process,
cold cache -- on the same pod, same protocol as our kit: WaveDB wire
cold 67.7s / hot 10.5s vs DuckDB cold 58.4s / hot 51.9s; hot WaveDB
faster on 43/43 (median 2.38x); cold faster on 34/43 (median 1.87x);
storage 9.9 GB vs 26.3 GB; load 866s vs 90s. The cold total is worse
only because of three queries (Q20 16.2s, Q22 15.7s, Q27 11.6s): the
LIKE-on-URL/Title family paying the RAM-only joined text buffer on first
touch -- without them ~24s vs ~50s. NEXT SESSION (before the JOB board):
(1) THE JOINED-BUFFER SIDECAR under the registry (cold 67.7s -> ~25s);
(2) the encoder's candidate-at-a-time lever (load); (3) the wire floor
(~0.1s per query: HTTP+JSON+dispatch -- a persistent connection or raw
socket); (4) the rare swap race, now instrumented; (5) the single-
segment path keeping decoded codes resident like the union does
(Jackson's observation). Then C: the key-space fixpoint.


## THE FRAMED TEXT AND THE PARALLEL NEEDLE (2026-09-14): the cold LIKE family
The four cold spikes were _typed_dict decoding 6M front-coded URLs into
Python objects (16s) and joining them into a haystack -- every process,
never written down. THE TEXT BUFFER SIDECAR (born once under the
registry); raw at first: 2.35s warm but ~8s from disk, I/O-bound on
3.4 GB of uncompressed URLs. THE FRAMED TEXT: zstd frames split at
VALUE boundaries (a needle never spans two values, so never two
frames), exposing exactly what the scan needs -- find(needle, start)
and slicing; URL 3.4 GB -> 519 MB, Title 1.26 GB -> 230 MB. THE
PARALLEL NEEDLE: frames decompress in threads (zstd releases the GIL); a
two-pass numba kernel (count, then fill into exact offsets -- the first
version zeroed a 33 MB buffer per frame, 28 GB of memset) finds every
occurrence across cores. URL LIKE: 16.3s -> 1.36s; Title: 11.3s ->
0.52s. THE KIT vs DuckDB native, same pod: WaveDB cold 67.7s -> 44.2s
(now under duck cold 58.4s; faster on 34/43, median 1.87x), hot 10.7s
vs 51.9s (42/43, 2.51x). Left cold: Q27 (12.2s, the 6M-group
SearchPhrase aggregate -- not LIKE) and Q28 (a draw). Loose thread: the
warm step re-birthed fresh text sidecars once (a level-9 birth then a
level-3 one); a warm step must never rewrite what it finds.


## SEAL 2026-09-14: the cold face, most of it
THE FRAMED TEXT + THE PARALLEL NEEDLE (LIKE: 16.3s -> 1.36s), THE BIG-
DICTIONARY LITERAL (a sorted dictionary is a binary search: the fused
door had decoded 6M URLs to test URL <> '' and then declined; Q27 12.6s
-> 2.5s), THE LENGTH SIDECAR (length() over a dictionary as a V-scale
array, no strings). Kit vs DuckDB native, same pod: WaveDB cold 67.7s
-> 42.9s (duck 58.4s), hot 10.8s (duck 51.9s), 42/43 hot. LEFT: Q28
(13.2s cold) -- the Referer regex group decodes the dictionary for the
regex; it wants the framed text per frame, never 6M Python objects (the
typed_dict-from-frames attempt was 30s: building the strings faster is
the wrong lesson; not building them is the right one). Cold-run
comparisons on this pod are +/-1s noisy (no drop_caches). NEXT: Q28;
the warm step must never rewrite a fresh sidecar; the encoder lever;
the wire floor; then C.


## THE COLD FACE CLOSED, AND THE LOAD STORY OPENED (2026-09-14, late)
THE REGEX-GROUP SIDECAR + the census: Q28 13.2s -> 2.0s (the prefix-run
road WAS serving and WAS the 15s; its V-scale result is written once;
the door recounted 100M rows the registry had already counted). THE
CLEAN KIT RUN vs DuckDB native, same pod: WaveDB cold 20.4s / hot 10.2s
vs 58.4s / 51.9s; cold faster on 36/43 (2.44x), hot 42/43 (2.86x); no
query over 2.6s cold; cold was 67.7s two sessions ago. Verified: a warm
pass never rewrites a fresh sidecar; the warm pass is 30s. THE LOAD:
866s vs duck 90s. THE SHARED VIEW: _code_section made nine int64
copies of the same stream; one view, transients freed: ClientIP 10.8 ->
5.8 GB, byte-identical. Five learner classes; budget three-quarters;
THE HONEST STRING START (480 B/row: three strings together killed the
pool -- the self-healing retreated and re-queued, as designed). 866 ->
732 -> 671s, up to 8 in flight, zero retreats. THE STRING TAIL: 103/105
columns done at 515s; the last 150s is OriginalURL and Title alone --
the strings' own working set (47 GB each: the front-coder holds the
column as Python objects) is the load story's next lever, then the
candidate zoo's time (each candidate compresses the full stream).
NEXT: the string columns' memory, the wire floor, then C.


## THE STRING TAIL, CLOSED (2026-09-14): the arrow path was never running
Title alone: 170s, 57 GB. _arrow_string_prep concatenated the chunked
column; 100M titles exceed arrow's 2 GB string offset space; the
'offset overflow' was swallowed by , and
every big string column had fallen to the Python-object path for weeks
-- the 9 GB arrow path we had measured never ran on the columns that
mattered. THE LARGE-STRING CAST (before the concat AND before the
cluster gather's take, which overflows the same way): 170s -> 65s, 57
-> 29 GB. THE PARALLEL FRAMES (dictionary zstd frames in a thread
pool). Per-chunk encode + unify_dictionaries: measured 5+ minutes on
one core, rejected. LAW: A FALLBACK THAT IS SILENT IS A FAST PATH THAT
ISN'T THERE -- the decline now prints by name, and it named the second
overflow within a minute of existing. THE LOAD: 866 -> 732 -> 671 ->
405s, zero retreats, 12 in flight, children peak 39 GB; exact vs duck.
Thirty hours to six and three-quarter minutes in one week. What is
left in the encoder is time, not memory: the candidate zoo compresses
the full stream once per candidate, and arrow's dictionary_encode +
sort_indices are single-threaded (15s per big string). NEXT: C.


## C, THE FIRST ACT (2026-09-14/15): the key-space fixpoint, the stream, isolation first
JOB was a death by a thousand cuts (113/113 at 137.8s vs 13.2s; no
query over 10s; the median ~1s vs 0.1s). The bill on 17e named the
disease: the sweep asked cast_info (36M rows) about name and title
before the one selective filter had reached it -- 5.5s of blind passes.
JACKSON'S LAW: PEMDAS IN ISOLATION FIRST, THEN THE CONJOINED SPACE.
Phase 1: every table does its own work and reduces to KEY SETS. Phase
2: the join columns collapse into SHARED VALUE SPACES (union-find); a
space is the intersection of its restricting members; a table whose
spaces shrank re-derives its rows -- the smallest signal first (the
table whose restricting space is smallest, not the smallest table), the
giant last, each pass costing the rows matched. Delta-driven rounds
fall out of the structure. THE STREAM (Jackson): everything in phase 2
only ever shrinks, so the order signals arrive does not change the
answer -- monotonicity licenses running isolation concurrently and
letting the worklist start on whatever has landed; a late key set is
one more shrink, never a correction. ISOLATION FIRST, PER TABLE: a
table whose own filter is still running is not restricted from the
space yet -- its own cut is deeper and cheaper. The laws that made it
cheap, each from a profile: key sets cached per (table, column, count);
A KEY SET IS A BITMAP, NOT A SORT; A SPACE IS A BITMAP INTERSECTION;
THE PREDICATE SHELF (a local predicate's result on an immutable segment
is a fact, shelved across queries); THE KEY COLUMN ON THE SHELF (decoded
once per process as int32); the smallest signal first within a table's
columns too; the live row list never sorted. THE JOB WARM STEP as the
kit has. RESULT: JOB 137.8s -> 16.8s warm (1.5x duck), 2 -> 23 wins;
JOB-COUNT 149s -> 44.3s single pass; all 226 exact throughout. Measured
truth about the stream: on first touch the overlap is real (4.3s ->
0.76s); warm, the pool buys nothing because the predicate evaluators
hold the GIL -- ISOLATION SHOULD BE NUMBA KERNELS (nogil) so the
concurrency is real; that is the next organ. Then the per-step kernel.
