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
