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


## C, THE SECOND ACT (2026-09-15): from the tax to the band
After the key-space fixpoint the JOB gap was a TAX -- ~60ms on each of
113 queries, no monster. Each cut came from a bill: THE STRING RANK ROAD
(MIN/MAX over an inline column as one argmin over int32 ranks: 17e 0.32
-> 0.18); THE APPLIED SPACE (a space already applied is not applied
again) -- which EXPOSED a real bug: a late isolation landing had
overwritten the keep and the redundant re-application had been silently
repairing it (JOB 5c: '#1' for '11,830,420'); A LANDING INTERSECTS. THE
PLAN CACHE and THE FRESHNESS CACHE: measured neutral within the board's
+/-1s, kept -- and the measurement said the glue is not where the second
lives. The scratch bitmap: rejected (np.zeros is a lazy calloc). JOB-
COUNT had no warm step: 39.9s was mostly first-touch. 2,440 deepcopy
calls per query rebuilding stripped conjuncts: in the plan now. DECLINE
BEFORE THE WORK, twice more: the fused door factorised 4.2M inline
names to build a code lookup for a LIKE and then declined (3.85s ->
0.011s with THE PREFIX ON THE RANK ROAD: an anchored prefix is a range
of the sorted order); the road engine rebuilt a 36M-row cast_info ->
name pointer every COUNT query to re-discover that IMDB has orphan
foreign keys -- THE REMEMBERED REFUSAL (a .no birthmark; the second-ask
memory had lived in db.__dict__, which the per-query flush wiped, so
cast_info's roads were never born). Jackson asked whether isolation
runs concurrently: measured -- structurally yes; really on first touch
(4.3s -> 0.5s); warm, the pool buys little because the predicate
evaluators hold the GIL; THE NEEDLE IN THE INLINE LIKE fixed the biggest
holder; isolation as nogil kernels is the general answer. STANDING: JOB
warm 12-13s vs 10.5-11 (31-39 wins); JOB-COUNT warm 15.0s vs 8.8s; all
226 exact throughout; from 137.8s / 149s a week ago. NEXT: the regex
verify in numba; the weights during the settle; the JOB first-touch as a
load step; the per-step kernel.


## SEAL 2026-09-15: C's standing, and the floor named
JOB warm 12.6s vs duck 10.9 (38 wins); JOB-COUNT warm 15.2s vs 8.9;
all 226 exact; from 137.8s / 149s a week ago. The per-step kernels
(item 3) measured at parity; the profile named the floor: ORCHESTRATION
-- ~15 Python-orchestrated steps per query at ~10-15ms each -- and the
clean-segment checks asking the filesystem 53 times per query. A DML
WRITE MOVES THE CATALOG STAMP; THE CLEAN VERDICT is memoised under it;
THE REMEMBERED EXISTENCE. The boards now sit at their +/-1s noise for
this design. THE NEXT LEVER: FEWER, BIGGER STEPS -- restrict a whole
level of the worklist together (every pending table whose signal has
settled) in one pass, ~15 steps to ~4; the same algorithm at a coarser
grain; the only thing expected to move a board beyond the noise short
of a compiled orchestrator. Also owed: the regex verify in numba; the
weights during the settle; JOB first-touch as a load step.


## SEAL 2026-09-15 (afternoon): the step count was not the floor; the static gate
FEWER, BIGGER STEPS (the level-wise fixpoint): measured neutral (2.54s
-> 2.46s over 12 queries) and WRONG on 19d -- it converged early
(cast_info cut once and never revisited; a superset fed the emit).
Removed; the one-at-a-time loop stands. Kept: compaction by sorting
only when n is tiny (a 4M sweep is ~4ms, a 250K sort ~20ms), the live
list built once at a landing and only for a table that shrank. Net 3%.
THE STATIC GATE (Jackson): ruff/vulture over the hot modules -- zero
real bugs, real dead code removed, 2 false positives in the bug classes
(numpy != None) excluded by rule; a one-second lint with zero false
positives now runs before the suite. The codebase's waste is
architectural, not lexical. STANDING: JOB warm 12.6s vs 10.9 (38 wins),
JOB-COUNT warm 15.2s vs 8.9, all exact. NEXT: JOB-COUNT's weights
during the settle; the ClickBench results.json refresh (the union, the
roads, the memoised verdicts are all unmeasured there); the JOB
submission notes.


## C, THE THIRD ACT (2026-09-17): the plumbing was the floor -- JOB won
Jackson brought independent research: a join-chain cache over SQLite
(per-value position chains; cached RELATIONSHIP chains for hot multi-
key semi-joins gave 30-90x; curves only on synthetic data; coarse maps
lose on scattered data; a few percent of the file). His claim for us:
with chains for the whole join, the settled space is one serial
overlay -- 10-30ms. THE CENSUS: 113/113 JOB queries acyclic; the two-
sweep licensed everywhere -- and the worklist already beats its bound
(1,272 steps vs 1,954). Not fewer steps, then. THE PROFILE OVER ALL
113: the settle work was ~1.5s for the board, 1.2ms a step; the other
8s were CONVERSIONS BETWEEN REPRESENTATIONS -- his instinct, in our own
plumbing. THE INDEX LIST RIDES ALONG (the shelf already held what a
landing rebuilt in two 36M-row passes); a column's max is a fact
about the column (S[-1] is free); UNDER AN UNMOVED STAMP NOTHING
MOVED (open_segment); A FULL TABLE HAS NO LIVE LIST (290 MB of arange,
discarded). 12.7s -> 9.06s in-process. The win exposed a hole: the
stamp's 0.25s ttl hid an in-process INSERT from its own process. AN
IN-PROCESS WRITER MOVES ITS OWN STAMP; the ttl (20ms) covers other
processes only; harness PROVEN. STANDING: JOB warm 8.6-9.1s vs duck
10.7-11.4 (66-72 wins) -- WON; JOB-COUNT 10.6 vs 8.4 (41 wins). What
we already had under other names: value chains = the reverse road;
dimension chains = the predicate shelf; AND/OR = the spaces. What we
do not have: the RELATIONSHIP chain -- the propagated space, keyed by
the predicates that reached it. That is next: THE SETTLED-SPACE CACHE.


## C, THE FOURTH ACT (2026-09-17, later): the relationship chain, in the engine
THE REUSE CENSUS: 31% of the giant postings gathers across the 113
repeat exactly, and the repeats are FIRST HOPS (it.info = 'rating' ->
movie_info_idx 29 times; rt.role = 'actor' -> cast_info 13). THE
SETTLED-SPACE CACHE: a junction's key set after a cut is a fact about
(segment, column, THE PROVENANCE OF THE KEEP -- its local predicates
plus the cuts applied, order-free because a keep is an intersection);
shelved across queries, DML moves the stamp. Lessons: A LIVE SET IS
NAMED BY WHERE IT CAME FROM, NOT BY ITS BYTES (hashing 1.5M keys per
cut cost more than the kernel saved); nested symbolic provenance grew
exponentially (8.6s -> 20-48s) -- THE CACHE IS BOUNDED; depth one
missed the junction's usual first cut, which is through a dimension --
THE SECOND HOP; THE BIG FIRST HOP SHELVED WITH ITS KEEP (a hit does no
N-scale work). JOB-COUNT: one word's difference -- MIN's emit is one
look, COUNT counts CHAINS with a weights walk; its extra cost was 1.8s
of the road engine planning and declining every query (THE REMEMBERED
DECLINE: adaptive routing, the fixpoint first once the road engine has
declined a shape) and a walk that road-walked keys, used np.add.at,
and allocated LUTs over the KEY SPACE for a hundred live keys (sparse
LUTs sized to the live keys). STANDING: JOB warm 7.0s vs duck 9.9
(85 wins); JOB-COUNT 8.5s vs 8.1 (67 wins); all exact; from 137.8s /
149s and 2 wins two weeks ago. The settle profile is flat: postings
1.1s (unique gathers), key sets 0.7, intersections 0.8, dispatch 0.8.


## C, THE FIFTH ACT (2026-09-18): the census, the pin, and both JOB boards won
THE CENSUS ACROSS EVERY BOARD after a week that changed the engine under
all of them: TPC-H 14/14 (13 wins), Megaboard 103/103 (87 faster, 3.26x),
ClickBench 43/43 (warm 6.9s vs 34.4), H2O joins 8/8, Scope 100/100,
Dates 35/35, H2O groupby 9/10 (q10: the working-set gate refusing a 50M-
row result -- a refusal, not a loss) -- and ONE REGRESSION: four join
families WRONG on the 5-segment realm. Wednesday's clean-verdict memo
keyed a table's segment set on (db, table, stamp); segment partials PIN
a table to one member while they iterate a union, and the verdict
memoised under the pin was served for the whole table: sums low by one
segment of five. A 1696-test suite passed with it in -- nothing ran a
join over a union AFTER a partials-served query. THE PIN IS PART OF THE
VERDICT; the test exists and fails without the fix. LAW: a memo's key
must carry everything the verdict depended on -- a pin, a stamp, a
provenance; a key that omits one serves the wrong world silently.
THEN JOB-COUNT: aimed at the walk (the weighted first-hop cache), the
profile said the walk's LUT builds were 0.48s over 113 queries; the
second was keys_at at 1.0s, ALL FROM RESTRICT -- the fixpoint's cut of a
big table took the road walk on mc/mi/ci.movie_id because keys() shelves
a key column once per process but the big-table branch of the cut only
looked in the per-query keycache. THE SHELVED COLUMN, IN THE CUT TOO:
one lookup. SPARSE TO BUILD IS NOT SPARSE TO LOOK UP (the sparse LUT
only when keys are 64x rarer than the space). RESULT: JOB-COUNT 7.2s vs
duck 9.2 (71 wins) -- WON; JOB 5.8s vs 9.5 (90 wins). Both boards taken;
137.8s / 149s two weeks ago. THE WARM STEP STANDS ALONE (wdb warm) and
its parser bug (comment lines ate the statement after them: 113 of
226). Measured: a warm in another process does not carry -- the shelf
(key columns, predicate lists, settled spaces) dies with the process;
--warm is the steady-state protocol; PERSISTING THE SHELF is the lever
on the cold number (~20s), nothing on warm. Suite 1697/0.

## THE SWITCH (2026-09-18): sidecars are an extension the operator turns on
Jackson's question was scope: "the file is 9-10 GB -- what persists?"
THE CENSUS answered it and corrected a claim of mine. JOB: 771 MB of
segments and 2,219 MB of sidecars on disk (cast_info's reverse road
alone 1,159 MB against a 162 MB segment) -- 3x the data, once per
database, not per segment. ClickBench: the 8.78 GB submitted at load is
the segment alone; after the 43 queries the directory is 11.3 GB --
2.9 GB born by queries (text zones 849 MB, the range guide 471, two
posting lists 808, censuses 290, text offsets 211, group-by codes 234).
I had said ClickBench was sidecar-free. It was not. The published board
(ClickBench repo, c6a.4xlarge, 2026): DuckDB 20.46 GB, ClickHouse 9.42,
Umbra 8.31, Parquet 14.78 -- and data_size is measured after load,
before any query, so query-born files never show. Jackson's ruling:
"allowing 25% of our file size to be a sidecar in a persistent manner
is like claiming compression but secretly storing things"; and the
design: THE OPERATOR DECIDES -- the engine ships small and ClickHouse-
class on joins by default; the extension is a switch, and with it on we
beat DuckDB and are still smaller than DuckDB.
THE MECHANISM: catalog.json carries 'sidecars': on|off; a new database
is born OFF; a catalog without the key (born before the switch) is on;
WDB_SIDECARS=0|1 in the environment overrides the catalog (an A/B
without touching the realm). `wdb sidecars DB status|on|off|build
QUERIES|drop`; `wdb load --warm` turns it on (asking for a warm IS
asking for the extension); drop deletes every derived file and the
birth ledgers and flips off. may_birth() refuses first on the switch;
every other birth site (twenty-two, across seventeen modules -- the
registry's docstring said "births go through may_birth" and five did)
asks births_on(dbdir) before it writes. THE SENTINEL: while off,
Database.run remembers the directory before a query and compares it
after; a newborn is a missed gate -- removed and named on stderr,
raised under WDB_SIDECAR_STRICT=1, which the suite sets. The registry
learned five families it did not know (fk-pointer, lower-map, number-
line, tier-shelf, and that .cluster/.cube are written by the ENCODER --
loaded, not derived). test_sidecar_switch: off births nothing and
answers duck-exact on a thirteen-query corpus; drop returns the loaded
database; the environment overrides; the CLI round-trips.
OFF IS NOT COLD: the first JOB board with the switch off ran 342s --
off meant "rehash every query", because a refused road left the per-
query cache and the shelf never saw it. THE ROAD ON THE SHELF: a
reverse road or a pointer that is not persisted still lives for the
process under the shelf's ceiling, keyed by both parents' identity
(path+size+mtime). Off means nothing on disk, not nothing in RAM.
MEASURED, on hardlink copies holding only the loaded files (JOB 771 MB
/ 22 files; ClickBench 8.37 GB / 2 files; strict sentinel; nothing
born): JOB first pass 53.4s vs duck 11.0, 113/113 exact, 41 wins (on:
6.0s, 93 wins, 3.0 GB). ClickBench, each query its own process: cold
197.6s, warm 23.2s vs duck 37.1s, 43/43 exact, 30 wins (median 1.83x);
the one file the run added was routing_ledger.jsonl, 24 KB, a journal
the registry counts as data. Suite 1702/0.


## THE COORDINATE ROAD (2026-09-18): Jackson's blocks, and rank retired
He asked what the sidecars hold, bit by bit, and how they pop. Four of
the eight families were the same object: a list of ascending row
numbers per key, stored as flat 32-bit integers, read as "find the key,
walk its group" -- nothing ever indexes into the middle of a group. His
proposal: store each row as (block, position-in-block), pop the block,
pay fewer bits because the address is local. THE ARITHMETIC: a 65,536-
row block makes a position 16 bits; the block id is paid once per
(key, block) as a 32-bit header; a block holding more than 4,096 of a
key's rows is cheaper as an 8 KB bitmap (Roaring's container rule). So
a key with k rows in b blocks costs 32b + 16k bits against 32k -- half
when the rows share blocks, worse than flat when every row is alone in
its block. MEASURED FIRST, on the real roads: role_id 145 -> 4.6 MB
(31x, all bitmaps), info_type_id 59 -> 2.5 (23x), person_id 2.0x,
person_role_id 2.5x, SearchPhrase's posting list 4.9x, CounterID's
3.1x -- and cast_info.movie_id WORSE (15 rows per movie in 15 blocks).
THE RULE AT BIRTH: whole file against whole file (boffs + headers +
payload vs offs + order; u is shared), coordinates when smaller. Two
of nineteen JOB roads stay flat. THE WALK is what it was -- searchsorted,
then each container: add the block base to sixteen-bit positions, or
iterate set bits -- one numba pass, sized from the headers (a bitmap's
rows are its popcount; no per-key row count stored). boffs is int32
(containers, not rows). u keeps its width -- the coordinate road had
been widening int32 keys to int64 and title.id came out 45 MB against
40 flat; a leak in the new form, not the idea. RANK RETIRED from every
birth: its one reader (keys_at) has the dictionary at 0.2-61 ms; an old
road's rank is read if present. Two lessons in the birth path: a road
asked again within two seconds of its own birth was reborn (the
registry's negative-answer window is for other processes' births --
wdb_sidecar.born flips it), and the stream's isolation threads birthed
the same road twice (one lock per road). RESULT, roads born fresh on a
hardlink realm, all exact: JOB 6.3s vs duck 11.6 (92 wins; 6.0-6.3 /
90-93 before), JOB-COUNT 7.5 vs 8.2 (71 wins; 7.2 / 71 before), a
fresh process 27.0s against the flat realm's 31.1 -- less to page in.
ROADS ON DISK: 2,000 MB -> 673 MB (3.0x); the realm 2,990 -> 1,499 MB
with the pointers still to be reborn (165 MB). Suite 1707/0.


## THE REFEREES (2026-09-19): ClickHouse and Umbra, measured on our pod, saved
"We probably crush ClickHouse on everything" was expected, not measured.
The measurement, same pod, same parquet, three runs per query, kept in
bench/referee/*.json so it is never re-run: CLICKHOUSE 26.10 loads hits
in 32.6s to 9.41 GB (their published 9.42 -- the load is apples to
apples); ClickBench warm 12.4s against our 6.5 and duck's 33.7 -- we
take 27 of 43, it takes the trivial ones (Q00/01/19) and the
ORDER BY/LIMIT tail (Q21-23, Q37-42); JOB 34.9s against our 6.3 and
duck's 11.6 -- we take 112 of 113. UMBRA 26.09 (the image unpacked and
run through its own loader, no docker on the pod; memlock refused, so
its writeback buffers ran degraded by its own account): hits in 273s to
8.01 GB (published 8.31); ClickBench warm 5.58s -- Umbra takes 26 of
43, we take the heavy string and group-by queries (Q04/05/08/13/32-34
by 10-60x), it takes the point and LIMIT queries by the same margins;
JOB 2.78s -- Umbra takes 96 of 113, 2.2x ahead in total, up to 25x on
10c. IMDB on disk: ClickHouse 1.94 GB, Umbra 2.66, ours 0.77 loaded /
1.66 with the extension. THE SENTENCE THAT IS TRUE: smaller than every
published engine but Umbra with the extension off, faster than DuckDB
and ClickHouse on both boards, and behind Umbra -- slightly on
ClickBench, 2.2x on JOB. Umbra is the bar now; its JOB queries are 8-
25 ms where ours are 60-240, which is the settle, not the read.


## THE FLOOR (2026-09-19): four milliseconds under every query
Q29 -- the ninety sums -- was Umbra's closest win (37.7 ms to our 46.4
on the board). Jackson's instinct was a stored per-column sum. The
profile said the sum was already free (the census dotted with the
dictionary) and the query was 9-12 ms in-process, all of it plumbing:
ninety projection names generated by sqlglot with a deep copy each
(31 us a name; 8.6 without the copy), hidden_rewrite deep-copying the
whole tree before looking for anything to rewrite (5 ms of the 9), and
the dictionary's values re-fetched one at a time every run (2,159
calls). Q06 -- two dictionary lookups -- was 6 ms: the block-stats
sidecar re-read from its .npz every query by the old cold-truth law
(4 ms on the network volume; the shelf is the lawful resident form
now, keyed by the segment's identity), and THE LEDGER: the routing
ledger opened and closed a file per query -- 2 ms, on EVERY query on
every board, for a year. One open handle per ledger path. Measured:
Q00 3.0 -> 1.4 ms (Umbra 1.6), Q02 9.0 -> 1.5 (42), Q06 6.2 -> 1.4
(1.4), Q29 8.8 -> 4.0 (37.7). The board: warm 6.47 -> 6.40s, cold
34.1 -> 31.8, 43/43 exact; against Umbra 20 of 43 from 17. Suite
1707/0. LAW: a per-query cost that is small is paid times every query
on every board; the floor is a number to profile like any other.


## THE SCAN WITHOUT THE GIL (2026-09-19): Q21-23, and the board under Umbra's total
The hunch was "LIKE then gather is our territory"; the profile said the
LIKE itself was cheap (the dictionary is the haystack; 8,326 of 18.3M
URLs contain google, 15,911 rows) and everything after it was not.
_scan_flag over 100M codes: zstd releases the GIL but flag[raw] and
nonzero held it, so fourteen threads ran one at a time -- 221 ms, when
decompression alone scales to 61 ms on fourteen. The membership test
is a nogil kernel; the flag is packed to bits (18 MB of bools missed
the cache on every lookup; 2.3 MB sits in L2); the frame is sliced as
a memoryview, not copied. 221 -> 76 ms. THE SCANNED CODES: the scan
keeps the column's codes at the hits, so the caller's codes_at over
the same positions -- which decompressed every frame again, 93 ms for
16K rows -- is a lookup (0.2 ms), exact on the full set and on any
subset. Q23 (SELECT * ... LIMIT 10) was 535 ms for ten rows: six enc-1
columns are one zstd frame each and a point read inflated all 100M
codes; zstd streams, so the prefix to the highest row asked is what is
inflated (the first ten hits of a scan are early rows), and a thread
pool for ten frames cost more than the frames (serial under sixteen).
535 -> 54 ms. MEASURED on the board, all 43 exact: Q21 390 -> 163 ms
(Umbra 49), Q22 798 -> 514 (59), Q23 552 -> 54 (37); warm total 6.40
-> 5.41s -- UNDER Umbra's 5.58 for the first time -- cold 31.8 ->
23.8; still 20 of 43 head to head. Suite 1707/0. WHAT REMAINS on Q21/
Q22 is the floor of the form: 257 MB of zstd'd codes inflate to 400 MB
at ~65 ms on fourteen cores, and Umbra reads byte-aligned columns at
memory speed. That is an ELECTION question (zstd's 55 MB against a 4x
faster scan on the hottest string column), not a kernel question.



## THE PACKED FRAMES (2026-09-20): enc 18, Jackson's "bitpack, then zstd the 1s and 0s"
The question was what zstd does with a bit-packed stream instead of
byte-aligned codes. Measured on cbdb's hottest wide columns (65536-row
frames, level 9): Title 224.3 MB packed+zstd against 225.7 today; URL
284 against 257 (worse -- the byte lanes were what zstd was matching);
UserID 308 against 246. So it is a CANDIDATE, not a dress: enc 18 runs
in the election for every column of 17-32 bits and wins on bytes alone
against zstd, blocked frames and bitpack. Frame = LE bit-pack of 65536
codes (eight slack bytes) then one zstd stream; header
[18][bits][BR][nfr][offs u32 x nfr+1]; meta carries pbits/poffs and
DELIBERATELY no boffs/cwidth, so every enc-3-only reader falls back to
_raw_codes instead of misreading a frame. Readers: pk32_pack/unpack/
gather/flag_hits kernels (five-byte windows, nogil), point reads inflate
a frame PREFIX to the highest row asked (as enc 1), range reads inflate
touched frames, the flag scan unpacks and tests membership in one loop
and keeps THE SCANNED CODES, = / <> / IN drive through the same scan.
THE REDRESS (wdb_retype.redress): the engine's own parse now records
each column's blob span and code_off, so a segment can be re-elected
column by column with the dictionary and every other column
byte-copied -- the A/B tool this needed, and the measurement that
mirrors never drift: twelve of fifteen wide columns came back
byte-identical. THE ELECTION on cbdb: Title 225.7 -> 224.3 MB,
ClientEventTime 267.7 -> 263.7, and HID 337.5 -> 337.0 over bitpack --
by 0.15%, for a frame inflate on every point read where bitpack's random
access is free. Jackson: "isn't bitpack with no zstd better?" It is.
THE BITPACK GUARD: against plain bitpack the packed frames must win by
10%; against zstd and blocked frames, which already pay the inflate,
strictly smaller elects. HID wears bitpack. URL,
UserID, Referer, both hashes, both IPs and the rest kept their dress.
Segment 8,778.6 -> 8,773.2 MB with the guard. THE BOARD, sidecars settled, 43/43
exact both: warm 4.87s -> 4.63s, cold 28.9 -> 23.1, 20 of 43 against
Umbra -- Q22 (Title LIKE) 392 -> 413 and head-to-head 407/490 vs
506/401: inside the noise, no win, no loss. The first A/B read 5.38s and
was wrong: the copy was birthing its own sidecars mid-board (Q27 +615 ms
on a column never touched). LAW: an A/B against a fresh directory is
not settled until its sidecars are; run it twice and read the second.
Q17 (GROUP BY UserID, SearchPhrase, LIMIT without ORDER) swings 52 to
259 ms run to run on BOTH directories -- a routing bimodality, its own
hunt. Jackson: "the ones we did it on, like Title, should improve --
bitpacked zstd is faster AND smaller." ISOLATED, Title's LIKE scan on
fourteen threads: enc 3 inflates in 48.5 ms and tests in 63.1; enc 18
inflates in 27.1 (300 MB of bits out instead of 400 MB of u32) and
tested in 65.9 -- the first kernel assembled every code from FIVE BYTE
LOADS and gave the inflate's win back. THE WORD VIEW: the frame as
aligned u64 words, one load per code and a second only when the code
straddles a word (safe inside the eight slack bytes for any prefix the
point reader asks): 48.8 ms. The packed frames scan Title 23% faster
than the u32 frames AND are smaller. Q22 in-process, ten runs each,
alternating directories: min 339 -> 303 ms, medians 394 and 393 --
the run-to-run spread of Q22 (340-463) is four times the win, which is
why no board could show it; Q21 (URL, untouched) 87.8 vs 86.7, the
control. Board, second run, 43/43: warm 4.72s, 21 of 43 against Umbra.
Suite 1713/0. VERDICT: correct, and the scan is faster where it wears
it; small on cbdb (0.06% of the segment, one column on the board)
because these codes are hashes and IDs with little structure for the
packing to expose. The candidate stays in the election for the data
where there is more.



## THE VANILLA LAW (2026-09-20): the number that is the engine
Jackson asked whether it was time to submit to ClickBench, and what
would embarrass us. The rules: query result caches off; "creation of
pre-aggregated tables or indices, projections, or materialized views is
not recommended"; hash tables fine "unless they mimic query result
caching". Every board we have run was with sidecars ON, and the cbdb
directory after a board holds `Referer.rg-<hash>.npz` (494 MB, Q27's
grouped result keyed by its regex), `MobilePhoneModel__UserID.gdc` (the
answer to Q10), the posting lists, the text indexes -- born by the
queries they answer. That number is not the engine. Jackson: "that was
always kinda a concern I had with them." MEASURED, the switch off,
strict sentinel, a directory holding only the segment and the catalog:
warm 21.1 s. DuckDB 34.0, ClickHouse 12.4, Umbra 5.58. Faster than
DuckDB on 30 of 43, than Umbra on 8. The profiles said the 16 seconds
were not the engine's floor but reads written assuming their sidecar
exists, building it when it doesn't -- slowly, in Python, every query:
AVG(UserID) 1,097 ms (block stats for 3,052 blocks in a Python loop,
forgotten after the query), WHERE UserID = k 1,525 ms (the whole
group-by census, an argsort of 100M codes, to look up one code; the
frame scan never got the chance because rows mode demanded ORDER BY
the cluster column), COUNT(*) WHERE URL LIKE 19 s on the first query
of a process (18M URLs decoded into Python objects to join a
haystack). THE LAW (wdb_sidecar.may_build): with the switch off a read
may USE a structure that exists on disk but never BUILDS one to answer;
it declines and the streaming reads serve. What is the query's own work
-- a count per code, a per-group distinct -- is computed the way a scan
would, with kernels, and forgotten. Block stats and the scalar count
decline; gbcount counts with THE PARALLEL CENSUS (bincount_par: per-
thread boards and a reduce, 300 ms one-thread numpy -> ~70) and orders
by a radix sort on u16-capped counts instead of the 645 ms argsort; the
frame scan takes unordered rows queries (any order is lawful; the
driver's positions in row order are one) and `=` rides the nogil flag
scan; the group-mix read uses the MSD scatter lane that already existed
one module over (2.09 s serial walk -> 0.4); LIKE with no text buffer
walks the dictionary's own front-coded bytes in plike_fc with a PREFIX
CARRY (a match inside the shared prefix is still a match; only the new
bytes are searched); the routing ledger writes nothing when off.
MEASURED, same directory, same rules: 21.1 -> 13.25 s, 43/43 exact;
33 of 43 against DuckDB, 10 against Umbra; Q01 188 -> 35 ms, Q02 488
-> 119, Q03 1,097 -> 421, Q07 237 -> 100, Q15 1,393 -> 548, Q19 1,525
-> 259 (83 in-process), Q20 1,162 -> 69, Q25 787 -> 400, Q33 1,136 ->
418. Suite 1713/0. WHAT REMAINS on the vanilla bill: Q09 +1.66 s,
the COUNT DISTINCT UserID family (Q08/Q10/Q11/Q13, +2.6 s together),
Q31/Q32, Q15/Q33, Q03 -- and the first LIKE per process at ~2 s (1.4
of it joining a 2 GB decoded dictionary). THE SUBMISSION LIST, still
open: their cold run drops the page cache on a 500 GB gp2 volume; 32 GB
of RAM; the JIT warm-up belongs in load time; the official queries.sql
verbatim; encode time on the record; one afternoon on a c6a.4xlarge.



## THE STATISTICS OF THE LOAD (2026-09-20): metadata per block, written by the encoder
Jackson: "I already thought this was on disk -- we are just collecting
metadata per block, a few bytes per block, which won't amount to much."
It is now. Block statistics -- row count, non-null count, sum of the
dictionary values, min and max code, per 32,768-row block -- are written
by the ENCODER as the last step of a load, one file beside the segment
(<seg>.stats.npz), classified as DATA with the segment, the cluster and
the cubes: `wdb sidecars drop` keeps it, the sentinel ignores it, the
switch does not govern it, `wdb loadstats DB` writes it for realms
loaded before. The block-stats read reads it before any birth is
considered. What it is NOT: the per-code histogram (the gbc census) --
that is V-sized, 136 MB for UserID, not a few bytes per block; it stays
a sidecar. The Python loop that computed the stats (0.91 s on UserID,
3,052 blocks) is a kernel now (block_stats, blocks across threads),
which also serves the on-mode first touch. MEASURED: cbdb, 77 eligible
columns, 9.3 MB, 20 s of load. Vanilla board, switch off, strict
sentinel, 43/43 exact: 13.25 -> 12.52 s; Q02 119 -> 2.5 ms, Q03 421
-> 2.7, Q06 33 -> 2.7. Against DuckDB 34 of 43, against Umbra 12; even
with ClickHouse's 12.4 on this box. LAW: what every engine computes at
load is not a sidecar; what only a query would ask for is.



## THE KEY IS ITS CODE (2026-09-20): under ClickHouse, vanilla
Jackson: "we have the superior atomic parts; a lot of our issues is we
don't have our best foot wired in" -- take the biggest loser, say it in
kid lang, itemize the time. Q09, kid lang: for every region, how many
hits, add up AdvEngineID, average the screen width, count the DIFFERENT
people; show the ten busiest. 1,781 ms vanilla, DuckDB 506. The line
items: inflating four columns 225 (real); turning RegionID's codes into
values with a 100M fancy-index and then astype, twice, 630 (waste --
the code IS a small integer); three one-thread np.bincounts with float
weights 360 (should be one parallel pass); the two-pass distinct
scatter 353 (real); AdvEngineID's enc-5 decode 45; assembly 100. A
thousand of the 1,781 were conversions that converted nothing and sums
on one thread. THE FIX: the group id is the dictionary code as stored,
decoded only at emission (a shelf born in the raw-value id space is
translated through the int table, exactly); SUM/AVG numerators are
group_fold_dict -- per-thread boards over the two code streams and the
dictionary, no value array ever built (30 ms for both); the count is
the parallel census; the target's codes go to the scatter as stored
(the kernel widens per row -- no int64 copies); gd_pass2_count keeps
per-THREAD rows via get_thread_id instead of per-bucket (4096 x 9,040
x 8 = 296 MB zeroed and reduced serially). Q09 1,781 -> 702. The same
law applied to the distinct family: Q08 868 -> 612; Q10's small key
space (166 phone models) left the per-slice sort, whose one giant slice
ran on one thread (336 ms), for the MSD lane balanced by target: 540 ->
172; Q11's filter IS the tiered column's planes (the non-default rows
and their codes, no 100M compare): 754 -> 486; the top-k over millions
of groups is a partition extended to the lim-th count's plateau, exact
to the full sort (_top_sel); _gdc_save2 asks the switch before sorting
a podium it will not write. THE BOARD, vanilla, 43/43 exact: 12.52 ->
10.43 s. UNDER CLICKHOUSE's 12.4 on this box; DuckDB 31.9 (37 of 43);
Umbra 5.58 (12 of 43). Suite 1714/0. WHAT REMAINS: Q08/Q13/Q09 at 0.6-
0.7 s each (the inflate and the scatter -- the atomic parts now), Q15/
Q33 (a 17M-bin census: 1.1 GB of per-thread boards to zero), Q31/Q32,
Q18, Q16 (two-key GROUP BY at DuckDB parity, 914 ms).




## THE BAR (2026-09-20): stream the argument, stop early
Jackson, on Q16 (kid lang: for every person, for every search phrase
they typed, how many hits; the ten busiest pairs), 947 ms vanilla with
a 100M-row inflate of both keys, a 100M sort and a 100M np.argsort of
the counts: "it should just stream the person argument instead of
inflating all the values, because chances are we have the counts for
the most common user id and therefore we can almost certainly stop
early after inspecting like 100 of the top users." THE BAR: a top-k
over counts never sorts the counts. topk_bar lowers a bar from the
maximum (step doubling from (max-min)/64) until at least k counts
clear it -- each probe one parallel _count_ge pass over an int32
board -- then sorts only the survivors above the bar, exact to the
full stable argsort. The census that feeds it is bincount_par (per-
thread int32 boards, numpy fallback below 4M rows); the survivors'
keys sort in sort_keys_par (4096-bucket MSD partition, per-bucket in-
place). Applied: Q16 947 -> 510 (tripletop: user census, pair census,
bar, survivors), Q15 548 -> 378 and Q33 418 -> 346 (gbcount's bar
lane: a census, a bar, counts >= 2 kept -- no order array over the
group space), Q18 1,085 -> 694 (the survivors' sort). pairtop's
vanilla pass takes np.arange(N, uint32), not int64 (400 MB less).
THE EXCEPTION LIST IS A STATISTIC: a near-unique column's rows whose
value repeats (WatchID: 8 rows in 100M) is the same species as a
distinct count or a discovered unique constraint -- a fact about the
column, not a query's answer -- so the encoder writes <col>.rep into
the load statistics (dictionary columns with V >= N/2, V >= 1024, list
under 0.75% of V) and pairtop reads it before the .ptrep shelf. Q31/
Q32 vanilla 440 -> 6-8 ms in process (the 33 s regeneration of the
statistics is load time, paid once). Under WDB_SEQ_NARROW_OK the
suite's sequence column is mode 4, not a dictionary: no list, by
rule. FAIL-LOUD: write_for_segment no longer swallows the list's
exceptions; eligibility is decided inside differentiator_rows.
THE BOARD, vanilla, clean (nothing else of ours on the box; host load
average 42 from other tenants): 43/43 exact, 9.22 s, 36 of 43 faster
than DuckDB (36.4 s this run). 10.43 -> 9.22. Q31 4.7 ms, Q32 23 ms.



## THE BLOCK DICTIONARIES (2026-09-23): enc 19, pointer compression one level down
Jackson, on the inflate: "store each distinct user id value as a bitpacked array and populate the
database with pointers to these values so that the scope of V collapses to the distinct count of
V; decode becomes a single jump." The value side already was this (the dictionary); the cost was
the POINTERS: UserID's 100M row-order codes, enc 3 (u32 per row, zstd per 65,536-row frame), 246
MB, 122 ms to inflate -- 108 of it zstd itself, 16 lanes slower than 8. The same move one level
down: a block of 65,536 rows sees only ~19,600 of the 17.6M users, so each block keeps its sorted
list of the global codes present (first code + bitpacked gaps -- Jackson's "alphabetical order"
shares the leading bits) and each row a 15-bit pointer into its block's list. Run-length would
not have done it (mean run 1.06 rows, RLE 389 MB): the win is few DISTINCT per block, not repeats.
Every block starts on a u64 word -> encode and decode block-parallel, no shared words; a point
read decodes its block's list only as far as its highest pointer.
UserID: 246.0 -> 246.9 MB (+0.4%), full decode 121 -> 47 ms.
THE ELECTION (Jackson's general rule): enc 19 replaces an INFLATING dress (zstd 1 / blocked 3 /
packed frames 18) when its bytes are within 5% of that dress's (E19_SLACK). The census of the 26
inflating columns (ratio enc19 / current): ClientIP 0.989, RemoteIP 0.991, UserID 1.003, FUniqID
1.012, URLHash 1.027, IPNetworkID 0.896 -> take it; RefererHash 1.082, Title 1.155, RegionID 1.19
(Jackson: "leave it"), URL 1.22, the timings 2.4-4.5x -> keep zstd. The line falls where it should:
near-unique columns whose blocks see few distinct values tie zstd; skewed or clustered mid-V
columns are zstd's home ground (entropy coding, runs) and stay. Sizes are computed exactly before
a byte is written (one sort per block, both 16,384 and 65,536 rows sized, the smaller wins).
Gates that listed tags by name: sampletop (Q17's LIMIT-without-ORDER sampler) and lenagg now admit
19 (they only call codes_at / _raw_codes); valtopk and the wherescan frame drivers call enc-3/18
frame internals and decline, exactly. Q17 found it: 22 -> 429 ms until sampletop admitted 19.
A/B on the UserID queries, in process, best of 5, results identical: Q08 579 -> 530, Q10 218 ->
104, Q11 559 -> 440, Q13 604 -> 501, Q22 473 -> 360, total 4.75 -> 4.35 s. THE BOARD, vanilla,
UserID redressed only (cbdb_e19): 43/43 exact, 9.22 -> 8.42 s, 39 of 43 faster than DuckDB (37.1
s). Q27/Q30/Q33/Q37/Q39 read +35..63 on the board and -47..+11 in process A/B: host noise (load
average 38 from other tenants). Suite 1719/0.



## THE LISTS ARE NOT BUILT (2026-09-23): the vanilla law, second sweep
Scoping the short queries against Umbra (Q07/Q19/Q37-Q41 at 30-95 ms vs its 2-6) found the first
run of Q37 at 6.1 s with the data in page cache: import 3.2 s, segment open 0.6 s, and 5.0 s of
wdb_funnel._plist -- a 100M-row argsort building every CounterID's position list, "RAM only" with
the switch off, then memoized on the Segment for the life of the process (drop_derived never
cleared _plistmemo). Every later funnel query rode it; the board's "warm" 30-90 ms were warm-INDEX
numbers. THE VANILLA LAW forbids exactly this, and the first sweep missed it; wdb_fpm (the frame-
presence map: a full decode + a Python loop over 1,526 frames) was the same species.
THE FIX: positions(seg, col, code, lo, hi) -- the lists when they may serve (loaded, on disk, or
the switch allows the birth); otherwise the load statistics' per-block min/max pick the blocks
that can hold the code (CounterID = 62: 386 of 3,052), clipped to the staircase window computed
FIRST, and the frame scan reads only those. _plist asserts may_build (FAIL-LOUD); fpm declines
under vanilla; pairfold's plist lane requires plist_ready; sampletop (Q17) draws from the sparse
dress's own stored rows -- a census of its 13M literal codes, rejection-sampled small codes (a
permutation of 6M eligible codes cost 115 ms), one flagged pass for every slice, each COMPLETE so
each count is exact. The mode-4 trap the test caught: a sequence's load min/max are VALUES while
its codes are positions -- positions() prunes only dictionary columns (modes 0/2).
THE BOARD, vanilla (cbdb_e19), 43/43 exact: hot 8.42 -> 8.61 s (the honest price of finding the
rows: Q14 224 -> 396 via pairfold, Q36 72 -> 163, Q40 95 -> 126, Q42 34 -> 61; Q37 98 -> 77);
COLD 130.8 -> 75.5 s (Q37 5.9 s -> 566 ms, Q38 6.0 s -> 527, Q40 6.1 s -> 664, Q17 3.9 s -> 601,
Q14 4.1 s -> 1.2 s). 38 of 43 faster than DuckDB. Suite 1721/0 (+2: positions exact and windowed
on a sequence and a dictionary, _plist refuses, fpm declines, no memo left; the vanilla sample's
every count equals DuckDB's).
WHAT REMAINS OF THE SWEEP: ~30 Segment memos and module caches outlive the query in wdb_qmem's
terms (Jackson's law: "when the outermost run() returns it is as if the query was never there").
To be measured, not read: run the 43 in one process and list what still holds data after each.



## JACKSON'S LAW, ENFORCED (2026-09-23): the hot path was standing on memos
Jackson: "the base that needs to get developed is the cold places ... it's easier to make some
kind of hot shortcut that patches a problem rather than addressing the underlying cold problem."
The cold table (each query's first run, files in page cache; Umbra/ClickHouse first runs of the
Sep-19 referee, DuckDB native first runs in a fresh process each) said it plainly: WaveDB 75.5 s,
Umbra 5.8, ClickHouse 17.8, DuckDB 47.9; 0 of 43 first runs faster than Umbra. Every query paid
~410 ms before its first byte of work, and Q20-Q28 paid 4-9 s against 60-430 ms hot.
THE START: Q00 (COUNT(*), 1.9 ms hot) took 407 ms first -- opening the segment (a Python parse of
the layout through ~249K memmap slices) and first-use imports. ClickBench restarts the database
before each first run and requires its caches cleared; opening files and loading the program is
starting, not caching. Database.open now loads the program (every read module, the parser) and
opens every segment; nothing decoded from the data. Q00's first run 407 -> 5 ms. (A startup that
pre-built decoded state would refill what the rule says is empty -- not done, by rule.)
THE COLD NO: stairs() spent ~180 ms decoding and diffing UserID to learn it is not a staircase;
the load statistics' block min/max refute it without a byte. Q19 first 677 -> 168.
THE LAW had drifted: wdb_qmem says "when the outermost run() returns it is as if the query was
never there", but drop_derived cleared a NAMED list, and the list had fallen behind -- position
lists, LIKE flags (Q22's run 2 skipped a 1.3 s match), regex derivations (Q28's 3.9 s), sparse
planes, scanned codes, censuses, _dictbytes and other lazy column keys, SQL-keyed plan caches,
an lru parse cache whose own docstring said it was removed, and a shelf holding roads, keys,
ranks and predicates. Now: drop_derived sweeps every underscore container but _synth/_tdict/
_civil_lut_cache and every lazy '_' column key; flush clears the database memos and keeps only
the shelf's VOCABULARY (decoded dictionaries); the module caches are registered. THE LAW'S
WITNESS (wdb_qmem.audit, baseline = the loaded program) lists residue; WDB_QMEM_STRICT, set by the
suite, raises on any: 1721/0 under it, and all 43 ClickBench queries leave zero residue.
THE BOARD, the honest one (cbdb_e19, vanilla, 43/43 exact):
  first runs 75.5 -> 53.8 s (Umbra 5.8, ClickHouse 17.8, DuckDB 47.9); first-run wins vs DuckDB
  24/43, ClickHouse 11, Umbra 5.
  hot 8.61 -> 29.96 s (Umbra 5.58, ClickHouse 12.40, DuckDB native 19.60): the memos were most of
  the hot story -- Q20 123 -> 2,649 ms, Q21 138 -> 2,881, Q22 433 -> 4,186, Q23 59 -> 3,095, Q28
  348 -> 6,237, Q29 6 -> 249, Q10 138 -> 486, Q34/Q35 ~180/96 -> ~640. The string family (Q20-Q23,
  Q28, Q29) is ~22 of the 30 s: every one expands a whole string dictionary before it can look.
WHAT THE BASE NEEDS, now that hot is cold: (1) the string family on the dictionary AS STORED (LIKE
and REGEXP over the front-coded chunks, no join of 18M strings); (2) the two kernels numba cannot
cache (group_fold_dict, grid2_count: dynamic globals) recompile in every process -- ~2.4 s of Q09's
first run; (3) then the distinct family (Q08-Q14) and Q34/Q35.



## THE TWO TIERS (2026-09-23): keep the legal hot, the cold stays the heart
Jackson: "if our hot time is legal for the hot board then we should keep what we have but just
make it not interfere with the cold aspect, and the cold aspect will still be the focus."
ClickBench draws the line: "Caching source data (e.g. buffer pools) is fine"; caches "near the
end of the query execution pipeline ... similar to query result caching ... should be disabled".
TIER 1 (source data decoded, may outlive a query): dictionaries (_tdict, _dictbytes, joined text,
string arrays), sparse planes, staircase steps, inline streams. It only KEEPS what a query already
produced; the official cold run restarts the process, so it is empty there; WDB_HOT_KEEP=0 is the
pure-cold A/B. TIER 2 (what a query computed -- LIKE flags, regex groupings, counts, position
lists, frame maps, scanned hits, ranks): dies, witnessed strict. Program (plans, the parse memo,
now handing out a COPY of the tree) persists. One process, the 43 in order: tier 1 on 41.4 s, off
41.8 s -- it neither helps nor hurts a first touch.
THE BOARD (cbdb_e19, vanilla, 43/43 exact):
               this morning   the law   the tiers    Umbra   ClickHouse   DuckDB native
  first runs       75.5 s      53.8 s     52.1 s      5.8      17.8          47.9
  hot               8.61 s     29.96 s    19.91 s     5.58     12.40         19.60
The hot that is left above this morning's is exactly the illegal memos: Q20-Q23 at 1.0-1.6 s (a
LIKE memo had them at 59-433 ms), Q28 at 6.1 s (a regex memo had it at 0.35). That is the string
family, and it is the cold work: LIKE and REGEXP over the dictionary as stored.



## THE THREE READS (2026-09-23): the only ways to touch a string
Jackson: "first thing is just diffrentiation, this is not returning the value and only requires
reading enough information such that you can tell one thing apart from another, and then there is
reading the value to return it as a result, and last but not least there is identification ...
we should read only the bytes needed to make an identity known inside a block by seeing that it
cannot be any other string in the block."
wdb_strings.py holds the three:
  DIFFERENTIATE (differentiate): the codes. Dictionaries are sorted, so equality, grouping, order,
    MIN and MAX are all differentiation -- no string byte is read.
  IDENTIFY (identify_contains): a decision per DISTINCT string, taken chunk by chunk on the
    front-coded dictionary as stored ([cp u16][sl u16][suffix], restart every 128). Each chunk
    (a zstd frame of 16,384 values, asserted to begin on a restart) is decided in parallel by
    plike_fc_serial, which carries the shared prefix instead of rebuilding each string. No join,
    no string array. The rows then take the answer by code.
  RETRIEVE (retrieve): the answer strings only, after the result is known.
Sites moved onto identify: wherescan _like_flags and engine like_mask_dict (LIKE / NOT LIKE; the
null bin is false under NOT), dict_charlens (chunk plan on the leaf pool), and Q28's REGEXP host
road (_derive_runs_one_read: per-chunk lengths + host runs, then an FNV-1a label hash, argsort and
an exact byte comparison inside each hash bucket -- 3,009,018 groups, 0 mismatches vs the old road).
_save_sidecar now asks may_build BEFORE it builds 3M labels (it spent 3.4 s to be told no), and
_emit takes first positions with a kernel instead of np.unique.
THE LAW, tightened: drop_derived also sweeps column-meta keys absent from the parse-time _shape
(charlens hid from the underscore-only sweep); the witness checks the same.
MEASURED (fresh process each, cbdb_e19, vanilla, exact vs DuckDB):
  URL LIKE '%google%' identify 2.1 -> 0.51 s
  Q20 first 805 ms, hot ~630     Q23 first 1.35 s, hot ~580     Q22 first ~3.6 s, hot ~1.2
  Q27 first 1.19 s, hot ~0.9     Q28 first 7.4 -> 2.06 s, hot 6.1 -> 1.88 s
REJECTED by measurement: fc_charlens2 (a two-pass length kernel) -- Title 0.73 -> 1.58 s. Removed.
WHAT IS LEFT in identify: the chunk's zstd decompression now dominates. The next cut is the
dictionary's own layout, so identify can read the decisive bytes without inflating whole chunks.
THE BOARD (cbdb_e19, vanilla, 43/43 exact; numba cache warm -- a first board right after the
kernels file changed paid recompiles in Q08-Q14 that no A/B reproduced):
               this morning   the law   the tiers   the reads    Umbra   ClickHouse   DuckDB native
  first runs       75.5 s      53.8 s     52.1 s      34.5 s      5.8      17.8          47.9
  hot               8.61 s     29.96 s    19.91 s     14.70 s     5.58     12.40         19.60
  first-run wins: DuckDB 27/43, ClickHouse 12, Umbra 5.  hot wins: DuckDB 21, ClickHouse 14, Umbra 8.
  moved (first / hot, ms): Q20 3703->860 / 1047->700; Q21 4182->1282 / 1143->793; Q22 5317->1815 /
  1638->1281; Q23 4344->1432 / 1032->614; Q28 7266->2080 / 6111->1958; Q27 1647->1180 / 458->874.
Q27's hot rose because the tighter sweep caught 'charlens' (URL's per-string character lengths),
which had hidden from the underscore-only sweep and carried across queries. It is computed from
the dictionary, not the dictionary itself: tier 2. The 874 ms is the honest number; the 458 was
the leak. A/B (three fresh processes each, prev vs this commit): Q01, Q08, Q10, Q12, Q13, Q14
identical; only Q27 moved.



## THE THREE STREAMS (2026-09-23): headers, mask, text
Jackson, on how identification should read a length: "capture the length by having a fixed width
and then measuring the mask of each entry." A chunked front-coded dictionary (aux bit 0x80) is now
stored as three streams per chunk, each kind contiguous on disk:
  HEADERS  <cp u16><sl u16> per entry -- differentiation's positions, and every byte length (cp+sl)
  MASK     one bit per text byte, 1 = the byte starts a character (its top two bits are not 10)
  TEXT     the suffixes back to back
The width is fixed in the READING: fc3_charlens takes the mask 64 bits at a time; a string's length
in characters is the set bits of its suffix plus those of the prefix it shares, carried as a stack
of pieces that empties at every restart (never more than R pieces). No text byte is read.
The interleaved form is rejoined byte for byte (fc3_join) for readers that walk it; point reads
(fetch, values_at) walk headers and text in place -- the restart offsets already index the text:
restart g's text begins at its interleaved offset minus 4 bytes per entry before it in the chunk.
On the new layout the interleaved frame keys do not exist, so an unlearned reader fails loudly.
MEASURED before building (full dictionaries, zstd 9, today's chunk size):
  size   dictionaries 1820 -> 1857 MB (+2.0%); string columns +1.3%; the file 8774.0 -> 8811.2 MB
         (+0.42%). URL and OriginalURL SHRINK (the headers compress apart from the text); Title
         +8.9% and SearchPhrase +4.8% of their columns (Cyrillic: the mask alternates 1,0).
  length per dictionary (inflate + count vs headers + mask): URL 68 -> 12 ms, Referer 40 -> 15,
         Title 31 -> 10 (1/8 samples); SearchPhrase 83 -> 27, OriginalURL 341 -> 37 (whole); bytes
         read URL 63 -> 7.2 MB. 0 wrong everywhere.
  REJECTED: fixed-width slots ON DISK -- padding each entry to its 128-block's longest costs
         2.2-4.3x. The width belongs to the word, not the file.
FOUND ON THE WAY (fixed, committed 9951554): fc_charlens carried continuation counts for a string's
first 4096 bytes only (1 wrong OriginalURL length measured; URL holds 6950-byte strings), and
fc_hostruns held 8192 bytes. Both now hold the format's bound, 131070.
THE INLINE RACE MOVED: mode 5 is chosen when inline is smaller than dictionary + codes. On
test_mode5's shared-200-byte-prefix data the three-stream dictionary (582 KB) now beats inline
(605 KB), where the interleaved one (621 KB) lost: the race picked the dictionary, correctly. The
test now uses high-entropy variable-length data (what mode 5 is for) and a new test pins the race.
A/B, same code, cbdb_e19 -> cbdb_fc3 (bench/fc3_convert.py: every chunk verified to rejoin to the
source bytes), fresh process each, median of 3:
  Q27 first 1158 -> 840 ms, hot 906 -> 668      Q28 first 2258 -> 2131, hot 2099 -> 1989
  Q17 252 -> 224; Q20-Q23 5-17% faster; Q24/Q26/Q33/Q36/Q37/Q39 within +-10 ms.
  A first cut rejoined whole chunks for point reads: Q24/Q26 +12 ms, Q39 +41 -- fixed by walking
  headers and text in place.
WHERE Q27 GOES NOW (845 ms): the length read 193 ms (was ~550); a staircase check on a column that
is not one 247 ms; a second route re-running the aggregation ~300 ms; pool thread start ~90 ms.
Q28 still rejoins each chunk for the host walk -- step 2 reads the host from headers + text.
THE BOARD (cbdb_fc3, vanilla, 43/43 exact; a second run -- the first after the kernels file changed
recompiled Q08-Q14/Q18/Q23/Q30 again, the same signature as before, gone on the rerun):
               the reads   the three streams    Umbra   ClickHouse   DuckDB native
  first runs     34.5 s         34.2 s            5.8      17.8          47.9
  hot            14.70 s        14.08 s           5.58     12.40         19.60
  first-run wins: DuckDB 27/43, ClickHouse 12, Umbra 6.  hot wins: DuckDB 21, ClickHouse 15, Umbra 8.
  moved (first / hot, ms): Q20 860->645 / 700->570; Q21 1282->1179 / 793->620; Q22 1815->1616 /
  1281->1155; Q27 1180->840 / 874->649; Q18 3178->3365 first (not a string query).



## THE FILTER BEFORE THE READ (2026-09-23)
Jackson: "if we know that what we are going to do will cost like 500ms but there is a filter that
is 48ms that cuts it in half then we need to run the filter before it ... our potency score might
not be picking this edge up." It was not, for two reasons found in the source: (1) the fused
path's _potency9 measures prune only for numeric dictionaries -- a string conjunct gets a flat 0.5
and cost 1.0, so SearchPhrase <> '' and a LIKE over 18M URLs look alike; (2) more deeply, every
LIKE was decided over its WHOLE dictionary before any row was filtered, so no order could save a
string read -- potency only ordered the row checks that came after.
MEASURED on cbdb before building: Q22's SearchPhrase <> '' then Title LIKE leave 7,128 rows whose
URLs are 2,450 distinct strings of 18.3M (1,598 of 143,298 restart groups); Q23 walked in time
order meets its 10th match at row 214,125 -- 77K distinct URLs; Q21's SearchPhrase <> '' leaves 23%
of URLs over 69% of groups (little to gain); Q20 has no other filter.
BUILT:
  plike_sel_fc3 / plike_sel_fc: decide only given codes -- each rebuilt from its restart, groups
    holding no needed code never walked; wdb_strings.identify_contains_at: chunks holding none of
    them never inflated.
  THE POTENCY OF A SUBSET (wdb_strings.at_cost / take_subset_road): a whole pass inflates every
    chunk and decides every entry (the halves measured equal, 292/323 ms); the subset bill is
    (share of chunks inflated + share of entries walked) / 2, taken below 0.8 of the whole.
  wherescan: only the DRIVING like decides its whole dictionary; every other like runs LAST (after
    every code-compare filter) and is decided at the survivors (_like_at).
  firstk: the staircase walk decides the LIKE only for the codes it meets, window by window, with
    a running bill; past 0.8 of a whole pass it decides the whole dictionary once and walks on.
FOUND ON THE WAY: the text-buffer road of _like_flags sized its flag without the NULL bin, so a
  residual LIKE over a nullable column indexed past it (IndexError) -- flags now cover every code,
  NULL False; and residual NOT LIKE now excludes NULL rows (SQL: NULL NOT LIKE x is not true).
A/B, HEAD vs this, cbdb_fc3, fresh process each, median of 3 (ms):
  Q22 first 1637 -> 1321, hot 1158 -> 840   (URL at survivors: 2,450 codes, 621/1,120 chunks,
      103,658 entries walked; 210 ms where the whole dictionary was 505)
  Q23 first 1243 -> 929, hot 519 -> 220     (identify at the codes met: 195 ms, was 550)
  Q20, Q21, Q37-Q39 unchanged (no residual LIKE).
WHAT IS LEFT: Q22's 210 ms is inflating 621 chunks for 2,450 strings -- 16,384-entry chunks; step 2's
small text frames bring it to ~1% of the text. Q23 is now its RETURN: 759 ms materialising SELECT *
(105 columns) for 10 rows. The fused path (wdb_join) still decides a LIKE whole when it builds its
code-LUT; none of Q20-Q23 route there.



## THE HOST FROM THE HEADERS (2026-09-23): block-prefix pruning
Jackson: read a block by its most restrictive position -- "if the third byte is like two distinct
letters you might start there." MEASURED FIRST: in a sorted 128-string block the leading positions
are ONE byte each (URL: 1.0 distinct at positions 0-10, 1.4 at 16, 2.8 at 23; Title 1.0 -> 12.9;
SearchPhrase 1.0 -> 17.8), so the block's shared prefix decides its members at once -- sort order
IS the first plane; true vertical planes would cost 4.3x (URL), 3.8x (Title), 2.9x (SearchPhrase)
the dictionary bytes (padding + lost front coding): REJECTED. ClickBench's LIKEs are all unanchored
contains; the position-tied predicate it runs is Q28's host.
fc3_hostruns: while an entry's shared prefix reaches past the host's slash its host is its
predecessor's -- decided by the header; its suffix is looked at only for a newline ('.' never
matches \n) and skipped entirely in newline-free chunks (645 of 1,204 Referer chunks). The previous
string is a stack of pieces into the text; only a BREAK (15.9% of Referer, 21.6% of URL) rebuilds
it. No rejoin of the chunk.
EXACT on the whole dictionaries: 0 mismatched chunks vs the byte walk (Referer 19.7M entries, URL
18.3M) -- breaks, host ends and every label. Also: a non-matching entry's label (REGEXP_REPLACE
returns the whole string) was capped at 8,192 bytes in fc_hostruns -- now the whole string.
MEASURED per chunk, one thread: old rejoin + byte walk 2.72 ms (Referer) / 4.03 (URL) -> new
newline check + header walk 1.10 / 2.11; inflate (1.1 ms) unchanged.
Q28, HEAD vs this, 5 fresh processes: first 2165 -> 1977 ms, hot 2025 -> 1792. Q27 control flat.
A NOTE ON THE POD: nproc and sched_getaffinity say 16 CPUs; os.cpu_count() says 64. The leaf pool
and ~10 other sizings use cpu_count -- 64 threads on 16 CPUs. Measured: inflate + lengths + host of
Referer scales to ~5x one thread by 8 threads and no further (8: 640 ms, 16: 589, 64: 606); reused
decompress buffers gained ~15%, not the ceiling. Numba's parallel kernels likely count 64 too -- to
measure. (ClickBench's c6a.4xlarge has 16 vCPU.)
WHAT IS LEFT IN Q28 (~1.9 s): the dictionary map ~600 ms (CPU-bound at 16 CPUs), Referer's row codes
~270, argsort of 3M label hashes ~260, emit ~300, hashing + exact grouping ~170.



## THE TRUE COLD (2026-09-23): what the boards have not been measuring
Every board's "first run" so far ran after something had just read the same file: the OS page
cache was warm. ClickBench's cold run drops it. The pod refuses /proc/sys/vm/drop_caches, but
posix_fadvise(DONTNEED) evicts one file without root (/workspace is MooseFS over FUSE: 6.7 GB/s
warm -> 0.71 GB/s after eviction). bench/true_cold.py: fresh process per query, open, evict the
database's files and numba's compiled kernels, time the query; fincore counts the bytes read.
THREAD SIZING, checked first: os.cpu_count() says 64, the pod may run on 16; numba already uses 16.
Python pools resized to 16 (a startup hook, no code change): warm first runs 32.8 vs 32.8 s, hot
13.10 vs 13.02; reversed order 32.1 vs 32.1. No effect -- code unchanged. The first pass of that
check exposed the cache: the first process to touch the data took 60.0 s, the next 33.2 s.
THE BOARD, TRUE COLD (cbdb_fc3, 43 queries):  true cold 55.2 s | warm first 34.2 s | hot 12.97 s |
9.4 GB read from storage. The penalty is the string family, and it is BYTES:
  Q28 +3,384 ms (839 MB: Referer's dictionary 613 + its row codes 225)
  Q22 +2,837 (1,217 MB)  Q20 +2,130 (740: URL dictionary headers+text + URL codes)
  Q21 +1,980 (800)       Q23 +1,813 (537)   Q27 +811 (371: headers + mask 58, not the text)
  Q30 +659, Q33 +651, Q34 +535, Q40 +522.
Q28 pulls 839 MB in +3.4 s = ~0.25 GB/s, a third of the file's sequential rate: memmap page
faults read it in small pieces. The referee runs on this pod (Umbra 5.8 s, ClickHouse 17.8 s) could
not drop caches either; their cache state is unknown.



## THE COLD READ (2026-09-23): read like the storage wants to be read
Jackson: the disk-to-CPU path is a hardware bandwidth (7.5 GB/s on his machine); a cold number
must be bounded by bytes over that bandwidth, not by how we ask for them. MEASURED on the pod's
storage (MooseFS over FUSE), cold: one stream 0.66-0.74 GB/s; 4 streams 1.1; 16 streams 1.72;
32-64 streams ~1.8 (the ceiling here). POSIX_FADV_WILLNEED does nothing on this filesystem. Q28 had
pulled 839 MB at ~0.25 GB/s: through the memory map, page fault by page fault.
BUILT: Segment.read_span -- a dictionary frame is read with ONE pread (fc_part, fc_chunk), and the
parallel chunk workers make that many large streams. Segment.warm_span -- a code section about to
be decoded whole (_raw_codes) or scanned (wherescan enc 3 / enc 18 frames) is first read by
parallel 8 MB preads into the page cache; mincore skips what is already resident, so warm runs pay
nothing. (An ADD COLUMN synth has no bytes in the file: skipped -- the suite caught it.)
A/B true cold (3 fresh processes each, median): Q20 2360 -> 925 ms, Q21 2877 -> 1517, Q22 3858 ->
2464, Q23 2677 -> 1848, Q27 1583 -> 1057, Q28 4525 -> 2231; hot unchanged; bytes read unchanged.
THE BOARD, TRUE COLD: 55.2 -> 44.0 s (warm first runs 34.2, hot 13.2; 9.4 GB read).
THE FLOOR: 9.4 GB at this pod's 1.8 GB/s is ~5.2 s for all 43; at 7.5 GB/s, ~1.25 s. What is left
over warm (~10 s) is spread: Q22 +923 ms, Q20 +593, Q23 +529, Q40 +527 (96 MB -- not bytes: another
read path), Q30 +514, Q18 +474, Q16 +448, Q15 +407, Q09 +406 -- the random-access code gathers
(enc 19 gathers, codes_at) still fault through the map. Next: warm those spans the same way.



## THE FAIR TRIAL (2026-09-23): every engine truly cold, same pod, same protocol
Jackson: "put the competitors through a fair trial so that we can have an accurate scope for our
progress." The earlier referee numbers (Umbra first runs 5.8 s vs hot 5.58) were warm-cache: no
engine could drop the pod's caches. bench/fair_cold.py gives each engine the ClickBench cold run by
the same means as bench/true_cold.py: fresh process / restarted server, its data files evicted from
the page cache (posix_fadvise DONTNEED), query x3 (first = cold, best of the other two = hot).
  Umbra 26.09: fresh umbra-sql per query from the image's rootfs (own loader), ASYNCIO=0 as its
    start script; time = exec + compile as Umbra reports it.
  ClickHouse: server stopped, data evicted, server started; clickhouse client --time. Reloaded fresh
    (30.3 s, 8.76 GiB).   DuckDB: fresh process on duck_native.db (parquet view where needed).
RESULT (43/43 each; score = ClickBench geometric mean of (t+10)/(best+10), 1.0 = best everywhere):
                 COLD total  score  fastest     HOT total  score  fastest
  Umbra            24.4 s    1.50     23          5.5 s    1.34     34
  ClickHouse       29.0 s    1.95      9         15.7 s    3.63      2
  WaveDB           43.2 s    2.39      9         12.9 s    3.08      8
  DuckDB           57.5 s    2.99      2         19.7 s    3.89      2
Cold, WaveDB beats DuckDB clearly and sits 1.5x ClickHouse's total, 1.8x Umbra's. The cold gap is
concentrated: Q09 5.0 s (Umbra 0.6), Q08 3.0 (0.45), Q18 3.6 (1.1), Q10 2.8 (0.19), Q16 1.8,
Q13-Q15 ~1.0-1.5, Q30 1.3 (0.4), Q11 1.1, Q07 0.84 (0.03) -- the DISTINCT / GROUP BY family, 
~20 s of our 43. The string family is now competitive cold: Q20 957 (Umbra 1,229), Q21 1,538
(1,436), Q22 2,479 (1,785), Q23 1,917 (1,961), Q28 2,222 (1,843), Q33 1,505 (1,451), Q34 860 (1,721).
Hot, Umbra is in a class of its own (5.5 s); WaveDB is second (12.9 s) ahead of ClickHouse (15.7).
Suspect: WaveDB Q31 hot 7 ms / Q32 14 ms vs cold 470 / 340 -- carried state to audit under the law.



## HANDOFF (2026-09-23, end of session): the state of the battlefield
Pod left clean: no engine processes running; pod src/bench/tests identical to main (55f2709, 251
files checked); everything committed and pushed. The canonical database is cbdb_fc3 (the three
streams); cbdb_e19 is the interleaved predecessor, kept for A/Bs. Referee data kept on the pod:
ClickHouse data (41 GB), Umbra db (11 GB), duck_native.db (25 GB).
THE MEASURING STICKS: bench/true_cold.py (WaveDB, per query cold) and bench/fair_cold.py (every
engine, same protocol). Warm-cache "first runs" (board_clickbench.py) are NOT cold -- never quote
them as cold again. Scores are ClickBench's geometric mean of (t+10)/(best+10).
STANDING (fair trial): cold -- Umbra 24.4 s, ClickHouse 29.0, WaveDB 43.2, DuckDB 57.5;
hot -- Umbra 5.5, WaveDB 12.9, ClickHouse 15.7, DuckDB 19.7.
OPEN FRONTS, for next session's plan:
  1. The distinct / group family, ~20 s of the 43 cold: Q09 5.0 s (Umbra 0.6), Q18 3.6 (1.1),
     Q08 3.0 (0.45), Q10 2.8 (0.19), Q16 1.8 (0.75), Q13-Q15, Q30, Q11, Q07 0.84 (0.03).
     Includes group_fold_dict / grid2_count recompiling every process (uncacheable numba).
  2. Scattered-position code reads (codes_at, enc-19 gathers) still fault through the memory map
     cold; warm_span covers only whole-column decodes and wherescan's frame scans.
  3. Step 2 of the dictionary layout: small text frames, so survivors-only reads (Q22) and the
     host read inflate ~1% of the text instead of whole 16,384-entry chunks.
  4. Audit: Q31 hot 7 ms / Q32 14 ms vs cold 470 / 340 -- carried state under the law?
  5. The fused path (wdb_join) still decides a LIKE whole and prices string filters flat.



## THE CACHEABLE THREAD COUNT (2026-09-24): the group family was compiling, not reading
Jackson: take on the group family first -- "we have not optimized the cold path here." A cold profile
of Q07-Q18 and Q30 showed kernel COMPILE, not bytes: Q09 6.7 s of 8.1, Q18 4.8 of 5.9, Q10 3.8 of
4.6, Q08 3.6 of 4.7, Q15/Q16 ~1.5 each. numba.get_num_threads() / get_thread_id() INSIDE an njit
kernel embeds a threading-layer pointer; numba will not cache it ("dynamic globals"), so eight
kernels recompiled in every process: sort_keys_par 2.8 s, gd_pass2_count 2.2, group_fold_dict 1.9
(twice in Q09), _count_ge 0.9, plus grid2_count, grouped_sum_codes, _part_scatter_nb, plike_fc.
Proved in isolation: same kernel 1.4 s every run; thread count as an ARGUMENT -> 33 ms from cache.
FIX (c108456): wdb_kernels._nt() reads the count in Python; each kernel became _<name>_nb(..., T)
behind a wrapper of its old name (no call site changed); gd_pass2_count takes its row from its own
prange index over a static bucket split (prange's own split). Arithmetic unchanged.
THE FIRST-BLOCK NO (582ae45): string columns have no load statistics, so stairs() decoded+diffed
100M codes to learn "not a staircase" inside wherescan's DETECT (360-480 ms in Q10/Q11/Q13/Q14);
one counterexample in the first 64K rows refutes it exactly. _dict_ints_at/_dict_ints: one pread
per integer-dictionary chunk (Q15-Q17 fetched 10 UserIDs by page faults).
MEASUREMENT RULE: numba's cache index is keyed on the SOURCE FILE -- editing wdb_kernels.py
invalidates every cached kernel in it. After a kernel change, run the board once (ClickBench's
install step) before measuring cold; the first pass after c108456 showed Q23 +2.3 s, Q36 +1.5 s of
one-time compiles that vanished on the rerun.
THE BOARD, TRUE COLD (fair-trial referees from 2026-09-23):
                 COLD total  score  fastest     HOT total  score
  Umbra            24.4 s    1.54     20          5.5 s    1.31
  WaveDB           28.5 s    2.02     13         13.9 s    3.32   (was 43.2 s / 2.39)
  ClickHouse       29.0 s    2.00      9         15.7 s    3.54
  DuckDB           57.5 s    3.06      1         19.7 s    3.79
Group family cold now: Q07 889, Q08 903, Q09 1393, Q10 704, Q11 976, Q12 646, Q13 928, Q14 865,
Q15 819, Q16 1056, Q17 468, Q18 1102, Q30 1282 ms (Umbra: 31/450/626/193/190/447/627/413/415/752/
465/1073/400).
LEFT: Q07's generated kernel (wdb_exprjit exec'd source: uncacheable, 1.1 s cold) -- fixed kernels
for common shapes, or a persisted compiled-shape cache (a ClickBench rules question: Jackson's call).


## 2026-09-24 -- THE ISOLATED PERSPECTIVE: the generator census, and Q07 off the generator

**The two perspectives (Jackson):** keep two things compiled. Kernels in isolation serve the
simple pipes -- one operation feeding the next, no extra read. A superposition kernel (every
piece as a switchable table lookup, latent fused paths unswitched by the compiler) serves the
queries where fusion pays. Both are compiled once and cached on disk; nothing compiles at query time.

**The census first (bench/jit_census.py):** every board query in a fresh process, every kernel
wdb_exprjit generates wrapped and timed (first call = compile + run, later calls = run).
Of 43 ClickBench queries ONE reaches the generator: Q07, and its shape is the simplest there is --
COUNT(*) per one direct group key, no value, no expression, no mask, no predicate (the <> 0 is
spent before the kernel). First call 737 ms, later calls 6.6-14.8 ms. The other 42 build nothing.
So on ClickBench the superposition kernel has no customer; its customers are the join boards
(TPC-H style SUM(price * (1 - discount)) through wdb_join). Parked until we work those boards;
the census runs there first.

**The route:** wdb_exprjit.grouped_multi -- the single chokepoint, so every caller gets it --
sends (no exprs, one direct key, no mask, no pred) to an isolated cached kernel:
wdb_kernels.count_codes (new; reads the codes in their own width, thread count as an argument so
it stays cacheable) below 4M rows, bincount_par above. Codes past K are not counted, the
generated kernel's contract. _ISOLATED=[True] is the A/B switch.
- np.bincount was the first try: 2.2 ms against the generated kernel's 0.2 ms on Q07's 630,500
  uint8 codes (K=19) -- it widens to int64 first. Serial count_codes 0.8 ms; parallel 0.3 ms.
  Query level 81.8 vs 81.5 ms (noise).

**Measured (true cold, fresh process, files + numba caches evicted):**
- Q07 cold, three single runs: before 922 / 825 / 856 ms, after 191 / 183 / 209 ms.
  On the boards: 903 -> 213 and 903 -> 200.
- Board total, two runs of identical new code: 29.4 s and 27.7 s (was 28.5). The 1.7 s spread
  between identical runs is the MooseFS noise band; Q07's -0.7 s reproduces every run.
  Q31/Q32 rose in both runs but sit inside the old code's own spread (Q31 379-469, Q32 356-443).
- Hot 13.89 -> 13.76 / 13.29 s.
- Suite 1737 passed, 0 failed. Q07 exact agreement against the parquet (verify_correctness).


## 2026-09-24 -- THE READ MATCHED TO THE QUESTION: integer dictionaries at the toll floor

**The law (Jackson, from the scatter census):** every read pays two prices -- a TOLL per trip
(the request, the lookup, the first byte back: a cold 64 KB read on the pod is 0.8 ms, nearly all
toll) and a CHARGE per byte (carrying: 3 MB cold is 14 ms, ~0.2 GB/s one stream; unpacking:
zstd 4-12 ms per 4 MB). They are equal at toll x speed ~ 100 KB here: the crossover size.
Point questions want read units down to the crossover and no smaller; whole questions want big
trips (neighbours on disk share one read); independent trips go at once so tolls overlap.
Every layer has the same shape with other numbers (cache line 64 B / ~100 ns, page 4 KB, SSD,
network storage). Family 1 of the census was the unit too BIG for its question; family 2 (codes
through the memory map, fault by fault) is the trips too SMALL and in a line -- one law, the two
sides of the crossover.

**The scatter census (bench/scatter_census.py):** true cold, every point-read method wrapped
(time, rows, major faults). 6.2 s of the 28.0 s board is point reads. Family 1: answer values
plucked one fetch at a time from the chunked integer dictionaries (UserID, WatchID, ClientIP,
URLHash, RefererHash): each pluck popped a whole 524,288-value chunk -- ~3 MB read (14 ms) +
4 MB inflated (4-12 ms) + a sum -- for 8 bytes. bench/pluck_probe.py: the board's mode-2 plucks
cost 1,466 ms serial, 482 ms batched.

**The change:**
- wdb_encode: I2CH 524288 -> 8192 values per chunk (~45 KB: the crossover). The reader needs no
  format change -- the chunk size is in each column's header.
- bench/i2_convert.py: cbdb_fc3 -> cbdb_i2, only the 11 chunked integer dictionaries rewritten
  (at the encoder's own level 9 -- level 19 is LARGER on these deltas), everything else byte for
  byte, every dictionary verified value for value. Size 8,811,172,032 -> 8,815,560,890 (+4.4 MB,
  +0.05%).
- wdb_engine._i2_pop: chunks that are neighbours on disk form a RUN -- one pread (runs sized so a
  wide read keeps 16 streams busy, at most 8 MB), frames inflated in the run's own thread, one
  reshaped cumsum per run written straight into the output (no concatenate). _dict_ints_at groups
  indices by chunk with one argsort (the old mask per chunk was chunks x indices).
- Two tolls found on the way, both by measurement: a pool task per chunk (12,207 futures: WatchID
  full read 680 -> 1450 ms) -> one task per run; and zstd's decompressobj(read_across_frames)
  HOLDS the GIL (HID 652 MB: 914 ms on 1 thread, 1130 on 16) while per-frame decompress releases
  it (150 ms on 16) -> per frame inside the run.

**Measured:**
- Plucks (the same codes, cold): 1,466 ms serial -> 101 ms serial (42 ms batched).
- Full dictionary reads, cold, old reader + old db -> new reader + new db (medians): WatchID
  679 -> 523, UserID 157 -> 151, URLHash 141 -> 158, ClientIP 53 -> 62, HID 217 -> 312 (per-frame
  inflate of 9,945 small frames: CPU, not disk), EventTime ~4.
- Board, interleaved A B A B (A = commit 323a0b5 on cbdb_fc3, B = this on cbdb_i2): runs A 27.7,
  27.9 / B 28.6, 26.7 s. Best of two per query: cold 27.2 -> 26.3 s, hot 12.90 -> 12.90 s.
  Every plucking query fell: Q15 -139, Q16 -162, Q17 -108, Q19 -96, Q23 -237, Q31 -225
  (85 -> 52 MB read), Q32 -168 (41 -> 6 MB), Q40 -85, Q41 -74, Q08 -63, Q18 -52. Rises only on
  big scans that touch no integer dictionary, bytes read unchanged (Q22 +91, Q27 +84, Q39 +64,
  Q28 +53): the run noise.
- Suite 1741 passed (tests/test_i2_chunks.py: 4 new -- chunk sizes 1000/8192/524288, partial
  last chunk, edges, runs forced tiny and huge, point->full->point). verify_correctness on
  cbdb_i2: 42/43 correct. Q23 (SELECT * ... ORDER BY EventTime LIMIT 10) is flagged WRONG -- and
  equally on commit 323a0b5 with cbdb_fc3, so it predates this. Checked cell by cell: on both
  databases WaveDB's 10 rows are the right rows (the 10th EventTime has no tie) and every cell
  of every row equals the parquet row, integer dictionary values included. The verdict is the
  verifier's star normalization, not the engine: TO AUDIT (bench/_cbnorm.py on star rows).

cbdb_i2 is the new canonical database; cbdb_fc3 its predecessor.


## 2026-09-24 -- THE ZSTD AUTOPSY and THE GAP PACKER (measured; parked: no customer on the board)

**Jackson's hypothesis:** zstd stores the bit matrix as literals plus generative copy rules; some
bits must already be in place (no rule stores them for less than they cost), so some values may be
readable without running every rule -- run a rule to 12% and pluck.

**The autopsy (bench/zstd_autopsy.py; zstd 1.5.6's own decoder built with DEBUGLEVEL=6 in
/workspace/zaut, fed frames cut from cbdb_i2):** confirmed, sharper than expected.
- Hash/ID dictionaries (URLHash, WatchID, UserID): the literal section is RAW -- Huffman could not
  beat the bytes as themselves -- 45-52% of the output stored in place in the file. By byte of each
  8-byte gap (low first) URLHash is literal 89 100 100 95 14 6 6 7%: the random low half in place,
  the near-zero high half written by ~one copy rule per value (7,698 rules / 8,192 values, median
  length 4, chains to depth 40-56). Those rules cost 14,952 of the frame's 49,111 bytes (~1.9 B per
  value) to say "the top bytes are small". Whole values in place: 0.5-5% (UserID 2-33%).
- A value k% into a frame is ready after ~k% of the rules: the median pluck runs half the frame.
- Rules are the right tool where nothing is random: EventTime's 8,192 values are 2 rules (30 B),
  HID/ClientIP 3.6-19% literals, copies of short repeating gaps.
- URL/Referer code frames: Huffman literals (4 streams), 23-31% of values all-literal.

**The gap packer (bench/gap_pack.py), every value in place by construction:** per 8,192-value chunk
a base, a width (bits of the largest gap), the gaps at that width (gap j at bit j*w), a checkpoint
every K values. Exact on all 11 dictionaries. K=64 against zstd:
- size: WatchID 543.98 -> 500.88 MB, RefererHash 129.10 -> 119.49, URLHash 124.04 -> 114.86 (width
  43.3 bits, as the arithmetic predicted), UserID 103.25 -> 96.02, FUniqID 86.90 -> 81.29, HID
  87.66 -> 84.71, ClientIP 17.80 -> 16.20, RemoteIP 17.53 -> 15.96; the time columns grow
  (EventTime 0.03 -> 0.47, ClientEventTime 1.10 -> 1.85): zstd keeps them. Per column best:
  1,111 -> 1,031 MB (-81 MB, ~0.9% of the database).
- full decode (warm, CPU): WatchID 401 -> 31 ms, HID 248 -> 20.5, URLHash 86 -> 11.5.
- one value: ~74 us (inflate 64 KB + sum) -> 0.05 us (no checkpoint 4.2 us; K=16 0.02 us at +7% size).
- a cold pluck needs ~340 bytes (header, a checkpoint, <= 63 gaps): under one page.

**The board census of the reads it would serve (scatter_census with _dict_ints/_i2_pop):** no board
query reads a big integer dictionary whole. Full reads are only small dictionaries (<= 10 ms each;
Q23's SELECT * touches ~70 of them at 0-4 ms). The big ones are only plucked, and the plucks already
sit at the toll floor since this morning: ~170 ms of _i2_pop across the whole board. The packer's
speed has no ClickBench customer today; its board value is the size, -81 MB. PARKED, ready: the
prototype, the kernels and the numbers are here for the join boards or a size push.

**Where the cold time is (same census):** full CODE decodes (_raw_codes) -- UserID enc 19 2.1 s,
URL/Referer enc 3 1.2 s, SearchPhrase 0.84 s, ClientIP 0.64 s, ... -- and the enc-3 point frames
read fault by fault (family 2). The autopsy's lens goes there next.


## 2026-09-24 -- THE CODE AUTOPSY and THE KERNEL LOAD BILL (measured)

**The code autopsy (bench/code_autopsy.py):** the 14 code sections the cold board decodes whole,
chosen by the scatter census. Per column a full decode cold (file evicted, fresh Segment) against
warm (CPU only), bytes pulled, stored bits per row against the information the codes carry (H0).
- The big columns carry no size fat: URL stores 20.56 bits/row against H0 18.57 (log2 V 24.13),
  Referer 18.03 / 15.57, ClientIP 19.37 against H0 20.67 and UserID (enc 19) ~19.7 against 22.41
  -- block locality already beats the global entropy. Their cold time is a READ THEN a DECODE in
  series: URL 360 ms cold = ~196 ms pulling 257 MB (1.31 GB/s, warm_span) + 170 ms decoding;
  ClientIP 378 = ~186 + 192; Referer 386 = ~220 + 164. The decode starts only when the whole span
  has landed: the CPU waits for the storage, then the storage waits for the CPU.
- The low-entropy columns are CPU-bound, not read-bound: MobilePhone 235 ms of CPU to write 100M
  codes carrying 0.51 bits/row (87% runs), MobilePhoneModel 142 ms (0.35 bits, 90% runs),
  SearchPhrase 244 ms (3.25 bits, 77% runs), IsRefresh 52 ms (0.35 bits).
- Code frames under the zstd lens: Huffman literals, 11-31% of values all-literal, a value ready
  after ~half the frame's rules -- the same shape as the dictionaries.
- (UserID's 1,635 ms here was a bare process's first parallel call; on the board it is ~280 ms:
  247 MB in 198 ms + ~60 ms decode.)

**The kernel load census (bench/kernel_load_census.py):** a numba kernel's first call in a process
loads its cached machine code (Dispatcher._compile_for_args) before running; every cold query is a
fresh process. Timed as WALL (the union of intervals in which any thread was loading -- the raw sum
counts every waiting thread: Q22 4,460 ms summed, 130 ms wall). Board: 1.78 s of 26.5 s cold (7%)
is loading already-compiled kernels. No compiles -- every first call is a cache hit. ~8-14 ms per
kernel, fc3_hostruns 138 ms (Q28, 37 threads idle meanwhile), fc3_charlens 72 ms. Most loaded:
unpack_any (14 queries), e8_pos (16), _hist_par (11), e19_decode (9), unpack24_be (8).
Database.open already loads five kernels and starts the parallel runtime (a bare process pays
~375 ms for its first parallel call; import wdb_kernels ~965 ms) -- outside the timed run.


## 2026-09-24 -- THE READ AND THE DECODE AT ONCE: enc-3 full decodes pipelined

**The wait (code autopsy):** an enc-3 full decode pulled the whole code section first (warm_span,
parallel preads) and only then inflated it in 8 lanes -- URL ~196 ms of reading, then 170 ms of
decoding, in series: the CPU waited for the storage, then the storage idled while the CPU worked.

**The change:** wdb_engine._e3_pipelined -- 16 lanes, each owning a stretch of frames, walking it
in RUNS of neighbouring frames: one pread of up to 8 MB, then its frames inflated (zstd releases
the GIL frame by frame), then the next run. One lane's inflate overlaps another's read. warm_span
is skipped for these columns (they read their own frames). Every frame's decoded row count is
asserted. WDB_PIPE3=0 restores the old path (the A/B switch).

**Measured:**
- bench/pipe_ab.py, one full decode per fresh process, cold, alternating, three each (median):
  URL 349 -> 223 ms, ClientIP 373 -> 238, Referer 329 -> 215, CounterID 119 -> 94,
  SearchEngineID 86 -> 53, RegionID 169 -> 163. Codes identical, byte for byte, every run.
- Board, same code and database, WDB_PIPE3 toggled, interleaved A B A B: runs A 26.9, 26.6 /
  B 25.8, 25.3 s. Best of two per query: cold 25.8 -> 24.4 s (-1.4 s), hot 13.01 -> 12.73 s.
  Q27 -217, Q22 -204, Q28 -147, Q30 -133, Q35 -126, Q34 -105, Q33 -87, Q40 -78.
  Board rises Q09 +124 / Q10 +77 / Q12 +56 re-run alone, alternating, three each: Q09
  1273/1171/1287 -> 1156/1110/1120 (faster every run), Q10 and Q12 even -- run noise.
- Suite 1741 passed.

Sibling paths with the same read-then-decode shape, not yet pipelined: enc 18 (packed frames),
enc 19 (UserID: 247 MB read, then e19_decode), enc 8/9/5.


## 2026-09-24 -- THE KERNEL LOAD BILL, dissected; the preload ceiling (measured, not built)

- What a load is (cProfile, cold Q12): per kernel ~9 ms -- the cache index read (~2 ms), the
  cached data read (~2 ms), then unserializing: parsing the stored bitcode and linking the
  object code into the process (~4 ms). fc3_hostruns: 140 ms (a large kernel).
- Cache files kept warm (bench/kernel_load_census.py with KLC_KEEP_NB=1): board load wall
  1.78 -> 1.56 s. ~88% of the bill is CPU inside the process, not file reads. __pycache__ holds
  875 files, 27 MB (stale overloads from old source line numbers included).
- numba's ahead-of-time compiler (pycc) does not take parallel kernels -- most hot kernels are.
- THE PRELOAD CEILING (bench/preload_probe.py): record each query's first-call kernels and
  their argument types; then, cold and alternating, load that exact list on a background thread
  from the moment the query starts (numba's compiler lock is global: loads stay serial, but they
  overlap the query's storage waits). Perfect prediction: best of two 25.1 -> 24.1 s (-1.0 s,
  ~60% of the bill). Noisy per query (Q16 +111, Q15 -170 on single runs).
- A real design must predict without memorizing the benchmark: each read method declares the
  kernels it calls (code, not a trained list), and the background loader brings in every cached
  overload of those. Open question for Jackson before building.


## 2026-09-24 -- THE READ AND THE DECODE AT ONCE, enc 19 (UserID)

- First try, segments pulled one at a time while the previous decoded: SLOWER (256 -> 532 ms).
  Each 31 MB segment kept only ~4 reads in flight: 0.5 GB/s against 1.25 for the whole span.
  The law again from the other side -- the trips must stay many and big.
- _e19_pipelined: every 8 MB read of the row pointers and block dictionaries is issued at once
  (16 streams), in segment order; the kernel (unchanged: each segment's slice of the words, the
  offsets re-based and read-only so the one cached signature serves) decodes segment k as soon as
  its reads have landed. Already resident (mincore > 90%): the plain single pass. WDB_PIPE19=0 is
  the switch. Segment._resident_share(fb, fe) added (the name _resident was taken).
- UserID full decode, cold, one per fresh process after Database.open, alternating (median of
  three): 285 -> 250 ms; codes identical. (bench/pipe_ab.py now opens the database first: a bare
  process pays ~1.4 s starting numba's parallel runtime on its first call.)
- Board, WDB_PIPE19 toggled, interleaved A B A B: runs A 27.7, 26.4 / B 25.9, 25.6 s (a noisy pod
  today: A's two runs differ by 1.3 s). Best of two: cold 25.9 -> 24.8 s, hot 13.16 -> 12.89 s.
  The UserID queries: Q09 -266, Q16 -277, Q18 -173, Q13 -72, Q08 -62. Suite 1741 passed.


## 2026-09-24 -- THE SPARSE DRESS FROM ITS PLANES: enc 8/9 full decodes

- Where the CPU went (warm, best of three): SearchPhrase (enc 8) 245 ms full decode, but its
  planes (present positions + literal codes, 13.2% of rows) cost 40 ms and a plain numpy default
  fill + scatter 49 ms -- the 3-pass kernel's in-kernel uint32 fill was the bill. MobilePhone
  (enc 9) 232 ms: the old path unpacked the 100M-bit presence plane into a bool array, copied it,
  and index-filtered the present rows per tier, serially; planes 132 ms + fill 27. MobilePhoneModel
  141 -> planes 60 + fill 22.
- The change: enc 8 and 9 full decodes = e8_planes (shared with the counting consumers) + one
  default fill + one scatter, at the same output widths (enc 8 uint32, enc 9 by its bits: no new
  kernel signature). WDB_PLANES=0 is the switch.
- Cold, one decode per fresh process after Database.open, alternating (median of three), codes and
  dtypes identical: SearchPhrase 277 -> 161 ms, MobilePhone 270 -> 205, MobilePhoneModel 182 -> 128.
- Board, WDB_PLANES toggled, interleaved: runs A 24.8, 24.4 / B 24.1, 23.8 s. Best of two: cold
  24.0 -> 23.2 s, HOT 12.84 -> 12.11 s (hot re-decodes per query). Q11 cold -228 (hot 449 -> 234),
  Q14 -209 (hot 561 -> 368), Q12 -201 (hot 408 -> 211), Q25 -193 (hot 419 -> 214), Q10 -165,
  Q23 -92. Board rises re-run alone, alternating, three each: Q40 even (noise); Q22 medians
  2059 -> 2117 with overlapping runs (2089/2059/1980 vs 2062/2117/2171) -- within its noise band,
  watch it. Suite 1741 passed.
- Next in line (not built): enc 9's tier walk as one compiled loop -- MobilePhone's planes are
  132 ms of Python-level passes; a kernel edit (the whole cache invalidates), ~100 ms on Q10/Q11.


## 2026-09-24 -- THE CENSUS OF THE LOAD: Q01 and Q07 from stored counts

**Q01 the old-fashioned way** (cold 321 ms under the profiler): 28 ms choosing a read (21 of them
opening the load statistics cold for a stair check), 56 ms decoding AdvEngineID's 100M codes,
188 ms BUILDING a bit-slice index from scratch (5 planes x 100M rows, serial numpy), 26 ms using
it, 10 ms counting. The index stays in the process: that is why hot was 50 ms.

**Jackson's two ideas, one fact:** (1) keep the per-value count and answer N - count(0);
(2) store the big exception as presence -- then the count of exceptions is a stored number (the
sparse dress already carries it: e8n). AdvEngineID is 0 on 99,366,997 of 99,997,497 rows (99.37%).

**Why the existing reads did not answer:** dict_count (scalar COUNT with WHERE k op v) and the
group-count read (GROUP BY k COUNT(*)) were built for exactly these shapes, but read per-code counts
from sidecars; with the switch off THE VANILLA LAW forbids building them, so they declined in
0.2 ms and the bit-slice index built its own structure for 300 ms.

**The change:**
- wdb_blockstats.value_counts / vcnt_from_load: the load writes rows-per-code (<col>.vcnt) for
  every dictionary column (mode 0/1/2) of at most VCNT_MAX = 65,536 codes into the load
  statistics -- DATA of the load (the file the encoder writes as its last step), a column
  statistic of the same species as the differentiator exception lists. Asserted: the counts cover
  exactly N rows.
- wdb_gbcount._census: the counting reads use the load's counts first, while they are the truth
  (a plain loaded segment, no overrides on the column, no deleted rows; merged multi-segment
  views decline). The scalar count's vanilla gate and the top-k bar yield to the census; the
  unbounded GROUP BY with ORDER BY is served exactly from it. WDB_CENSUS=0 is the switch.
- A BUG FOUND AND FIXED: GROUP BY k WHERE k <> v with no ORDER BY (small dictionaries) returned the
  excluded key as a count-1 group -- the heavy list dropped it, the singleton law put it back.
  Proven on commit 323a0b5: WaveDB [(0, 1), (2, 288), ...] against DuckDB [(2, 288), ...].
- bench/stats_vcnt.py adds the counts to an existing database: cbdb_i2, 87 columns in 16.2 s
  standalone (an upper bound on the added load time: on a fresh load most of these columns are
  already decoded for the block stats); stats file 9.56 -> 12.08 MB.
- tests/test_census.py: 4 tests -- counts written; =, <>, IN, NOT IN, >, BETWEEN on int and
  string columns equal DuckDB and are served by the census; group counts equal DuckDB; the
  excluded key stays out, with and without the census.

**Measured:** Q01 cold 309-320 -> 24 ms (hot 34-55 -> 6), Q07 cold 207-244 -> 25-28 ms (hot
79-91 -> 4-7), 0 bytes read (three alternating runs each; Umbra: 32 / 4 on both). Board, WDB_CENSUS
toggled, interleaved: best of two cold 23.5 -> 22.9 s, hot 12.03 -> 12.00 s; only Q01/Q07 move by
design, the rest within noise. Suite 1745 passed. verify_correctness 42/43 (Q23: the known
checker normalization).

**ClickBench score (one run, against the fair trial's referees):** COLD WaveDB 1.63, Umbra 1.67,
ClickHouse 2.17, DuckDB 3.32 -- WaveDB leads cold on score and total (23.6 s vs 24.4). HOT WaveDB
2.84 (second), Umbra 1.30.


## 2026-09-24 -- THE BIT-SLICE INDEX, AUDITED (Q01's old road)

- THE LAW HOLDS: the index Q01 used to build is cleared at query end (drop_derived clears every
  underscore container on the Segment; seg._bsi had 0 entries after each run) and the residue audit
  is empty. So every query that takes this road rebuilds its index, cold AND hot.
- Why Q01's hot looked fast (50 ms against 309 cold): in the same process, run 1 was answered by
  bsi_filter (282 ms, the index build), run 2 by fused_agg (55 ms) -- the router's in-process memory
  steers later runs away -- and run 3 from outside the read order (54 ms). Not a kept index.
- bench/route_census.py: which read answers each board query, three runs per fresh process. After
  the census of the load, NO board query reaches bsi_filter on any run (43 x 3). Cold-run answers:
  none 11 (outside the read order), dict_count 5 (Q01, Q07, Q12, Q15, Q33), blockstats 4,
  group_distinct 3, wherescan 3, fused_agg 2, pairfold 2, tripletop 2, pairtop 2, affinegroup 2,
  funnel 2, group_mix, pairdistinct, sampletop, regexgroup, affinesum 1 each.
- Its remaining customers: TPC-H-shaped filtered SUMs (tests/test_path_coverage.py ratchets two:
  bsi_discount_btw, bsi_q6_multi; tests/test_bsi_exec.py asserts the routing). And a hazard at 100M
  rows: any new shape that reaches it pays a full index build per query (Q01 paid 188 ms).
  Removal vs a size gate: Jackson's call, with TPC-H Q6 measured both ways first.


## 2026-09-24 -- COUNTERID'S ZSTD, AUTOPSIED; THE LINK CEILING (Q40 by hand)

- Jackson's hunch (one generative rule places 62): DENIED. bench/counter_autopsy.py on the traced
  decoder: CounterID is enc 3, 2-byte codes, 65,536 rows a frame; 62 = code 6 of 6,506, 738,172
  rows. The column changes value every ~4.4 rows; 62's rows are 93-96% pattern copies (multi-value
  stretches), 3-6% run copies, ~0.8% literals; ~800 runs of 62 a frame by ~900-975 sequences.
  But 62 lives in 193 CONTIGUOUS frames (726-918) and block min/max is already exact (386 of 3,052
  blocks flagged, 386 hold 62).
- July is the whole table (2013-07-02 .. 07-31, all 99,997,497 rows). Traffic -1/6 alone keeps
  55.3M. RefererHash = X alone keeps 98,213, of them 97,929 counter 62.
- JACKSON'S LINK (encode-time: one column's value points to another's, exceptions stored; the
  cheaper column to locate becomes the path to both). bench/link_ceiling.py measures the Q40 ceiling
  by hand, fresh process each, files evicted, 3 runs, median:
  A engine as is 357 ms cold / 167 hot; D RefererHash over the whole table 586 / 205 (the whole
  scan alone 353 ms); B RefererHash inside counter 62's blocks, CounterID checked 310 / 84;
  C region + link (skip CounterID, minus 284 stored exceptions, 2.5 KB) 282 / 79.
  All 284 exceptions lie OUTSIDE counter 62's blocks. Answers equal the engine's group counts
  (89,914 survivors, 41,194 groups); rows 101-110 differ only in tie order at equal counts.
  C's biggest remaining line: EventDate + URLHash point reads on 89,914 rows, ~112 ms cold.

- Q40's last lookups split (C, fresh process, cold x3): EventDate 5-13 ms (the staircase gives the
  same dates from position alone), URLHash 92-101 ms (hot ~20). URLHash is enc 3, 4-byte codes
  (V 20,714,865 = 25 bits), 65,536 rows a frame; the 89,914 survivors touch all 193 region frames
  (~172 KB compressed each on average): one survivor per ~140 rows, so every frame is read.
  bench/url_bytes.py: bytes to tell the 41,194 survivor URLs apart -- 2 bytes 30,592, 3 bytes
  41,141 (hash) / 41,178 (dict code), 4 bytes all. The region's population: 12,648,448 rows,
  2,311,287 distinct URLs (22 bits to number), entropy 14.2 bits/row; zstd today 21.5 bits/row
  (whole-column average).
