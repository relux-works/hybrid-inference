# RULER equivalence: Qwen f32 and bf16, r16e/r16f

CPU analysis on agents1, 2026-11-06, using the frozen tool and tests at accepted `e46e5623b5138c5212b23bb427510b9b14acb1c9` (Muse r7ap3 ACCEPT). Contract: [preregistration](RULER-EQUIVALENCE-PREREGISTRATION.md), `7556f46165b01216ff1028c381845c8e79c65955`. No statistical code or constants changed. Qwen now has all 390 paired examples at each of 8192, 32768, 65536 and 131072 tokens. Glimmer's committed JSON and generated table block are unchanged.

**Result:** the registered interval/Holm arithmetic passes at all four Qwen depths for offload-local and partition-local. The frozen accepted claim policy certifies 8k through its two sidecars, and 32k/64k through Newcombe; it **withholds a 128k equivalence claim** because the supplementary unseen-range interval is [-2.2429,+2.2429] pp (claim Holm p=0.0482095). The registered 128k bootstrap interval is [-0.6838,+0.8547] pp, registered Holm p=0.0004; reporting that alone would omit the accepted post-review guard. All f32 score-family depths are present, so their Holm p-values are final for these saved data and frozen rules. This does not establish equivalence simultaneously at all four depths under the claim policy. Text-endpoint equivalence is not established at any depth.

The original RC4 data cut-off was 2026-11-05 12:00Z. Both completed deep-depth archives arrived after that cut-off (64k finished 14:28:50Z on 11-05; 128k 05:56:57Z on 11-06). This is a post-cut-off completion analysis, not the result available at the original cut. No inference, provider request or download was performed.

## Frozen analysis rules and accepted review disposition

The registered primary endpoint remains the equally weighted mean over 390 paired task-score differences, with a 95% interval and inverted TOST p. Previously measured per-depth intervals and TOST p-values are unchanged; Holm is recomputed with all four Qwen depths. The interval rule frozen before the first analysis is Newcombe method 10 if every score in both compared arms is binary, otherwise one paired bootstrap over all 390 pairs. Each depth still requires 13 tasks x indices 0-29, shared prompts and seed 261102; incomplete, duplicate, mismatched or invalid rows refuse. CSV requires companion JSONL and manifest to verify identities and seeds.

[Newcombe (1998), section 5, method 10 and Table III](https://www.site.uottawa.ca/~nat/Courses/csi5388/Newcombe.1998.pdf) supplies the binary interval and published controls. Method 10 corrects positive paired phi by subtracting n/2 from its numerator, clipped at zero; negative phi is unchanged and a zero denominator gives phi=0. Marginal Wilson limits have no continuity correction.

### R1: degenerate primary and post-review sidecars

A point-mass bootstrap interval cannot certify equivalence. Qwen 8k has 377 fully correct, 11 zero and two 2/3 scores in each arm, with 0/390 nonzero paired differences. Its registered [0,0] interval, floored p=0.0001 and original Holm p=0.0004 remain visible, explicitly DEGENERATE and excluded as claim evidence. Both arm pairs instead get these post-review sidecars:

- Exact unseen discordance: with zero discordant pairs, the two-sided 95% Clopper-Pearson upper endpoint is `u = 1 - 0.025^(1/390) = 0.00941407`. Since raw differences lie in [-1,1], their population mean lies in [-u,+u], or [-0.9414,+0.9414] pp. Its inverted one-sided p is `0.98^390 = 0.0003785903`.
- Binarised (score == 1) Newcombe: cells (both correct, left only, right only, neither) = (377,0,0,13), interval [-1.0387,+1.0387] pp, TOST p=0.00238828. This is the fully-correct-rate estimand, not the registered mean raw score.

The Qwen 8k equivalence statement is supported by both sidecars with roughly 1 pp half-widths, not by [0,0]. The automatic rule requires both sidecar intervals inside the margin and uses their maximum p for a supplementary four-depth claim Holm calculation; pairs without sidecars use their registered p. At Qwen 8k the supplementary claim Holm p is 0.00955312 for both pairs in the complete family. This conservative supplementary decision is distinct from the registered primary Holm construction. JSON flags `equivalent_unadjusted` and `equivalent_holm` refer to this explicit `claim_basis`, with `claim_p` and `claim_holm_p`; `registered_interval_inside_margin` records only geometric containment, never a claim from a point mass.

The same sidecar rule applies to future constant-difference partial-credit sets, including bf16, and rejects the n=3 zero-difference claim. For an exactly represented constant nonzero difference c, the first pair fixes c and the remaining n-1 pairs bound unseen departures; at n=1 the bound is [-1,1]. With unseen mass bounded by u the raw-mean sidecar remains `[c-(1+c)*u, c+(1-c)*u]`. Binary-score sets retain their ordinary Newcombe primary.

### r16c / repeat of r7ap F1: dispersion and unseen-range guard

Exact float identity is no longer the degeneracy predicate. On the fixed [0,1] score scale, bootstrap differences with spread `max(D)-min(D) <= 32*ulp(1.0)` are marked degenerate. This tolerance is approximately 7.1054e-15 score units (7.1054e-13 pp), comfortably above the roughly 1e-17 subtraction jitter in the review witness and negligible relative to the 0.02 margin. It changes classification only: scores and the mean paired estimate are never rounded or replaced. Widening the conditioning range by this tolerance is conservative. The original exactly constant Qwen 8k sidecars remain byte-identical.

A tolerance alone would move the bypass to a slightly larger spread. Therefore every other raw-passing partial-credit bootstrap also requires the raw-score unseen-range sidecar and the existing score==1 sidecar before it can certify. For nonzero spread, the first two paired differences freeze an envelope `[lo,hi]`, expanded by 32 ulps and clipped to [-1,1]. Only the remaining n-2 pairs count toward a two-sided 95% Clopper-Pearson upper bound u on mass outside that fixed envelope. The score-mean bound is `[(1-u)*lo-u, (1-u)*hi+u]`; at zero held-out pairs, u=1 and the bound is [-1,1]. This uses held-out departures, not a zero count against a range selected on all observations. Its p is obtained by the same central-interval inversion; supplementary claim Holm remains unchanged in construction.

Both review witnesses now reject through `score_pair`, `analyze_family` and generated Markdown: mathematically constant +1.5 pp differences with subtraction jitter, and 389 differences of +1.5000 pp plus one of +1.5100 pp. The raw bootstrap intervals are retained, but cannot certify without the sidecars. The Qwen-8k-shaped positive and binary controls still pass; genuinely varied +/-1 pp differences also pass unadjusted equivalence with a nonzero unseen-range interval. This extra conservative guard may withhold a future bootstrap claim; it does not claim that every small unit fixture or every true equivalent distribution has sufficient power after Holm.

### r16c F2: timing-map refusals

`--run-times` validates the entire map before analysis: root, family, mode and depth entries must be objects, and start/end must be strings. Invalid shapes produce a specific `REFUSED:` message, exit 2 and no output directory, rather than an AttributeError traceback. Named CLI negatives cover a list root, non-dict family/mode/depth, and non-string start/end, alongside a valid-map positive control. The same validation protects direct `registration_info` calls. Existing date and timezone validation remains in place.

### R2: actual collection timing and registration scope

The registration timestamp is 2026-11-03 14:36Z. Explicit family/arm-variant/depth maps record run start and end; completion by registration means before, a start at or after registration means after, and overlap means straddling. Unknown dates remain unknown rather than inferred from depth. Qwen 8k ran 05:52-07:32Z and 32k ran 07:32-14:21Z, both before. Glimmer 8k ran 13:59-15:35Z including its pilot: collection straddled registration and is not confirmatory.

**Glimmer was not named in the registration. The same analysis rules are applied to Glimmer by extension**, including later after-registration runs; timing alone does not preregister that family. Qwen's later depths and bf16 arms are named by the contract. The supplied deep-depth start/end times are passed through `--run-times`, without changing constants. `--run-times` supplies witnessed future metadata without changing source; an absent future Glimmer32 timestamp stays unknown, and a recorded later run is labelled after, never before. Format example: `{"Glimmer":{"f32":{"32768":{"start":"2026-11-03T15:35:00Z","end":"2026-11-04T02:00:00Z"}}}}`; bf16 uses a separate `bf16` map. This example is a format fixture, not an observed future result.

### R3 / N1: incomplete families and Holm upper bounds

Pending p=1 is an upper bound, not an uncertain observed verdict. Replacing pending ones by future p-values can only lower or keep each observed adjusted p. Qwen's observed accepted per-depth decisions (8k from sidecars, 32k from registered intervals) therefore cannot reverse because pending depths arrive. That described the earlier incomplete Qwen cut; all four Qwen depths are present now, with the separate 128k sidecar limitation reported above. Glimmer remains incomplete. Glimmer 8k cannot become equivalent: its raw intervals already cross +2 pp; only its Holm p upper bound can decrease. A future raw-pass/Holm-fail row may become accepted as pending values arrive, and the tool distinguishes that case. Changes to data, frozen methods or metadata are outside this monotonicity statement.

Families are separate for each model and arm pair at alpha=0.025; this controls error per family, not a single family-wise level across pairs or models. Qwen's planned depths are 8192,32768,65536,131072; Glimmer's are 8192,32768,65536,122880. The four Qwen f32 archives contain no bf16 observations. The separate bf16 archives have `results.offload_bf16`, which the frozen loader refuses directly. The r16f adapter below now admits their unchanged baseline pairing for the two existing-depth analyses.

### N2-N4: frozen implementation choices and remaining limits

The registration specifies the score estimand, 95% intervals, Newcombe method 10 / partial-credit bootstrap with 10000 resamples and seed 261102, +/-2 pp margin, TOST alpha=0.025 per side, and four-depth Holm. It does **not** specify the following fill-ins, now frozen for future depths and bf16:

| Choice absent from registration | Frozen implementation | When fixed |
|---|---|---|
| Mixed binary/partial-credit sets | One bootstrap over all pairs if either arm contains any partial score; otherwise method 10 | r16 before calculation |
| Bootstrap construction | Percentile, unstratified, sorted (task,index); `random.Random(261102).choices` restarted per pair; type-7 quantiles at `(B-1)*q` | r16 before calculation |
| P construction and finite resolution | Invert central intervals at coverage `1-2*alpha`; max one-sided p, no doubling; bracket 1e-15; conservative bootstrap floor 1/10000; p=1 if no alpha<=0.5 certifies containment | r16 implementation |
| Holm families | Separate per model and arm pair, without cross-family error control | r16 before calculation |
| Pending entry | p=1 upper bounds, with the stability interpretation above | r16 entry; r16b clarifies wording |
| Degenerate-case sidecars | Keep registered results; add exact raw-score bound and score==1 Newcombe; both required, max sidecar p enters supplementary claim Holm | r16b; r16c replaces float equality and extends the raw-pass range guard after re-review |
| Near-margin diagnostics | If a nondegenerate bootstrap endpoint is within 0.1 pp of either margin, report five seeds 261100-261104 without changing the registered seed or decision | r16b, post-review |

For N2, an inverted percentile-bootstrap p is approximate, a construction for Holm rather than an exact registered test. P-values just above 1/10000 are still coarse even without a resolution flag. The registered criterion remains strict interval containment; the R1 sidecar claim is explicitly supplemental. Current Glimmer endpoints are over 0.6 pp from +2. No Qwen bootstrap endpoint triggers the near-margin diagnostic in the completed four-depth analysis; a named CPU test exercises its five-seed diagnostics on a near-margin fixture. No cross-Python-version stability is claimed; JSON records the actual Python version.

For N4, the public CLI now admits only Qwen or Glimmer and verifies that label against the recorded model SHA-256. The transitive bench/proxy import chain remains: the unchanged secondary analyser imports it for shared helpers. Removing it would require changing accepted secondary infrastructure or duplicating that analyser; this narrow statistical/reporting rework deliberately preserves its source hash and tests. These imports do not start services or make network calls in this workflow. Import isolation remains a non-blocking follow-up, not a claim that the chain was removed.

## Verified source archives and identity

Every compressed hash was checked before reading its inputs. Every regular member other than SHA256SUMS is covered by SHA256SUMS and verified: six archives, eight members each, 48 internal hashes. All 2340 rows, including the 780 bf16 rows, pass their own row_sha256 checks. Four f32 companion CSVs match their JSONL rows. The small input copies are removed after validation; original inputs are unchanged.

| Input | Original archive | SHA-256 |
|---|---|---|
| qwen-8192 | `/home/agent/lanes/hybrid/.temp/ruler-results-261103/depth-8192-complete.tar.gz` | `2be3eae64b783867258a1f22e1c7bbdb4188f257c378f17c07554a8868248aa4` |
| qwen-32768 | `/home/agent/lanes/hybrid/.temp/ruler-results-261103/depth-32768-complete.tar.gz` | `a6fb1e7e6ac0fb7d25ed3ed24ef1f47cdf4f6495b54a4866d6486870bfeef176` |
| qwen-65536 | `/home/agent/lanes/hybrid/.temp/ruler-results-261106/qwen-inputs/depth-65536-complete.tar.gz` | `012c9308fcd7cc4343deb023ced9f026d9fe26cc6a3e4dd8fdc63ebf75c13fd8` |
| qwen-131072 | `/home/agent/lanes/hybrid/.temp/ruler-results-261106/qwen-inputs/depth-131072-complete.tar.gz` | `36da2f0328ad2145dcab3ed47cc215744708d66b90602b42e0495666baecc94e` |
| bf16-8192 | `/home/agent/lanes/hybrid/.temp/ruler-results-261103/bf16-8d876c71/bf16-8192-complete.tar.gz` | `c0348572c2c1baf50dcfcdd0a73397594ba0d12e45ba157b0ae63980f25799b6` |
| bf16-32768 | `/home/agent/lanes/hybrid/.temp/ruler-results-261103/bf16-8d876c71/bf16-32768-complete.tar.gz` | `5efae77aa8b9902bd31b9b2a1ba336a0d8a090353898fd31fee8c980e73e1fbe` |

Historical Glimmer 8k source: `ruler-results-261103/glimmer-715c81a4/results-8192.tar.gz`, SHA-256 `2ff122d17f892dcb6f495306e06d2fb1671e0b145523df18605c58a3b6bfe8f1`; its already accepted snapshot is preserved, not recalculated here.

All four Qwen manifests agree on model SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`, input lock `f42a48cc9d2ca4a0e6ab0bb986fb7b87990dee3c61cc821da53d4147ed80594b`, seed 261102, source/native commit `18d3f114e46c4f801c4a09e2c6cf8ba2303ae42d`, all 50 Python source hashes, packages, CUDA backend, flags, layout and template. Every source hash was checked against that commit. The original insertion-order input-lock serialization reconstructs f42a48cc; the archive's key-sorted exported lock has different bytes, with identical values in all six archives.

| Compared manifests | Differences besides depth | Acceptance |
|---|---|---|
| 8k vs 32k | None | Accepted by the frozen analysis |
| 8k vs 64k | 64 binary-hash leaves: client 22, node 20, partition 22; fresh build at the same commit | Accepted; the statistical loader does not compare native binary hashes across depths |
| 8k vs 128k | 21 binary-hash leaves: client 6, node 9, partition 6; node B build | Accepted; same limitation |

For example, libggml-cuda.so hashes are `379a36ea...f9d8` at 8k/32k, `2c31632c...2d77` at 64k, and `cdc23883...b311` at 128k. The complete differences and build identities are persisted in the task's manifest-diff and build-identities JSON files. These are real build differences, not a claim of byte-identical binaries across depths. The statistical tool checks paired rows, seed and family/model identity; the additional data-lock/source checks were performed in this analysis. It does not enforce whole-manifest equality across depths. The evaluator's resume gate separately requires equality of the whole manifest at a given depth.

FACTS records 64k as node A (Vast 58864154, RTX PRO 6000 Server): 369 rows before the owner-ordered stop and 21 after same-node/same-build resume. At 128k, rows 1-226 ran on node B (58969419, RTX PRO 6000 WS) and rows 227-390 on A with B's exact binaries copied to the same paths. Each example's three arms ran on one host. Host identity is not a manifest field, so an unchanged whole manifest can accompany the host change. All 225 rows saved off-node before the stop were rechecked byte-for-byte against the final rows' original pretty-JSON serialization, and the saved manifest bytes match the final manifest. This verifies the saved prefix; the 226-row handover and binary-copy execution are attributed to the stage/FACTS custody record, not newly observed here.

## Timing and bf16 disclosure

64k is labelled **collection after registration**. 128k is labelled **collection straddled registration; not confirmatory**, retaining its main-stage start at 2026-11-03T14:29:06Z against the frozen 14:36:00Z constant (the registration commit is timestamped 14:36:38Z). The separate pilot began at 14:16:37Z. The margin was fixed after Qwen 8k/32k had already been observed.

The main-stage pre-registration window is 414 s. The first archived example, niah_single_1 index 0, has serial arm-call times 127.163560637 + 98.564235609 + 133.263844227 = 358.991640473 s. The second contributes 359.549984224 s: cumulative 718.541624697 s. The evaluator runs these calls and examples sequentially and writes a row only after all three arms. Even excluding generation, release, disk and other overhead, the second row could not finish before registration. **At most one of the 390 archived rows could have been written before 14:36Z.** No row/event/log carries a row-write wall-clock timestamp; the exact count is not established (zero or one). This bound does not count the separate pilot as a main-depth example and does not justify relabelling 128k as after registration. Log hashes and the derivation are in the task report.

The registration names bf16, but 64k and 128k were never run. At r16e the frozen tool refused the fourth-arm name offload_bf16. The separate r16f adapter below resolves input admission while preserving the statistical source. The report now includes both registered bf16 families at the two measured depths and explicit p=1 entries for the two absent depths.

## Paper tables

The generated Markdown below is verbatim from the frozen formatter apart from heading levels. Its conditional pending-depth text and sidecar column heading say 'upper bound'; Qwen's JSON has pending_depths=[], family_incomplete=false and holm_p_is_upper_bound=false. Its Qwen p-values are now complete-family values. The frozen decision-stability wording about pending arrivals is inert here. The additional claim table disambiguates registered Holm from the supplementary claim Holm.

<!-- BEGIN GENERATED TABLES -->
### Qwen: paired RULER equivalence

Differences and intervals are percentage points. Margin: +/-2 pp; alpha per side: 0.025.
Pending depths enter as p=1: reported Holm p-values are upper bounds. Accepted observed decisions cannot reverse when they arrive; raw failures cannot become equivalent.
A whole four-depth family claim remains incomplete while depths are missing.
Qwen was named in the registration.

| Depth | Pair | Difference | 95% interval | Method | TOST p | Holm p | Equivalent (raw / Holm) | Registration |
|---:|---|---:|---|---|---:|---:|---|---|
| 8192 | local_partition-local | +0.0000 | [+0.0000, +0.0000] | paired_percentile_bootstrap (DEGENERATE; not claim evidence) | 0.0001 | 0.0004 | not certified; see sidecars | observed before registration; named in registration |
| 8192 | offload-local | +0.0000 | [+0.0000, +0.0000] | paired_percentile_bootstrap (DEGENERATE; not claim evidence) | 0.0001 | 0.0004 | not certified; see sidecars | observed before registration; named in registration |
| 32768 | local_partition-local | -0.2564 | [-1.4704, +0.8517] | newcombe10 | 0.00725632 | 0.00725632 | True / True; equivalent; cannot reverse when pending depths arrive | observed before registration; named in registration |
| 32768 | offload-local | +0.0000 | [-1.0519, +1.0519] | newcombe10 | 0.00255609 | 0.00499888 | True / True; equivalent; cannot reverse when pending depths arrive | observed before registration; named in registration |
| 65536 | local_partition-local | +0.0000 | [-1.0475, +1.0475] | newcombe10 | 0.00249944 | 0.00499888 | True / True; equivalent; cannot reverse when pending depths arrive | collection after registration; named in registration |
| 65536 | offload-local | +0.0000 | [-1.0475, +1.0475] | newcombe10 | 0.00249944 | 0.00499888 | True / True; equivalent; cannot reverse when pending depths arrive | collection after registration; named in registration |
| 131072 | local_partition-local | +0.0855 | [-0.6838, +0.8547] | paired_percentile_bootstrap (unseen-range guard) | 0.0001 | 0.0004 | not certified; see sidecars | collection straddled registration; not confirmatory; named in registration |
| 131072 | offload-local | +0.0855 | [-0.6838, +0.8547] | paired_percentile_bootstrap (unseen-range guard) | 0.0001 | 0.0004 | not certified; see sidecars | collection straddled registration; not confirmatory; named in registration |

## Sidecars for degenerate sets and bootstrap passes (post-review supplement)

Claims require both sidecars. Supplementary claim Holm uses max(sidecar p) where sidecars apply; registered primary numbers above are unchanged.

| Depth | Pair | Sidecar / estimand | 95% interval (pp) | TOST p | Combined claim Holm p (upper bound) | Equivalent from both sidecars |
|---:|---|---|---|---:|---:|---|
| 8192 | local_partition-local | exact unseen discordance sidecar / mean raw score difference | [-0.9414, +0.9414] | 0.00037859033 | 0.0095531169 | True; equivalent; cannot reverse when pending depths arrive |
| 8192 | local_partition-local | binarised (score == 1) Newcombe sidecar / fully-correct rate difference | [-1.0387, +1.0387] | 0.0023882792 | 0.0095531169 | True; equivalent; cannot reverse when pending depths arrive |
| 8192 | offload-local | exact unseen discordance sidecar / mean raw score difference | [-0.9414, +0.9414] | 0.00037859033 | 0.0095531169 | True; equivalent; cannot reverse when pending depths arrive |
| 8192 | offload-local | binarised (score == 1) Newcombe sidecar / fully-correct rate difference | [-1.0387, +1.0387] | 0.0023882792 | 0.0095531169 | True; equivalent; cannot reverse when pending depths arrive |
| 131072 | local_partition-local | exact unseen range departures sidecar / mean raw score difference | [-2.2429, +2.2429] | 0.048209478 | 0.048209478 | False; equivalence not established; raw interval/sidecars fail, cannot reverse when pending depths arrive |
| 131072 | local_partition-local | binarised (score == 1) Newcombe sidecar / fully-correct rate difference | [-1.0493, +1.6307] | 0.01056404 | 0.048209478 | False; equivalence not established; raw interval/sidecars fail, cannot reverse when pending depths arrive |
| 131072 | offload-local | exact unseen range departures sidecar / mean raw score difference | [-2.2429, +2.2429] | 0.048209478 | 0.048209478 | False; equivalence not established; raw interval/sidecars fail, cannot reverse when pending depths arrive |
| 131072 | offload-local | binarised (score == 1) Newcombe sidecar / fully-correct rate difference | [-1.0493, +1.6307] | 0.01056404 | 0.048209478 | False; equivalence not established; raw interval/sidecars fail, cannot reverse when pending depths arrive |

## Collection timing against 2026-11-03 14:36Z

| Depth / pair | Start (UTC) | End (UTC) | Timing |
|---|---|---|---|
| 8192 / local_partition-local | 2026-11-03T05:52:00Z | 2026-11-03T07:32:00Z | before |
| 8192 / offload-local | 2026-11-03T05:52:00Z | 2026-11-03T07:32:00Z | before |
| 32768 / local_partition-local | 2026-11-03T07:32:00Z | 2026-11-03T14:21:00Z | before |
| 32768 / offload-local | 2026-11-03T07:32:00Z | 2026-11-03T14:21:00Z | before |
| 65536 / local_partition-local | 2026-11-03T19:20:51Z | 2026-11-05T14:28:50Z | after |
| 65536 / offload-local | 2026-11-03T19:20:51Z | 2026-11-05T14:28:50Z | after |
| 131072 / local_partition-local | 2026-11-03T14:29:06Z | 2026-11-06T05:56:57Z | straddling |
| 131072 / offload-local | 2026-11-03T14:29:06Z | 2026-11-06T05:56:57Z | straddling |

#### Secondary: answer text (unchanged ruler_analysis.py)

| Depth | Offload changed / n | Partition changed / n | McNemar p | Excess 95% (pp) | Text equivalent +/-2 pp |
|---:|---:|---:|---:|---|---|
| 8192 | 34/390 | 32/390 | 0.726562 | [-2.1275, +3.1189] | False |
| 32768 | 33/390 | 29/390 | 0.617719 | [-4.0224, +6.0473] | False |
| 65536 | 28/390 | 32/390 | 0.571588 | [-5.5126, +3.4906] | False |
| 131072 | 32/390 | 27/390 | 0.38331 | [-2.6891, +5.2105] | False |

### Glimmer: paired RULER equivalence

Differences and intervals are percentage points. Margin: +/-2 pp; alpha per side: 0.025.
Pending depths enter as p=1: reported Holm p-values are upper bounds. Accepted observed decisions cannot reverse when they arrive; raw failures cannot become equivalent.
A whole four-depth family claim remains incomplete while depths are missing.
Glimmer was not named in the registration; these rules are applied to it by extension.

| Depth | Pair | Difference | 95% interval | Method | TOST p | Holm p | Equivalent (raw / Holm) | Registration |
|---:|---|---:|---|---|---:|---:|---|---|
| 8192 | local_partition-local | +1.1667 | [-0.1923, +2.6410] | paired_percentile_bootstrap | 0.123712 | 0.494849 | False / False; equivalence not established; raw interval/sidecars fail, cannot reverse when pending depths arrive | collection straddled registration; not confirmatory; not named in registration; rules applied by extension |
| 8192 | offload-local | +1.2179 | [-0.1410, +2.6923] | paired_percentile_bootstrap | 0.141114 | 0.564456 | False / False; equivalence not established; raw interval/sidecars fail, cannot reverse when pending depths arrive | collection straddled registration; not confirmatory; not named in registration; rules applied by extension |

## Collection timing against 2026-11-03 14:36Z

| Depth / pair | Start (UTC) | End (UTC) | Timing |
|---|---|---|---|
| 8192 / local_partition-local | 2026-11-03T13:59:00Z | 2026-11-03T15:35:00Z | straddling |
| 8192 / offload-local | 2026-11-03T13:59:00Z | 2026-11-03T15:35:00Z | straddling |

#### Secondary: answer text (unchanged ruler_analysis.py)

| Depth | Offload changed / n | Partition changed / n | McNemar p | Excess 95% (pp) | Text equivalent +/-2 pp |
|---:|---:|---:|---:|---|---|
| 8192 | 130/390 | 125/390 | 0.38331 | [-2.6891, +5.2105] | False |
<!-- END GENERATED TABLES -->

## Qwen claim decisions and descriptive cross-check

Registered Holm uses the registered TOST p. Claim Holm substitutes the maximum sidecar p wherever the frozen guard requires sidecars; it is a distinct supplementary analysis. All p-values below use their complete four-depth family within each arm pair, alpha=0.025.

<!-- BEGIN QWEN DETAIL TABLES -->
| Depth | Pair | Registered Holm p | Claim Holm p | Claim under the accepted policy |
|---:|---|---:|---:|---|
| 8192 | local_partition-local | 0.0004 | 0.00955311695 | equivalent via sidecars; degenerate primary unusable |
| 8192 | offload-local | 0.0004 | 0.00955311695 | equivalent via sidecars; degenerate primary unusable |
| 32768 | local_partition-local | 0.0072563175 | 0.014512635 | equivalent |
| 32768 | offload-local | 0.0049988767 | 0.00955311695 | equivalent |
| 65536 | local_partition-local | 0.0049988767 | 0.00955311695 | equivalent |
| 65536 | offload-local | 0.0049988767 | 0.00955311695 | equivalent |
| 131072 | local_partition-local | 0.0004 | 0.0482094776 | not established; unseen-range sidecar crosses margin |
| 131072 | offload-local | 0.0004 | 0.0482094776 | not established; unseen-range sidecar crosses margin |

Per-arm scores and secondary text statistics reproduce the accepted evaluator and the FACTS blocks at their printed precision. Text comparisons use outer-whitespace equality, exact two-sided unadjusted McNemar, Wilson 95% rate intervals, and the accepted conservative excess interval. A nonsignificant McNemar p is not equivalence.

| Depth | Local / offload / partition score (%) | Fully correct local / offload / partition | Accuracy McNemar p offload / partition |
|---:|---|---|---|
| 8192 | 97.01 / 97.01 / 97.01 | 377 / 377 / 377 | 1 / 1 |
| 32768 | 97.44 / 97.44 / 97.18 | 380 / 380 / 379 | 1 / 1 |
| 65536 | 97.18 / 97.18 / 97.18 | 379 / 379 / 379 | 1 / 1 |
| 131072 | 95.81 / 95.90 / 95.90 | 373 / 374 / 374 | 1 / 1 |

| Depth | Offload changed/n, % [Wilson 95%] | Partition changed/n, % [Wilson 95%] | Offload-only / partition-only | Text McNemar p | Excess pp [95%] | Text equivalent +/-2 pp |
|---:|---|---|---|---:|---|---|
| 8192 | 34/390, 8.72 [6.31, 11.94] | 32/390, 8.21 [5.87, 11.35] | 5 / 3 | 0.726562 | +0.5128 [-2.1275, +3.1189] | False |
| 32768 | 33/390, 8.46 [6.09, 11.65] | 29/390, 7.44 [5.23, 10.48] | 20 / 16 | 0.617719 | +1.0256 [-4.0224, +6.0473] | False |
| 65536 | 28/390, 7.18 [5.01, 10.18] | 32/390, 8.21 [5.87, 11.35] | 12 / 16 | 0.571588 | -1.0256 [-5.5126, +3.4906] | False |
| 131072 | 32/390, 8.21 [5.87, 11.35] | 27/390, 6.92 [4.80, 9.89] | 13 / 8 | 0.38331 | +1.2821 [-2.6891, +5.2105] | False |
<!-- END QWEN DETAIL TABLES -->

At 128k, the three nonzero paired-score differences, identical for offload-local and partition-local, are fwe index 12 (+1/3; local 2/3 versus the other arm 1), qa_2 index 13 (+1; local 0 versus the other arm 1), and qa_2 index 14 (-1; local 1 versus the other arm 0); the first two sorted training pairs are cwe indices 0 and 1, both with zero difference. The frozen range guard counts three departures among the remaining 388 pairs, giving the [-2.2429,+2.2429] pp sidecar. The binary score==1 sidecar passes alone, but both sidecars are required. None of the bootstrap intervals triggers the near-margin sensitivity diagnostic, so no extra-seed result is generated. Finite bootstrap p resolution remains 1/10000.

All 56 Qwen ALL/per-task secondary tables match the archived analyses exactly after excluding their single-depth Bonferroni adjustment field; the frozen secondary analyser now emits its four-depth Holm field. All 52 per-task score triplets and text counts are preserved in qwen.json. The 128k cross-check is 95.81/95.90/95.90%, changed answers 32/390 and 27/390, discordant counts 13 and 8, text McNemar p=0.38331, matching the supplied archive and FACTS. Historical Glimmer descriptive numbers and its incomplete family remain unchanged. These data establish bounded RULER task-score claims at 8k/32k/64k under the accepted policy; they do not establish identical answer text, general quality equivalence outside RULER, a confirmatory 128k claim, or a complete bf16 family.

## bf16 admission and the two measured depths (r16f)

The separate [input adapter](ruler_bf16_adapter.py) admits the existing reviewed state-bf16 experiment at 8d876c71. It performs no statistics. All four compressed input hashes and all 32 SHA256SUMS members pass before manifest/row use. Every original row_sha256 is mandatory; all 1560 source rows and 780 derived rows validate. Source archives are never modified. Candidate adapter source is self-reviewed and CPU-tested, ready for review; this does not claim a new independent acceptance verdict.

Admission pins model 31629f53..., data lock f42a48cc..., seed 261102, baseline source 18d3f114 and declared bf16 implementation 8d876c71b798651a8c5df768dc374468ced77544. The canonical implementation digest a0a79998860655482a11add7d8fea80dd61fbd2f5478fabaa5e56a5e5349250e covers all 55 recorded source hashes and the commits/binary hashes of three services. Its values agree between the two owner-hashed archives; all 55 source hashes also match the signed 8d876c71 object in the local mirror. This verifies declared build identity, not a new execution of destroyed hardware. The adapter checks the exact baseline manifest and implementation links, every baseline_row_sha256, the same complete task/index set, unchanged three baseline arm results and input metadata, and bf16 trace identity. Unknown or additional fourth arms refuse. Only results.offload_bf16 becomes results.bf16; all other source fields stay intact. Each derived row records original_row_sha256 beside its new row_sha256. A receipt retains all 780 original/derived/baseline row-hash links, archive/member hashes and the adapter source hash.

Committed outputs: [bf16 frozen-tool snapshot](equivalence-results/qwen-bf16.json) and [input/timing provenance](equivalence-results/bf16-input-provenance.json). The raw snapshot also contains the frozen baseline comparisons and f32 secondary tables that the unchanged CLI always emits; these do not replace qwen.json or update the accepted four-depth f32 results. Only bf16-local and bf16-offload are reported below. The statistical tool, its 48 existing tests, qwen.json, glimmer.json and their generated table blocks remain byte-identical to r16e.

### bf16 timing: bounds, not invented timestamps

The only stage window is 2026-11-03T14:23:39Z to 18:41:33Z, with 8k first. Stage return code 1 came from the post-arm service restore; both arms were complete. The registration constant is unchanged at 14:36:00Z. No per-depth wall-clock timestamp survives. The pinned stage executes build, service restart and preflight after execute begins, before either depth; treating execute start as a measured 8k start would be unjustified (stage_ruler_bf16.py at 8d876c71, lines 165-206). The single-arm evaluator runs calls serially (ruler_eval.py at that commit, lines 297-315).

| Depth | Sum of its 390 serial bf16 call durations | What follows | Classification |
|---:|---:|---|---|
| 8192 | 3027.088313445449 s | Exceeds the 741 s execute-to-registration gap, so completion is after registration; first-call time unknown | unknown; may be after or straddling |
| 32768 | 11315.0215302594 s | Follows the complete 8k arm, hence begins no earlier than execute start +3027.088313445449 s | after registration |

A conservative enclosing window for 32k is [2026-11-03T15:14:06Z, 2026-11-03T18:41:33Z]: start is the lower bound rounded down, end is the whole stage's upper bound. These are not measured per-depth endpoints. That enclosing window is passed through --run-times only for bf16 32768; bf16 8192 has no map entry, so the frozen tool correctly emits unknown and not confirmatory. The saved timing provenance marks these semantics explicitly. No claim is made that 8k was observed before registration.

### bf16 score claims and missing-depth policy

Both comparisons have 390 matched examples at each measured depth. Every bf16 score equals both its local and f32-offload score. At 8k the registered percentile bootstrap is degenerate [0,0] pp; its p=0.0001 floor and registered Holm p=0.0004 cannot certify a claim. Both post-review sidecars instead support equivalence, each with approximately 1 pp half-width. At 32k, all scores are binary and Newcombe method 10 applies. No near-margin diagnostic triggers. All four observed pair/depth claims pass the accepted policy, with supplementary claim Holm p=0.00955311695 for each. The 32k registered TOST p=0.00255608516 and registered Holm p=0.00766825548 are distinct from that supplementary claim p.

64k and 128k bf16 were never measured. Both enter each of the two four-depth families as absent p=1, not observed failures or planned completed measurements. Every displayed Holm p is an upper bound while those entries are absent. Replacing these p=1 values by future data cannot reverse these accepted observed-depth decisions under the frozen rules. Neither whole four-depth bf16 family is complete, and no equivalence or general quality claim is made for those missing depths. The 8k timing classification prevents a confirmatory temporal claim for that depth.

<!-- BEGIN BF16 TABLES -->
### Qwen bf16: paired RULER score equivalence

Differences and intervals are percentage points. Margin: +/-2 pp; alpha per side: 0.025.
Pending depths enter as p=1: reported Holm p-values are upper bounds. Accepted observed decisions cannot reverse when they arrive; raw failures cannot become equivalent.
A whole four-depth family claim remains incomplete while depths are missing.
Qwen was named in the registration.

| Depth | Pair | Difference | 95% interval | Method | TOST p | Holm p | Equivalent (raw / Holm) | Registration |
|---:|---|---:|---|---|---:|---:|---|---|
| 8192 | bf16-local | +0.0000 | [+0.0000, +0.0000] | paired_percentile_bootstrap (DEGENERATE; not claim evidence) | 0.0001 | 0.0004 | not certified; see sidecars | unknown collection timing; named in registration; not confirmatory |
| 8192 | bf16-offload | +0.0000 | [+0.0000, +0.0000] | paired_percentile_bootstrap (DEGENERATE; not claim evidence) | 0.0001 | 0.0004 | not certified; see sidecars | unknown collection timing; named in registration; not confirmatory |
| 32768 | bf16-local | +0.0000 | [-1.0519, +1.0519] | newcombe10 | 0.00255609 | 0.00766826 | True / True; equivalent; cannot reverse when pending depths arrive | collection after registration; named in registration |
| 32768 | bf16-offload | +0.0000 | [-1.0519, +1.0519] | newcombe10 | 0.00255609 | 0.00766826 | True / True; equivalent; cannot reverse when pending depths arrive | collection after registration; named in registration |

## Sidecars for degenerate sets (post-review supplement)

Claims require both sidecars. Supplementary claim Holm uses max(sidecar p) at a degenerate depth and the registered p otherwise; registered primary numbers above are unchanged.

| Depth | Pair | Sidecar / estimand | 95% interval (pp) | TOST p | Combined claim Holm p (upper bound) | Equivalent from both sidecars |
|---:|---|---|---|---:|---:|---|
| 8192 | bf16-local | exact unseen discordance sidecar / mean raw score difference | [-0.9414, +0.9414] | 0.00037859033 | 0.0095531169 | True; equivalent; cannot reverse when pending depths arrive |
| 8192 | bf16-local | binarised (score == 1) Newcombe sidecar / fully-correct rate difference | [-1.0387, +1.0387] | 0.0023882792 | 0.0095531169 | True; equivalent; cannot reverse when pending depths arrive |
| 8192 | bf16-offload | exact unseen discordance sidecar / mean raw score difference | [-0.9414, +0.9414] | 0.00037859033 | 0.0095531169 | True; equivalent; cannot reverse when pending depths arrive |
| 8192 | bf16-offload | binarised (score == 1) Newcombe sidecar / fully-correct rate difference | [-1.0387, +1.0387] | 0.0023882792 | 0.0095531169 | True; equivalent; cannot reverse when pending depths arrive |

## Collection timing against 2026-11-03 14:36Z

| Depth / pair | Start (UTC) | End (UTC) | Timing |
|---|---|---|---|
| 8192 / bf16-local | None | None | unknown |
| 8192 / bf16-offload | None | None | unknown |
| 32768 / bf16-local | 2026-11-03T15:14:06Z | 2026-11-03T18:41:33Z | after |
| 32768 / bf16-offload | 2026-11-03T15:14:06Z | 2026-11-03T18:41:33Z | after |

<!-- END BF16 TABLES -->

### Adapter validation and review coverage (r16f)

At r16f, the full CPU suite had 72 tests: the unchanged 48 plus 24 new adapter tests. Real 8k and 32k inputs are positive controls, not synthetic substitutes; tests require four staged archives via RULER_BF16_TEST_INPUTS and do not download. Every named negative checks the intended exit 2, a clear refusal, no traceback and no created output. The existing-output negative instead verifies preservation of its sentinel. All original input bytes are preserved.

| Check | Count / real exit | Evidence under .temp/hybrid-d-r16f-261106/ |
|---|---|---|
| Full old + new CPU suite | 72 / 0 | tests-full-01.log; repeated on signed head in tests-full-head-01.log |
| Real adapter runs | 2 processes / 0 | adapt-8192-03.log, adapt-32768-03.log |
| Frozen bf16-family analysis | 0 | analysis-02.log |
| Exact identity, provenance, unchanged f32/Glimmer, regeneration checks | 0 | validation-01.log |

| Mutant | What it narrows the gate to | Named test that fails | Real exit / bound |
|---|---|---|---|
| Map f32 offload instead of bf16 | Correct score equality alone must not substitute the wrong transported arm | test_real_8k_archive_admitted_with_exact_mapping_and_provenance | 1; intended assertion failure |
| Skip baseline-arm equality | A valid resealed row must not replace any frozen baseline result | test_baseline_mismatch_refuses | 1; intended assertion failure |

Both requested mutants are killed; no survivors. These finite controls establish the named mapping/baseline properties, not exhaustive malformed-input coverage. Source changes are confined to tools/hybrid-eval. Scope/security/design/tests/general self-review found no remaining blocker. No surface table was supplied; no surface-table coverage is claimed. No out-of-contract requested row remains: bf16 at 64k/128k is explicitly absent under task clause 3, and 8k timing is unknown under clause 4. Native/CUDA tests and builds are inapplicable to this stdlib-only input adapter and were not run. No network, GPU or supervisor changes occurred; no push or landing is claimed. Temporary adapted row copies are deleted after evidence/regeneration checks; row-hash receipts remain committed.

### r16g: attribution correction and bounded review notes

Muse r7ap4 independently reproduced every statistical value. The corrected 128k attribution above names fwe index 12 (+1/3), qa_2 index 13 (+1) and qa_2 index 14 (-1), identical for both arm pairs. The r16e handoff body and its supporting live table copy are corrected too; the r16f handoff has no such attribution. Historical before/candidate byte snapshots retain the earlier signed-head evidence. No generated JSON, statistical value, statistical code or adapter source changes.

N2 adds test_frozen_baseline_metadata_mismatch_refuses: a resealed max_tokens change must produce exit 2, the specific metadata refusal, no traceback and no output. N3 gives each baseline-arm subtest its own output path. The full suite is now 73 tests (48 unchanged plus 25 adapter tests), exit 0, with both existing real-input positive controls retained. Evidence is under .temp/hybrid-d-r16g-261106/ (tests-full-01.log and signed-head tests-full-head-01.log).

| Mutant rerun | Property narrowed | Named failing test | Real exit / evidence |
|---|---|---|---|
| Map f32 offload instead of bf16 | Correct transported-arm identity, beyond equal scores | test_real_8k_archive_admitted_with_exact_mapping_and_provenance | 1; intended assertion failure |
| Skip baseline-arm equality | Each frozen baseline arm must match independently | test_baseline_mismatch_refuses | 1; all three subtests admit unexpectedly (0 vs 2), with no output-already-exists artifact |

Both mutants are killed; no survivors. The repair changes only report prose and the adapter test file. Native builds are inapplicable and were not run. The live FACTS copy is outside this worktree; its matching attribution correction is included in the coordinator handoff.

## Historical r16e validation and review coverage

CPU only, PYTHONDONTWRITEBYTECODE=1, disk-backed .temp/hybrid-d-r16e-261106/tmp. No downloads, GPU activity or supervisor/output changes. No statistical code, constants or test sources changed from accepted e46e5623. All 48 existing test names and positive/negative controls are retained.

| Check | Count / real exit | Evidence under .temp/hybrid-d-r16e-261106/ |
|---|---|---|
| Frozen equivalence suite | 35 tests / 0 | tests-equivalence-01.log |
| Existing evaluator suite | 11 tests / 0 | tests-eval-01.log |
| Existing paired analysis/archive tests | 2 tests / 0 | tests-secondary-01.log |
| Frozen four-depth CLI | 0 | analysis-01.log |
| bf16 direct admission negative | 2, intended refusal; no output directory | bf16-refusal-01.log |
| Rows, source identities, CSVs, all 56 secondary tables, prefix custody | 0 | verification-01.log |

Total: 48 CPU tests green. The frozen loader's bf16 refusal is reported as an expected exit 2, not a passing analysis. Glimmer JSON/table bytes and frozen statistical/test sources are checked against the accepted head. This is a report/data-only update; no new behavior or mutant claim is introduced. The prior accepted r16c mutation evidence (11 intended kills, no survivors) applies to unchanged code, not a new run here. Review history is retained in git at e46e5623. No native build or remote CI is needed for this CPU report update, and neither was run. No brief surface table was supplied; coverage is limited to the stated frozen tests and saved-input checks. At r16e, bf16 was unissued under that task's clause 4 no-new-code fallback; r16f now supplies the separately tested input adapter and existing-depth analysis. No push or landing is claimed.

## Regeneration of the preserved f32 results (r16e)

Run the following from the repository root. Original inputs stay read-only. All archives are small; verify all compressed and member hashes before selecting analysis files. This block regenerates the frozen output and replaces only the Qwen generated table, preserving Glimmer. The descriptive/claim tables are also regenerated from accepted JSON fields; no statistic is added.

```sh
set -eu
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/.temp/hybrid-d-r16e-261106/tmp"
mkdir -p "$TMPDIR"
python3 - <<'PY_EXTRACT'
from pathlib import Path
import hashlib
import shutil
import tarfile

base = Path('/home/agent/lanes/hybrid/.temp')
work = Path('.temp/hybrid-d-r16e-261106')
inputs = [
    ('qwen-8192', 'ruler-results-261103/depth-8192-complete.tar.gz', '2be3eae64b783867258a1f22e1c7bbdb4188f257c378f17c07554a8868248aa4'),
    ('qwen-32768', 'ruler-results-261103/depth-32768-complete.tar.gz', 'a6fb1e7e6ac0fb7d25ed3ed24ef1f47cdf4f6495b54a4866d6486870bfeef176'),
    ('qwen-65536', 'ruler-results-261106/qwen-inputs/depth-65536-complete.tar.gz', '012c9308fcd7cc4343deb023ced9f026d9fe26cc6a3e4dd8fdc63ebf75c13fd8'),
    ('qwen-131072', 'ruler-results-261106/qwen-inputs/depth-131072-complete.tar.gz', '36da2f0328ad2145dcab3ed47cc215744708d66b90602b42e0495666baecc94e'),
]
for depth in (8192, 32768):
    relative = f'ruler-results-261103/bf16-8d876c71/bf16-{depth}-complete.tar.gz'
    expected = Path(str(base/relative)+'.sha256').read_text().split()[0]
    inputs.append((f'bf16-{depth}', relative, expected))
for label, relative, expected in inputs:
    assert shutil.disk_usage('.').free >= 28*1024**3, 'disk below 28 GiB'
    target = work/'archives'/Path(relative).name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base/relative, target)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == expected
    with tarfile.open(target) as archive:
        members = archive.getmembers()
        assert len({m.name for m in members}) == len(members)
        assert all(m.isfile() for m in members)
        assert sum(m.size for m in members) < 30_000_000
        checked = set()
        for line in archive.extractfile('SHA256SUMS').read().decode().splitlines():
            checksum, name = line.split('  ', 1)
            assert name not in checked
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == checksum
            checked.add(name)
        assert checked == {m.name for m in members}-{'SHA256SUMS'}
        folder = work/'extracted'/label
        folder.mkdir(parents=True, exist_ok=True)
        for name in ('rows.jsonl', 'manifest.json', 'per-example.csv', 'analysis.json'):
            (folder/name).write_bytes(archive.extractfile(name).read())
(work/'run-times.json').write_text('{"Qwen":{"f32":{"65536":{"start":"2026-11-03T19:20:51Z","end":"2026-11-05T14:28:50Z"},"131072":{"start":"2026-11-03T14:29:06Z","end":"2026-11-06T05:56:57Z"}}}}\n')
PY_EXTRACT
python3 tools/hybrid-eval/ruler_equivalence.py \
    .temp/hybrid-d-r16e-261106/extracted/qwen-8192/rows.jsonl \
    .temp/hybrid-d-r16e-261106/extracted/qwen-32768/rows.jsonl \
    .temp/hybrid-d-r16e-261106/extracted/qwen-65536/rows.jsonl \
    .temp/hybrid-d-r16e-261106/extracted/qwen-131072/rows.jsonl \
    --family Qwen --run-times .temp/hybrid-d-r16e-261106/run-times.json \
    --output .temp/hybrid-d-r16e-261106/qwen
python3 -m unittest discover -s tools/hybrid-eval -p test_ruler_equivalence.py -v
PYTHONPATH=tools/hybrid-eval python3 -m unittest \
    test_ruler.RulerTests.test_known_paired_table_and_intervals \
    test_ruler.RulerTests.test_complete_archive_and_member_hashes
python3 -m unittest discover -s tools/hybrid-eval -p test_eval.py
python3 - <<'PY_REPORT'
from pathlib import Path
import json
import shutil

work = Path('.temp/hybrid-d-r16e-261106')
root = Path('tools/hybrid-eval')
report = root/'RULER-EQUIVALENCE-RESULTS.md'
text = report.read_text()
start = text.index('<!-- BEGIN GENERATED TABLES -->')+len('<!-- BEGIN GENERATED TABLES -->')
stop = text.index('### Glimmer: paired RULER equivalence', start)
table = (work/'qwen/equivalence.md').read_text()
table = table.replace('# Qwen:', '### Qwen:', 1).replace('## Secondary:', '#### Secondary:')
text = text[:start]+'\n'+table+'\n'+text[stop:]
r = json.loads((work/'qwen/equivalence.json').read_text())
lines = [
    '| Depth | Pair | Registered Holm p | Claim Holm p | Claim under the accepted policy |',
    '|---:|---|---:|---:|---|',
]
for row in r['primary']:
    verdict = 'equivalent' if row['equivalent_holm'] else 'not established'
    if row['degenerate']:
        verdict += ' via sidecars; degenerate primary unusable'
    elif 'sidecars' in row:
        verdict += '; unseen-range sidecar crosses margin'
    lines.append(f"| {row['depth']} | {row['pair']} | {row['holm_p']:.9g} | {row['claim_holm_p']:.9g} | {verdict} |")
lines += ['', 'Per-arm scores and secondary text statistics reproduce the accepted evaluator and the FACTS blocks at their printed precision. Text comparisons use outer-whitespace equality, exact two-sided unadjusted McNemar, Wilson 95% rate intervals, and the accepted conservative excess interval. A nonsignificant McNemar p is not equivalence.', '',
          '| Depth | Local / offload / partition score (%) | Fully correct local / offload / partition | Accuracy McNemar p offload / partition |',
          '|---:|---|---|---|']
all_rows = [row for row in r['secondary'] if row['task'] == 'ALL']
for row in all_rows:
    off, part = [row['accuracy_pairs'][a] for a in ('offload', 'local_partition')]
    counts = [off['both_true']+off['right_only'], off['both_true']+off['left_only'], part['both_true']+part['left_only']]
    means = ' / '.join(f"{100*row['score_means'][a]:.2f}" for a in ('local', 'offload', 'local_partition'))
    correct = ' / '.join(str(n) for n in counts)
    ps = ' / '.join(f"{row['accuracy_pairs'][a]['mcnemar_p']:.6g}" for a in ('offload', 'local_partition'))
    lines.append(f"| {row['depth']} | {means} | {correct} | {ps} |")
lines += ['', '| Depth | Offload changed/n, % [Wilson 95%] | Partition changed/n, % [Wilson 95%] | Offload-only / partition-only | Text McNemar p | Excess pp [95%] | Text equivalent +/-2 pp |',
          '|---:|---|---|---|---:|---|---|']
for row in all_rows:
    rates = []
    for arm in ('offload', 'partition'):
        lo, hi = row[arm+'_wilson95']
        rates.append(f"{row[arm+'_changed']}/{row['n']}, {100*row[arm+'_rate']:.2f} [{100*lo:.2f}, {100*hi:.2f}]")
    lo, hi = row['excess_conservative95']
    lines.append(f"| {row['depth']} | {rates[0]} | {rates[1]} | {row['left_only']} / {row['right_only']} | {row['mcnemar_p']:.6g} | {100*row['excess']:+.4f} [{100*lo:+.4f}, {100*hi:+.4f}] | {row['equivalent_at_margin']} |")
start = text.index('<!-- BEGIN QWEN DETAIL TABLES -->')+len('<!-- BEGIN QWEN DETAIL TABLES -->')
stop = text.index('<!-- END QWEN DETAIL TABLES -->', start)
text = text[:start]+'\n'+'\n'.join(lines)+'\n'+text[stop:]
report.write_text(text)
shutil.copyfile(work/'qwen/equivalence.json', root/'equivalence-results/qwen.json')
# Remove only this run's temporary selective extracts after validation.
shutil.rmtree(work/'extracted')
PY_REPORT
```

A direct bf16 check uses the two extracted bf16 rows.jsonl paths with the same family/depth declarations and its own output directory, before the cleanup. It must currently refuse with exit 2. That direct frozen-loader check still refuses. For admitted bf16 inputs use the separate adapter and the regeneration commands below; original checksums and derived identities remain linked.

## Regeneration of bf16 admission and analysis (r16f)

Run from the repository root, using an absent adapted/depth-8192 and adapted/depth-32768 below BF16_WORK. Use a fresh task-owned directory for a second run; the adapter refuses overwrite. These commands copy only small archives, independently verify each archive and every member in the adapter, derive conservative timing bounds, run the frozen statistical tool, and replace only the bf16 JSON/receipt/table. The source archives and f32/Glimmer results stay unchanged. The two declared source hashes in the timing evidence were checked against the signed local mirror object 8d876c71; no network is needed.

```sh
set -eu
export PYTHONDONTWRITEBYTECODE=1
export BF16_WORK="${BF16_WORK:-$PWD/.temp/hybrid-d-r16f-261106}"
export TMPDIR="$BF16_WORK/tmp"
mkdir -p "$TMPDIR" "$BF16_WORK/inputs" "$BF16_WORK/adapted"
python3 - <<'PY_INPUTS'
import hashlib
import os
from pathlib import Path
import shutil

source = Path('/home/agent/lanes/hybrid/.temp/ruler-results-261103')
work = Path(os.environ['BF16_WORK'])
inputs = [
    ('bf16-8d876c71/bf16-8192-complete.tar.gz', 'c0348572c2c1baf50dcfcdd0a73397594ba0d12e45ba157b0ae63980f25799b6'),
    ('bf16-8d876c71/bf16-32768-complete.tar.gz', '5efae77aa8b9902bd31b9b2a1ba336a0d8a090353898fd31fee8c980e73e1fbe'),
    ('depth-8192-complete.tar.gz', '2be3eae64b783867258a1f22e1c7bbdb4188f257c378f17c07554a8868248aa4'),
    ('depth-32768-complete.tar.gz', 'a6fb1e7e6ac0fb7d25ed3ed24ef1f47cdf4f6495b54a4866d6486870bfeef176'),
]
for name, expected in inputs:
    assert shutil.disk_usage('.').free >= 28*1024**3, 'disk below 28 GiB'
    path = work/'inputs'/Path(name).name
    shutil.copyfile(source/name, path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
PY_INPUTS
python3 tools/hybrid-eval/ruler_bf16_adapter.py \
    --bf16-archive "$BF16_WORK/inputs/bf16-8192-complete.tar.gz" \
    --bf16-sha256 c0348572c2c1baf50dcfcdd0a73397594ba0d12e45ba157b0ae63980f25799b6 \
    --baseline-archive "$BF16_WORK/inputs/depth-8192-complete.tar.gz" \
    --baseline-sha256 2be3eae64b783867258a1f22e1c7bbdb4188f257c378f17c07554a8868248aa4 \
    --output "$BF16_WORK/adapted/depth-8192"
python3 tools/hybrid-eval/ruler_bf16_adapter.py \
    --bf16-archive "$BF16_WORK/inputs/bf16-32768-complete.tar.gz" \
    --bf16-sha256 5efae77aa8b9902bd31b9b2a1ba336a0d8a090353898fd31fee8c980e73e1fbe \
    --baseline-archive "$BF16_WORK/inputs/depth-32768-complete.tar.gz" \
    --baseline-sha256 a6fb1e7e6ac0fb7d25ed3ed24ef1f47cdf4f6495b54a4866d6486870bfeef176 \
    --output "$BF16_WORK/adapted/depth-32768"
python3 - <<'PY_TIMING'
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import tarfile

work = Path(os.environ['BF16_WORK'])
start = datetime(2026, 11, 3, 14, 23, 39, tzinfo=timezone.utc)
end = '2026-11-03T18:41:33Z'
timing = {}
for depth in (8192, 32768):
    with tarfile.open(work/f'inputs/bf16-{depth}-complete.tar.gz') as archive:
        rows = [json.loads(line) for line in archive.extractfile('rows.jsonl')]
    seconds = math.fsum(row['results']['offload_bf16']['seconds'] for row in rows)
    timing[str(depth)] = dict(n=390, serial_bf16_call_seconds=seconds,
        classification='unknown' if depth == 8192 else 'after',
        start_semantics='not known' if depth == 8192 else 'conservative lower bound, not a measured start',
        end_semantics='stage end is an upper bound, not a measured depth end')
assert timing['8192']['serial_bf16_call_seconds'] > 741
lower = start+timedelta(seconds=timing['8192']['serial_bf16_call_seconds'])
lower = lower.replace(microsecond=0).isoformat().replace('+00:00', 'Z')
run_times = {'Qwen': {'bf16': {'32768': dict(start=lower, end=end)}}}
(work/'run-times.json').write_text(json.dumps(run_times, sort_keys=True, indent=2)+'\n')
timing.update(stage_execute_start='2026-11-03T14:23:39Z', stage_end=end, stage_return_code=1,
    stage_failure='post-arm service restore; both depths complete', registration='2026-11-03T14:36:00Z',
    pre_registration_seconds=741, earliest_32k_start_bound=lower, run_times=run_times,
    note='8k ends after registration but setup/build latency leaves its start unknown; 32k follows all 8k serial calls. Times passed to the frozen tool enclose collection; no per-depth timestamps are asserted.')
timing['stage_ruler_bf16-pinned.py_sha256'] = 'dde8090436b3f732040a2fe3773f94b9eda45d1698181972ed7013bcb8b5e8f5'
timing['ruler_eval-bf16-pinned.py_sha256'] = '5a5402ba05db37e02c9916f243a244afbff4e33633bb1f972f2c1210bac00623'
(work/'timing-evidence.json').write_text(json.dumps(timing, sort_keys=True, indent=2)+'\n')
PY_TIMING
python3 tools/hybrid-eval/ruler_equivalence.py \
    "$BF16_WORK/adapted/depth-8192/rows.jsonl" "$BF16_WORK/adapted/depth-32768/rows.jsonl" \
    --family Qwen --run-times "$BF16_WORK/run-times.json" --output "$BF16_WORK/bf16"
python3 - <<'PY_BF16_REPORT'
from pathlib import Path
import json
import os
import shutil
import sys
sys.path.insert(0, 'tools/hybrid-eval')
from ruler_equivalence import markdown

work = Path(os.environ['BF16_WORK'])
root = Path('tools/hybrid-eval')
result = json.loads((work/'bf16/equivalence.json').read_text())
pairs = [row for row in result['primary'] if row['pair'].startswith('bf16-')]
receipts = [json.loads((work/f'adapted/depth-{d}/provenance.json').read_text()) for d in (8192, 32768)]
provenance = dict(schema='ruler-bf16-admission-v1', inputs=receipts,
                  timing=json.loads((work/'timing-evidence.json').read_text()), families={})
for pair in ('bf16-local', 'bf16-offload'):
    observed = {row['depth']: row for row in pairs if row['pair'] == pair}
    provenance['families'][pair] = [dict(depth=d, measured=d in observed,
        status='observed' if d in observed else 'absent, never measured',
        registered_holm_input=observed[d]['tost_p'] if d in observed else 1,
        claim_holm_input=observed[d]['claim_p'] if d in observed else 1) for d in result['depths']]
(root/'equivalence-results/bf16-input-provenance.json').write_text(json.dumps(provenance, sort_keys=True, indent=2)+'\n')
shutil.copyfile(work/'bf16/equivalence.json', root/'equivalence-results/qwen-bf16.json')
view = dict(result, primary=pairs, secondary=[])
table = markdown(view).split('\n## Secondary:')[0]
table = table.replace('# Qwen: paired RULER equivalence', '### Qwen bf16: paired RULER score equivalence', 1)+'\n'
report = root/'RULER-EQUIVALENCE-RESULTS.md'
text = report.read_text()
start = text.index('<!-- BEGIN BF16 TABLES -->')+len('<!-- BEGIN BF16 TABLES -->')
stop = text.index('<!-- END BF16 TABLES -->', start)
report.write_text(text[:start]+'\n'+table+text[stop:])
PY_BF16_REPORT
export RULER_BF16_TEST_INPUTS="$BF16_WORK/inputs"
PYTHONPATH=tools/hybrid-eval python3 -m unittest -v \
    test_ruler_equivalence test_eval \
    test_ruler.RulerTests.test_known_paired_table_and_intervals \
    test_ruler.RulerTests.test_complete_archive_and_member_hashes test_ruler_bf16_adapter
```

After checking the regenerated bytes and retaining provenance.json (including all original/derived row-hash links), remove only BF16_WORK/adapted. The source archive copies are small and can remain for the real-input CPU tests. No inference, model, source checkout/build or provider request is part of this regeneration.
