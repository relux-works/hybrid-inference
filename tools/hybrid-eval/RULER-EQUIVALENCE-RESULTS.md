# RULER equivalence: review rework hybrid-d-r16b-261103

Analysis on agents1, 2026-11-03; rework of signed head `422813a69` after Muse r7ap CHANGES REQUESTED. Base and contract: `m3-ruler-prereg` at `7556f46165b01216ff1028c381845c8e79c65955`, [preregistration](RULER-EQUIVALENCE-PREREGISTRATION.md). This is a local analysis on the private prototype fork, with no publication or GPU execution. RC4 data cut-off: 2026-11-05 12:00Z.

## Analysis rule and review disposition

The registered primary endpoint and every numerical primary result from `422813a69` are unchanged: the equally weighted mean over 390 paired task scores, a 95% interval, inverted TOST p, and the original four-depth Holm p. The interval rule frozen before the first analysis is Newcombe method 10 if every score in both compared arms is binary, otherwise one paired bootstrap over all 390 pairs. Each depth still requires 13 tasks x indices 0-29, shared prompts and seed 261102; incomplete, duplicate, mismatched or invalid rows refuse. CSV requires companion JSONL and manifest to verify identities and seeds.

[Newcombe (1998), section 5, method 10 and Table III](https://www.site.uottawa.ca/~nat/Courses/csi5388/Newcombe.1998.pdf) supplies the binary interval and published controls. Method 10 corrects positive paired phi by subtracting n/2 from its numerator, clipped at zero; negative phi is unchanged and a zero denominator gives phi=0. Marginal Wilson limits have no continuity correction.

### R1: degenerate primary and post-review sidecars

A point-mass bootstrap interval cannot certify equivalence. Qwen 8k has 377 fully correct, 11 zero and two 2/3 scores in each arm, with 0/390 nonzero paired differences. Its registered [0,0] interval, floored p=0.0001 and original Holm p=0.0004 remain visible, explicitly DEGENERATE and excluded as claim evidence. Both arm pairs instead get these post-review sidecars:

- Exact unseen discordance: with zero discordant pairs, the two-sided 95% Clopper-Pearson upper endpoint is `u = 1 - 0.025^(1/390) = 0.00941407`. Since raw differences lie in [-1,1], their population mean lies in [-u,+u], or [-0.9414,+0.9414] pp. Its inverted one-sided p is `0.98^390 = 0.0003785903`.
- Binarised (score == 1) Newcombe: cells (both correct, left only, right only, neither) = (377,0,0,13), interval [-1.0387,+1.0387] pp, TOST p=0.00238828. This is the fully-correct-rate estimand, not the registered mean raw score.

The Qwen 8k equivalence statement is supported by both sidecars with roughly 1 pp half-widths, not by [0,0]. The automatic rule requires both sidecar intervals inside the margin and uses their maximum p for a supplementary four-depth claim Holm calculation; other depths use their registered p. At Qwen 8k the supplementary Holm upper bound is 0.00955312 for both pairs. This conservative supplementary decision is distinct from the unchanged registered primary Holm arithmetic. JSON flags `equivalent_unadjusted` and `equivalent_holm` refer to this explicit `claim_basis`, with `claim_p` and `claim_holm_p`; `registered_interval_inside_margin` records only geometric containment, never a claim from a point mass.

The same rule applies automatically to every future constant-difference partial-credit bootstrap set, including bf16, and refuses the n=3 zero-difference claim. For a constant nonzero difference c, the first pair fixes that observed value, and the remaining n-1 pairs bound unseen departures without selecting the value on those same trials; at n=1 the bound is [-1,1]. With unseen mass bounded by u the raw-mean sidecar is `[c-(1+c)*u, c+(1-c)*u]`. A shifted partial-credit fixture cannot pass merely because binarisation discards its shift. Binary-score sets retain their ordinary nondegenerate Newcombe primary.

### R2: actual collection timing and registration scope

The registration timestamp is 2026-11-03 14:36Z. Explicit family/arm-variant/depth maps record run start and end; completion by registration means before, a start at or after registration means after, and overlap means straddling. Unknown dates remain unknown rather than inferred from depth. Qwen 8k ran 05:52-07:32Z and 32k ran 07:32-14:21Z, both before. Glimmer 8k ran 13:59-15:35Z including its pilot: collection straddled registration and is not confirmatory.

**Glimmer was not named in the registration. The same analysis rules are applied to Glimmer by extension**, including later after-registration runs; timing alone does not preregister that family. Qwen's later depths and bf16 arms are named by the contract, but start/end timestamps for unmeasured arrivals are not invented. `--run-times` supplies witnessed future metadata without changing source; an absent future Glimmer32 timestamp stays unknown, and a recorded later run is labelled after, never before. Format example: `{"Glimmer":{"f32":{"32768":{"start":"2026-11-03T15:35:00Z","end":"2026-11-04T02:00:00Z"}}}}`; bf16 uses a separate `bf16` map. This example is a format fixture, not an observed future result.

### R3 / N1: incomplete families and Holm upper bounds

Pending p=1 is an upper bound, not an uncertain observed verdict. Replacing pending ones by future p-values can only lower or keep each observed adjusted p. Qwen's observed accepted per-depth decisions (8k from sidecars, 32k from registered intervals) therefore cannot reverse because pending depths arrive. The whole four-depth family claim remains incomplete. Glimmer 8k cannot become equivalent: its raw intervals already cross +2 pp; only its Holm p upper bound can decrease. A future raw-pass/Holm-fail row may become accepted as pending values arrive, and the tool distinguishes that case. Changes to data, frozen methods or metadata are outside this monotonicity statement.

Families are separate for each model and arm pair at alpha=0.025; this controls error per family, not a single family-wise level across pairs or models. Qwen's planned depths are 8192,32768,65536,131072; Glimmer's are 8192,32768,65536,122880. No bf16 observations are in these three archives. Its two pairs remain pending, with explicit `results.bf16` required on all matched rows when available.

### N2-N4: frozen implementation choices and remaining limits

The registration specifies the score estimand, 95% intervals, Newcombe method 10 / partial-credit bootstrap with 10000 resamples and seed 261102, +/-2 pp margin, TOST alpha=0.025 per side, and four-depth Holm. It does **not** specify the following fill-ins, now frozen for future depths and bf16:

| Choice absent from registration | Frozen implementation | When fixed |
|---|---|---|
| Mixed binary/partial-credit sets | One bootstrap over all pairs if either arm contains any partial score; otherwise method 10 | r16 before calculation |
| Bootstrap construction | Percentile, unstratified, sorted (task,index); `random.Random(261102).choices` restarted per pair; type-7 quantiles at `(B-1)*q` | r16 before calculation |
| P construction and finite resolution | Invert central intervals at coverage `1-2*alpha`; max one-sided p, no doubling; bracket 1e-15; conservative bootstrap floor 1/10000; p=1 if no alpha<=0.5 certifies containment | r16 implementation |
| Holm families | Separate per model and arm pair, without cross-family error control | r16 before calculation |
| Pending entry | p=1 upper bounds, with the stability interpretation above | r16 entry; r16b clarifies wording |
| Degenerate-case sidecars | Keep registered results; add exact raw-score bound and score==1 Newcombe; both required, max sidecar p enters supplementary claim Holm | r16b, post-review and post-observation |
| Near-margin diagnostics | If a nondegenerate bootstrap endpoint is within 0.1 pp of either margin, report five seeds 261100-261104 without changing the registered seed or decision | r16b, post-review |

For N2, an inverted percentile-bootstrap p is approximate, a construction for Holm rather than an exact registered test. P-values just above 1/10000 are still coarse even without a resolution flag. The registered criterion remains strict interval containment; the R1 sidecar claim is explicitly supplemental. Current Glimmer endpoints are over 0.6 pp from +2, so the new near-margin trigger is false for this cut; a named CPU test exercises its five-seed diagnostics on a near-margin fixture. No cross-Python-version stability is claimed; JSON records the actual Python version.

For N4, the public CLI now admits only Qwen or Glimmer and verifies that label against the recorded model SHA-256. The transitive bench/proxy import chain remains: the unchanged secondary analyser imports it for shared helpers. Removing it would require changing accepted secondary infrastructure or duplicating that analyser; this narrow statistical/reporting rework deliberately preserves its source hash and tests. These imports do not start services or make network calls in this workflow. Import isolation remains a non-blocking follow-up, not a claim that the chain was removed.

## Verified source archives

All three compressed SHA-256 values matched the supplied abbreviated hashes before extraction. Every member listed in each archive's `SHA256SUMS` also matched. Selective extracts occupied under 6 MB and are deleted after validation. Original archives and supervisor outputs are unchanged.

| Model | Depth | Archive under `/home/agent/lanes/hybrid/.temp/ruler-results-261103/` | SHA-256 |
|---|---:|---|---|
| Qwen | 8192 | `depth-8192-complete.tar.gz` | `2be3eae64b783867258a1f22e1c7bbdb4188f257c378f17c07554a8868248aa4` |
| Qwen | 32768 | `depth-32768-complete.tar.gz` | `a6fb1e7e6ac0fb7d25ed3ed24ef1f47cdf4f6495b54a4866d6486870bfeef176` |
| Glimmer | 8192 | `glimmer-715c81a4/results-8192.tar.gz` | `2ff122d17f892dcb6f495306e06d2fb1671e0b145523df18605c58a3b6bfe8f1` |

Committed machine-readable results: [Qwen](equivalence-results/qwen.json), [Glimmer](equivalence-results/glimmer.json). They include input hashes, analysis source hashes, Python identity, descriptive task means, every unchanged secondary task table and explicit pending depths. Native inference is not part of this analysis.

## Paper tables

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
| 32768 | local_partition-local | -0.2564 | [-1.4704, +0.8517] | newcombe10 | 0.00725632 | 0.021769 | True / True; equivalent; cannot reverse when pending depths arrive | observed before registration; named in registration |
| 32768 | offload-local | +0.0000 | [-1.0519, +1.0519] | newcombe10 | 0.00255609 | 0.00766826 | True / True; equivalent; cannot reverse when pending depths arrive | observed before registration; named in registration |

## Sidecars for degenerate sets (post-review supplement)

Claims require both sidecars. Supplementary claim Holm uses max(sidecar p) at a degenerate depth and the registered p otherwise; registered primary numbers above are unchanged.

| Depth | Pair | Sidecar / estimand | 95% interval (pp) | TOST p | Combined claim Holm p (upper bound) | Equivalent from both sidecars |
|---:|---|---|---|---:|---:|---|
| 8192 | local_partition-local | exact unseen discordance sidecar / mean raw score difference | [-0.9414, +0.9414] | 0.00037859033 | 0.0095531169 | True; equivalent; cannot reverse when pending depths arrive |
| 8192 | local_partition-local | binarised (score == 1) Newcombe sidecar / fully-correct rate difference | [-1.0387, +1.0387] | 0.0023882792 | 0.0095531169 | True; equivalent; cannot reverse when pending depths arrive |
| 8192 | offload-local | exact unseen discordance sidecar / mean raw score difference | [-0.9414, +0.9414] | 0.00037859033 | 0.0095531169 | True; equivalent; cannot reverse when pending depths arrive |
| 8192 | offload-local | binarised (score == 1) Newcombe sidecar / fully-correct rate difference | [-1.0387, +1.0387] | 0.0023882792 | 0.0095531169 | True; equivalent; cannot reverse when pending depths arrive |

## Collection timing against 2026-11-03 14:36Z

| Depth / pair | Start (UTC) | End (UTC) | Timing |
|---|---|---|---|
| 8192 / local_partition-local | 2026-11-03T05:52:00Z | 2026-11-03T07:32:00Z | before |
| 8192 / offload-local | 2026-11-03T05:52:00Z | 2026-11-03T07:32:00Z | before |
| 32768 / local_partition-local | 2026-11-03T07:32:00Z | 2026-11-03T14:21:00Z | before |
| 32768 / offload-local | 2026-11-03T07:32:00Z | 2026-11-03T14:21:00Z | before |

#### Secondary: answer text (unchanged ruler_analysis.py)

| Depth | Offload changed / n | Partition changed / n | McNemar p | Excess 95% (pp) | Text equivalent +/-2 pp |
|---:|---:|---:|---:|---|---|
| 8192 | 34/390 | 32/390 | 0.726562 | [-2.1275, +3.1189] | False |
| 32768 | 33/390 | 29/390 | 0.617719 | [-4.0224, +6.0473] | False |

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

## FACTS cross-check

All totals and all 13 per-task score triplets match section 3 of `/home/agent/lanes/hybrid/.temp/paper-v2/FACTS.md` at its published precision. All answer-text counts, discordant cell counts and interval endpoints also match. Every archived secondary table was compared to the new tool's delegated `ruler_analysis.py` output with exact equality, including per-task means, Wilson intervals, McNemar p-values and accuracy cells. The secondary implementation is unchanged, SHA-256 `f0fa95fdf3890b0a485a6f9ae1a0be96c8177ffd46312664a6857ad370990635`.

| Model/depth | Local score % | Offload score % | Partition score % | Offload/partition changed answers | Offload-only/partition-only text changes |
|---|---:|---:|---:|---|---|
| Qwen 8k | 97.01 | 97.01 | 97.01 | 34/390, 32/390 | 5, 3 |
| Qwen 32k | 97.44 | 97.44 | 97.18 | 33/390, 29/390 | 20, 16 |
| Glimmer 8k | 87.91 | 89.13 | 89.08 | 130/390, 125/390 | 13, 8 |

Text equivalence is not established for any of these depths. The nonsignificant McNemar tests are not used to infer score or text equivalence.

## Validation and review coverage

CPU only, `PYTHONDONTWRITEBYTECODE=1`, disk-backed `.temp/hybrid-d-r16b-261103/tmp`. No installs, model/data downloads, GPU activity or supervisor/output access. The r16b extract command stops below 28.2 GB free; extracts stay under 6 MB and are removed at the end.

| Check | Count / real exit | Evidence under `.temp/hybrid-d-r16b-261103/` |
|---|---|---|
| Updated equivalence suite, all original test names retained | 26 tests / 0 | `regeneration-02.log` (final source; `unit-02.log` also green) |
| Existing evaluator suite | 11 tests / 0 | `evaluator-regression-01.log` |
| Existing paired analysis and archive tests | 2 tests / 0 | `secondary-regression-01.log` |
| Byte-for-byte report/JSON regeneration, archive/member hashes and both CLI analyses | 0 | `regeneration-03.log`, `reproduction-01.log` |
| Registered primary numerical fields vs reviewed head; every secondary/descriptive table unchanged; sidecar golden values and timing | 0 | `result-comparison-01.log` |
| Real CSV and JSONL input endpoints agree | 0 | `csv-comparison-01.log` |
| Syntax, scoped style and diff checks | 0 | `syntax-style-01.log`, `diff-check-01.log` |

Total: 39 CPU tests green. Regression names from r16 are preserved, including `test_identical_partial_pairs_and_bootstrap_resolution`, whose n=3 equivalence assertion now correctly rejects. This is rework within `tools/hybrid-eval/` only. No requested analysis rows were excluded as out of contract. The referenced review's surface rows map to these bounded controls; this is not a claim of exhaustive attack-surface coverage.

| Review surface | Named test / evidence | Narrowing mutant and real test exit |
|---|---|---|
| 1a Mixed-set interval rule | `test_mixed_scores_use_whole_set_bootstrap` | Force binary branch / 1 |
| 1b Family/run registration timing | `test_glimmer32_registration_uses_run_times_not_depth` | Restore depth-only before label / 1 |
| 2 Degenerate claim evidence | `test_identical_partial_pairs_and_bootstrap_resolution` | Skip sidecar gate / 1 |
| 3 Holm ordering / bounds | `test_holm_ordering_and_pending_depth`, `test_pending_holm_upper_bounds_do_not_reverse_observed_decisions` | Ignore Holm order / 1 |
| 4 Mean and independent numerical reproduction | `test_shifted_fixture_not_equivalent_positive_control`, result comparison | Replace reported mean by zero / 1 |
| 5 Newcombe / bootstrap / inversion | `test_newcombe10_published_reference_cells`, deterministic bootstrap and normal inversion tests | Omit positive phi correction / 1 |
| 6 Named negatives | `test_interval_crossing_margin_is_not_equivalent`, `test_changed_seed_refuses`, `test_nonsignificant_mcnemar_never_implies_equivalence` | Accept every interval; skip row seed check; force raw equivalence / 1 each |

The initial control driver exited 1 before applying the new sidecar mutant because its replacement matched two sites; it was narrowed to the score gate and rerun. That setup failure is not credited as a killed mutant. All nine mutants were confined to `.temp/`, each named test exited 1 with the intended assertion failure, and the control driver exited 0 only after verifying those failures; no survivors. Evidence: `plant-controls-03.log` and per-mutant logs. The raw primary numbers and unchanged secondary implementation are also independently verified by Muse r7ap for the prior head; this candidate awaits re-review. Native C++ build, hosted CI, `ty` and `editorconfig-checker` were not run: native code is unchanged and the latter tools are unavailable under the no-download rule. Syntax/style checks cover the changed Python files. Scope/progress remain in `.temp/tasks.md` because no native task-board exists in this workbench. No push or landing is claimed.

## Regeneration

Run from the repository root. The original archives stay in place. This verifies full hashes and internal hashes before extracting only the small analysis inputs. It refuses duplicate archive members and oversized archives. It writes to the task's own `.temp/` directory.

```sh
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/.temp/hybrid-d-r16b-261103/tmp"
mkdir -p "$TMPDIR"
python3 - <<'PY_EXTRACT'
import hashlib
from pathlib import Path
import shutil
import tarfile

source = Path('/home/agent/lanes/hybrid/.temp/ruler-results-261103')
work = Path('.temp/hybrid-d-r16b-261103')
archives = [
    ('qwen-8192', 'depth-8192-complete.tar.gz', '2be3eae64b783867258a1f22e1c7bbdb4188f257c378f17c07554a8868248aa4'),
    ('qwen-32768', 'depth-32768-complete.tar.gz', 'a6fb1e7e6ac0fb7d25ed3ed24ef1f47cdf4f6495b54a4866d6486870bfeef176'),
    ('glimmer-8192', 'glimmer-715c81a4/results-8192.tar.gz', '2ff122d17f892dcb6f495306e06d2fb1671e0b145523df18605c58a3b6bfe8f1'),
]
for label, name, expected in archives:
    if shutil.disk_usage('.').free < 28_200_000_000:
        raise SystemExit('disk below 28.2 GB; stop before extraction')
    path = source/name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        assert len({m.name for m in members}) == len(members)
        assert sum(m.size for m in members) < 8_000_000
        for line in archive.extractfile('SHA256SUMS').read().decode().splitlines():
            checksum, member = line.split('  ', 1)
            assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == checksum
        folder = work/'extracted'/label
        folder.mkdir(parents=True, exist_ok=True)
        for member in ('rows.jsonl', 'manifest.json', 'per-example.csv', 'analysis.json'):
            (folder/member).write_bytes(archive.extractfile(member).read())
PY_EXTRACT
python3 tools/hybrid-eval/ruler_equivalence.py \
    .temp/hybrid-d-r16b-261103/extracted/qwen-8192/rows.jsonl \
    .temp/hybrid-d-r16b-261103/extracted/qwen-32768/rows.jsonl \
    --family Qwen --output .temp/hybrid-d-r16b-261103/qwen
python3 tools/hybrid-eval/ruler_equivalence.py \
    .temp/hybrid-d-r16b-261103/extracted/glimmer-8192/rows.jsonl \
    --family Glimmer --depths 8192 32768 65536 122880 \
    --output .temp/hybrid-d-r16b-261103/glimmer
python3 -m unittest discover -s tools/hybrid-eval -p test_ruler_equivalence.py -v
PYTHONPATH=tools/hybrid-eval python3 -m unittest \
    test_ruler.RulerTests.test_known_paired_table_and_intervals \
    test_ruler.RulerTests.test_complete_archive_and_member_hashes
python3 -m unittest discover -s tools/hybrid-eval -p test_eval.py
python3 - <<'PY_REPORT'
from pathlib import Path
import shutil

work = Path('.temp/hybrid-d-r16b-261103')
root = Path('tools/hybrid-eval')
report = root/'RULER-EQUIVALENCE-RESULTS.md'
begin = '<!-- BEGIN GENERATED TABLES -->'
end = '<!-- END GENERATED TABLES -->'
text = report.read_text()
tables = []
for family in ('qwen', 'glimmer'):
    shutil.copyfile(work/family/'equivalence.json', root/'equivalence-results'/f'{family}.json')
    table = (work/family/'equivalence.md').read_text()
    table = table.replace('# '+family.title()+':', '### '+family.title()+':', 1)
    tables.append(table.replace('## Secondary:', '#### Secondary:'))
start = text.index(begin)+len(begin)
stop = text.index(end)
report.write_text(text[:start]+'\n'+'\n'.join(tables)+text[stop:])
# Only this task's selective copies are removed; preserve originals and evidence logs.
shutil.rmtree(work/'extracted')
PY_REPORT
```

For each new completed depth, verify its archive hash, add it to the extraction list and append its `rows.jsonl` path to the correct family command, supply witnessed start/end metadata with `--run-times`, then rerun that family and the report update. Do not change the four planned depths or reuse another model's result. JSON pending-depth fields and Holm values are recomputed automatically; update the archive ledger and descriptive cross-check for the newly admitted data. The original three-archive ledger and FACTS snapshot above identify this run's historical cut.
