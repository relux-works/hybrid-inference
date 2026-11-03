# Preregistered RULER equivalence: hybrid-d-r16-261103

Analysis on agents1, 2026-11-03. Base and contract: `m3-ruler-prereg` at `7556f46165b01216ff1028c381845c8e79c65955`, [preregistration](RULER-EQUIVALENCE-PREREGISTRATION.md). This is a local analysis on the private prototype fork, with no publication or GPU execution. RC4 data cut-off: 2026-11-05 12:00Z.

## Analysis rule and interpretation

The interval choice was recorded in `.temp/tasks.md` before calculating results: use Newcombe method 10 if every score in both compared arms is 0 or 1; otherwise bootstrap the entire set, including the binary tasks. No thresholding of partial credit or averaging of task intervals is used. The endpoint is the equally weighted mean of all 390 paired example scores, left arm minus right arm. Each completed depth must contain exactly 13 specified tasks x indices 0-29, shared prompts and seed 261102. Duplicate, missing, mismatched, nonfinite or inconsistent rows refuse. CSV input requires companion JSONL and manifest, because the evaluator CSV does not contain seeds or prompt identities. Explicit per-arm seeds, when present, must also match. The original evaluator records one shared seed per paired row.

[Newcombe (1998), section 5, method 10 and Table III, pp. 2641-2642](https://www.site.uottawa.ca/~nat/Courses/csi5388/Newcombe.1998.pdf) supplies the binary interval and the published test values. Method 10 corrects positive paired phi by subtracting n/2 from its numerator, clipped at zero; it leaves negative phi unchanged and uses phi=0 for a zero denominator. The marginal Wilson intervals have no continuity correction. Tests include nonzero discordance and identical-arm boundary cases.

The partial-credit interval is the paired percentile bootstrap: sort examples by task and index, resample all 390 paired differences with replacement, unstratified, 10,000 resamples using Python `random.Random(261102)`, and linearly interpolate quantiles at positions `(B-1)*q` (type 7). Each pair restarts that seed. The same bootstrap draws are reused for every confidence level during inversion.

TOST p is the smallest one-sided tail alpha for which the central `1-2*alpha` interval is strictly inside (-0.02,+0.02). The two one-sided tests use their maximum p, without doubling; acceptance is strict p < 0.025. Bisection uses a lower numerical bracket of 1e-15. If no interval with alpha <=0.5 certifies the margin, report conservative p=1. The JSON retains the raw inverse. Bootstrap p-values below its finite tail resolution are conservatively reported as 1/10000, flagged `tost_p_resolution_limited`, and that reported value enters Holm. This is a reporting bound, not evidence of a precisely measured tiny tail probability.

Each model and arm pair has its own four-depth Holm family at 0.025. Missing depths enter the ordered family as p=1; they are neither omitted nor treated as successful. All decisions remain provisional until that pair's four depths are available. Qwen expects 8192, 32768, 65536, 131072; Glimmer expects 8192, 32768, 65536, 122880. The families are not pooled. No further cross-model, cross-arm or task adjustment was registered. No bf16 observations were present in these archives; bf16-local and bf16-offload remain pending. Supported future bf16 input has an explicit `results.bf16` on all 390 matched rows; f32 data cannot silently stand in for it.

Qwen 8k contains partial-credit scores but all paired differences are exactly zero. The literal rule therefore yields a degenerate [0,0] bootstrap interval and a resolution-limited p. This conditional empirical bootstrap cannot bound discordance absent from the observed sample; it is not proof of exact population equality. Qwen 32k has only binary scores and nonzero method-10 interval widths. Glimmer 8k has partial credit and upper limits above +2 pp, so equivalence is not established. The 8k and 32k rows are labelled "observed before registration" as requested; they must not be presented as unseen confirmatory evidence. The later depths and bf16 arms are registered before data.

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
Holm is separate for each arm pair across four depths. Pending depths enter as p=1; decisions are provisional.

| Depth | Pair | Difference | 95% interval | Method | TOST p | Holm p | Equivalent (raw / Holm) | Registration |
|---:|---|---:|---|---|---:|---:|---|---|
| 8192 | local_partition-local | +0.0000 | [+0.0000, +0.0000] | paired_percentile_bootstrap | 0.0001 | 0.0004 | True / True (provisional) | observed before registration |
| 8192 | offload-local | +0.0000 | [+0.0000, +0.0000] | paired_percentile_bootstrap | 0.0001 | 0.0004 | True / True (provisional) | observed before registration |
| 32768 | local_partition-local | -0.2564 | [-1.4704, +0.8517] | newcombe10 | 0.00725632 | 0.021769 | True / True (provisional) | observed before registration |
| 32768 | offload-local | +0.0000 | [-1.0519, +1.0519] | newcombe10 | 0.00255609 | 0.00766826 | True / True (provisional) | observed before registration |

#### Secondary: answer text (unchanged ruler_analysis.py)

| Depth | Offload changed / n | Partition changed / n | McNemar p | Excess 95% (pp) | Text equivalent +/-2 pp |
|---:|---:|---:|---:|---|---|
| 8192 | 34/390 | 32/390 | 0.726562 | [-2.1275, +3.1189] | False |
| 32768 | 33/390 | 29/390 | 0.617719 | [-4.0224, +6.0473] | False |

### Glimmer: paired RULER equivalence

Differences and intervals are percentage points. Margin: +/-2 pp; alpha per side: 0.025.
Holm is separate for each arm pair across four depths. Pending depths enter as p=1; decisions are provisional.

| Depth | Pair | Difference | 95% interval | Method | TOST p | Holm p | Equivalent (raw / Holm) | Registration |
|---:|---|---:|---|---|---:|---:|---|---|
| 8192 | local_partition-local | +1.1667 | [-0.1923, +2.6410] | paired_percentile_bootstrap | 0.123712 | 0.494849 | False / False (provisional) | observed before registration |
| 8192 | offload-local | +1.2179 | [-0.1410, +2.6923] | paired_percentile_bootstrap | 0.141114 | 0.564456 | False / False (provisional) | observed before registration |

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

## Validation and bounds

All commands ran on CPU with `PYTHONDONTWRITEBYTECODE=1` and disk-backed `TMPDIR=.temp/hybrid-d-r16-261103/tmp` for temporary-file tests. No packages, data or models were downloaded.

| Check | Real exit | Evidence |
|---|---:|---|
| New equivalence CPU suite, 19 tests | 0 | `.temp/hybrid-d-r16-261103/unit-05.log` |
| Existing evaluator suite, 11 tests | 0 | `evaluator-regression-01.log` in the same directory |
| Existing paired analysis and archive regression tests, 2 tests | 0 | `secondary-regression-01.log` |
| Three archive SHA-256 and internal member hash checks | 0 | `archive-verification-01.log` |
| Both model-family CLI analyses on the delivered source | 0 | `regeneration-02.log` (the exact commands below) |
| FACTS totals/tasks and unchanged archived secondary tables | 0 | `facts-comparison-01.log` |
| Real archive CSV and JSONL endpoint equality | 0 | `csv-analysis-01.log`, `pre-regeneration-01.log` |
| Exact commands below regenerate tables and JSON, then remove extracts | 0 | `regeneration-02.log`, `regeneration-comparison-02.log`; prior byte-for-byte replay in `regeneration-comparison-01.log` |
| Python syntax and scoped ASCII/whitespace checks | 0 | `syntax-style-01.log` |

The first unit run exited 1: binary floating scores reached an existing truth-table helper without conversion to bool and produced floating cell counts. Explicit boolean conversion fixed this; the published references and subsequent positive/negative suites pass. This failed run is retained as `unit-01.log`.

Four bounded mutants were run only in `.temp/`; the control runner exited 0 only after verifying each named test exited 1 with the intended assertion failure (not an import failure or crash). No mutants survived.

| Mutant | Property narrowed by the control | Named failing test | Real test exit |
|---|---|---|---:|
| Omit positive phi correction | Method 10 rather than method 8 | `test_newcombe10_published_reference_cells` | 1 |
| Accept every interval | Strict margin containment | `test_interval_crossing_margin_is_not_equivalent` | 1 |
| Omit row seed comparison | Refuse row/manifest seed mismatch | `test_changed_seed_refuses` | 1 |
| Ignore Holm ordering | Sort, multiply and restore original order | `test_holm_ordering_and_pending_depth` | 1 |

This brief supplies no attack-surface table; that is an explicit brief gap. The implemented input, statistical and reporting contracts are covered by named positive/negative tests, without a claim of exhaustive coverage. No rows in the requested endpoint analysis were treated as out of contract. Inference quality, future depths, bf16 measurements and hardware qualification are outside these CPU checks. No native C++ build was run because no native code changed. `ty` and `editorconfig-checker` are unavailable; their hosted commands were not run and no install was attempted under the no-download rule. Python syntax and scoped whitespace/ASCII checks are run locally. There is no active task-board in this workbench (`goal get` reports missing `.task-board`); scope and progress are tracked in the existing `.temp/tasks.md` flow. No external review, remote CI, push or landing is claimed or requested for this local delivery.

## Regeneration

Run from the repository root. The original archives stay in place. This verifies full hashes and internal hashes before extracting only the small analysis inputs. It refuses duplicate archive members and oversized archives. It writes to the task's own `.temp/` directory.

```sh
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/.temp/hybrid-d-r16-261103/tmp"
mkdir -p "$TMPDIR"
python3 - <<'PY_EXTRACT'
import hashlib
from pathlib import Path
import tarfile

source = Path('/home/agent/lanes/hybrid/.temp/ruler-results-261103')
work = Path('.temp/hybrid-d-r16-261103')
archives = [
    ('qwen-8192', 'depth-8192-complete.tar.gz', '2be3eae64b783867258a1f22e1c7bbdb4188f257c378f17c07554a8868248aa4'),
    ('qwen-32768', 'depth-32768-complete.tar.gz', 'a6fb1e7e6ac0fb7d25ed3ed24ef1f47cdf4f6495b54a4866d6486870bfeef176'),
    ('glimmer-8192', 'glimmer-715c81a4/results-8192.tar.gz', '2ff122d17f892dcb6f495306e06d2fb1671e0b145523df18605c58a3b6bfe8f1'),
]
for label, name, expected in archives:
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
    .temp/hybrid-d-r16-261103/extracted/qwen-8192/rows.jsonl \
    .temp/hybrid-d-r16-261103/extracted/qwen-32768/rows.jsonl \
    --family Qwen --output .temp/hybrid-d-r16-261103/qwen
python3 tools/hybrid-eval/ruler_equivalence.py \
    .temp/hybrid-d-r16-261103/extracted/glimmer-8192/rows.jsonl \
    --family Glimmer --depths 8192 32768 65536 122880 \
    --output .temp/hybrid-d-r16-261103/glimmer
python3 -m unittest discover -s tools/hybrid-eval -p test_ruler_equivalence.py -v
PYTHONPATH=tools/hybrid-eval python3 -m unittest \
    test_ruler.RulerTests.test_known_paired_table_and_intervals \
    test_ruler.RulerTests.test_complete_archive_and_member_hashes
python3 -m unittest discover -s tools/hybrid-eval -p test_eval.py
python3 - <<'PY_REPORT'
from pathlib import Path
import shutil

work = Path('.temp/hybrid-d-r16-261103')
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

For each new completed depth, verify its archive hash, add it to the extraction list and append its `rows.jsonl` path to the correct family command, then rerun that family and the report update. Do not change the four planned depths or reuse another model's result. JSON pending-depth fields and Holm values are recomputed automatically; update the archive ledger and descriptive cross-check for the newly admitted data. The original three-archive ledger and FACTS snapshot above identify this run's historical cut.
