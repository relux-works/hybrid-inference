# Paired RULER: equivalence pre-registration

Registered on 2026-11-03 at 14:36Z by the hybrid lane orchestrator, before any 64k or 128k result exists. The
evaluator is `m3-ruler-eval` at `18d3f114`, unchanged; this file only declares the analysis.

## What was already observed

The 8k depth (complete at 07:32Z) and the 32k depth (complete at 14:21Z) were observed before this registration. Their
results are reported under the same rules below, marked "observed before registration". The 64k and 128k depths, and
the bf16 arm at 8k and 32k, are registered here before their data.

## Primary endpoint: task score

- Per example, the RULER score of the offload arm minus that of the local arm, on the same example.
- Estimate: the mean paired difference over all 390 examples of a depth (13 tasks × 30). Interval: a two-sided 95%
  interval for a paired difference of proportions (Newcombe's hybrid score method, method 10). For partial-credit
  tasks, use the paired bootstrap, 10,000 resamples, seed 261102.
- **Equivalence margin: ±2 percentage points.** Offload is called equivalent to local on score at a depth when the whole
  95% interval lies inside (-2, +2) pp. This is the same as TOST at α = 0.025 per side.
- Holm correction across the four depths for the equivalence claims.
- The same analysis is reported for the partition arm against local, as the reference for ordinary numerical noise.

## Secondary endpoint: answer text

This endpoint is unchanged from the runbook. It is the rate of changed answer text (outer-whitespace equality) against
local, and the excess of offload over partition, with a ±2 pp margin. A failure to establish text equivalence is
reported as such and is not turned into a quality claim.

## Reporting rules

- Report every depth, including any that fail the margin.
- A non-significant McNemar test is never reported as equivalence.
- The bf16 arm gets the same primary analysis against local and against the f32 offload arm.
