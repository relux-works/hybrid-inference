# E2E preregistration amendment r49c

Protocol: hybrid-e2e-amendment-r49c-v1. This signed amendment precedes all study inference and all owner selection. It supplements PREREG-E2E.md at dfffda35763edb17b832f758f586367999ac8978; that document stays unchanged. Every new artifact binds both signed commits and document hashes. The claim rule, five random session clusters, identical replay, waits, portion bins and no-reroll rule remain unchanged.

## Published inventory

Freeze and selection are separate commands. Publish the complete freeze receipt to agents1_hybrid_orch in #hybrid, topic prototype-go, before selection. It includes the inventory artifact hash, canonical digest, observed/eligible counts, every exclusion with its exception message, dates, source hash and registration bindings. Selection fetches the actual a2a event and verifies its sender, destination, topic and exact receipt content against the frozen inventory. A local boolean or an unsent file is insufficient. Publish only once per study ID; a changed inventory requires a separately declared study, never a replacement for this one. No owner export is currently available, so no owner distribution or selection is claimed.

## First full rehearsal and stages

The public series is mandatory first, on the target Mac, pinned target model and chosen fast path. Run both B/A and C/A (four arms for one public session), through the full supervisor, timed loop, capture, checker, accounting, transport and analysis gates. It measures actual checker and slot-save cost at about 200k before owner inference. Main and favourable execution require its verified passing receipt, all four raw traces, hashes and analyses, matching device/model/source and qualification method. This cannot be replaced by CPU mocks or a hand-authored ready flag. Preserve a failed rehearsal and its ID; it does not authorize a main run. Retokenize and freeze the public session with the target tokenizer first.

Stage 1: public rehearsal, both comparisons, fast link. Stage 2: five owner main sessions, C/A only, fast link, ten arms with a fresh local A per C and the unchanged A-C-A-C order and r47 cooldown. The fast path is the owner's iPhone USB, observed 150-223 Mbit/s to European hubs (18.75-27.875 MB/s before overhead); measure the actual node path. Stage 3 is an explicit budget decision before a new comparison: B/A and the home Wi-Fi series (observed 30-48 Mbit/s, 3.75-6 MB/s), plus favourable cases if authorized. They remain separate preregistered comparisons; no partial main comparison or outcome-based stopping supports a claim. A skipped comparison is reported as not run, never pooled. Gigabit paths require their own measured qualification and identity.

## Reduced quality schedule and capture budget

Only hybrid arms B and C capture and check quality. Each checker constructs a fresh local teacher-forced reference outside the timed schedule. Local A captures/checks no snapshots. The unchanged 64-position top-1 gate is >=63/64; lossy bf16 imports never satisfy exact controls. This removes duplicate local-reference checks. The coverage trade is two depths per hybrid instead of four depths per arm; there is no claim about unobserved depths or task success.

Freeze checkpoint placement before each series from the frozen recorded waits, not performance or quality. Targets are the session midpoint and session end. For the midpoint choose the nearest non-final turn whose following recorded wait is at least the declared capture budget; break ties toward the earlier turn. If none qualifies, declare final-only in the plan (do not exclude the session). Always capture at the final boundary outside warm system time. Deduplicate targets. Public capture budget is 30 seconds, conservatively above the cited 4-9 s slot-save observations. For owner series the budget is max(30, ceil(2 * maximum actual public slot-save seconds)); freeze it with the public receipt and series plan before execution. Slot saves must fit BOTH that budget and the single actual following wait. No extra wait follows a save. An overrun disqualifies the entire series, retains all evidence and permits no reroll. Checkpoint hashing and checkers run after the timed session. Sleep dispatch lateness is measured after sleep.

## Analysis and Mac budget

Emit every descriptive pair, paired gains and median/min/max for public (n=1) and favourable (n=3), without confidence or success claims. Only the five distinct main-session clusters use the unchanged bootstrap and claim rule. Missing/failed arms remain in the flow table; no automatic replacement. Portion bins use active turn time, while the primary warm endpoint includes the recorded waits.

Planning uses the FACTS PRI31 M3 depth fixture, import 1.498 s + 0.0487 ms/token, server surrogate 1097 tok/s, bf16 geometry, recorded public counts and waits, 300 s cooldown/arm. Cold and checker local fills use segment-rate integration scaled by 59.7/80.8 (249k FACTS cumulative), with the unscaled bracket retained. Two hybrid checkpoints require two fresh local reference fills plus 60 s/checkpoint. Setup/calibration adds 0.5-1 h/device once. These are estimates, not target measurements; a slow public checker triggers a budget decision before main, never a weakened gate.

Planning envelopes before the reproducible estimate: public four arms about 5-7 h, main C/A five pairs about 12-18 h for public-like short responses; recorded 512-token responses can add about 10-12 h per five pairs. Stage 3 adds approximately another five-pair budget per comparison/link; it is not pre-authorized as 72 arms. The delivered BUDGET.json and BUDGET.md refine these envelopes directly from the frozen public reference and FACTS rates, including 18.75 and 27.875 MB/s. Actual owner hours remain unknown until the unfiltered owner set exists. Record GPU service seconds, rental wall separately and bytes in every arm.

## Transport choice before study execution

Option (a), recommended: retain the original qualification unchanged. Rent a full Linux VM with root/CAP_NET_ADMIN and network namespaces (not a Vast container), Python3.10+, OpenSSL, iproute2/tc, git, at least 4 vCPU/8 GiB RAM/10 GiB disk; no GPU or models. Run the exact shipped shaped_transport.py in an isolated net namespace, all six 15/20 MB/s x 50/100/150 ms paths, each >=80%. Allow about one hour including provisioning. Preserve RESULTS.json, qdisc output and source identity. agents1 and Vast capability failures remain unqualified evidence. The runbook supplies the exact command. This option changes no transport qualification rule.

Option (b) is a conditional signed amendment, selected explicitly before any study inference. On the actual node-to-Mac path use the shipped eight-stream HTTPS transport T and independent parallel curl ranges against a separate nginx static HTTPS server R. They share neither gateway nor transfer implementation. Authenticate both with per-run private tokens, verify TLS, never log secrets. Each T and R trial transfers >=384 MiB; run T/R/T before AND after EACH comparison block, retaining raw outputs, interface identity and physical counters at every boundary. BOTH T rates must be >=80% of the adjacent R rate, on BOTH sides of the block. Failure disqualifies the series without reroll. Requalify on the fast path itself; a home pass cannot qualify fast/gigabit. Lock method and link identity in the plan, with no fallback after observing rates. This gives up the six absolute rate/RTT grid points and the known denominator; it retains an independent actual-path bottleneck comparison. Extra qualification traffic is 2.25 GiB per block (six trials) plus ordinary policy probes. The operator chooses (a) or (b); neither has been executed here.

## Reviewer's power note (unchanged)

The analysis implements the prereg exactly: primary endpoint = paired warm session system
time (post-cold wall including recorded waits, `warm_system_seconds`), all five paired
differences reported, median paired fractional reduction, session-cluster bootstrap
(10000 resamples of the five gains, seed 261105, percentile interval), B/A and C/A as
separate comparisons with separately measured local baselines, turns never treated as
independent for the claim, failures/no-offload sessions retained in the flow table, no
optional stopping. Claim: all gates qualified AND median reduction >= 10% AND lower 95%
bound > 0, per comparison, qualified by link/device.

Plainly, on the rule as written: **five sessions can meet it only if the effect is both
large and present in every sampled session.** For n=5 the bootstrap percentile interval of
the median is [min, max] of the five gains (probe P1, four gain sets, exact in each), so
"lower bound > 0" is operationally "the worst of the five random sessions still sped up",
and the median of those five must clear 10% of a denominator that includes unchanged decode
time and unchanged recorded waits. Reviewer power simulation (normal per-session gains,
20000 draws per cell, claim = min>0 and median>=10%):

| true mean gain | SD 5% | SD 10% | SD 15% | SD 20% |
|---|---|---|---|---|
| 20% | - | 0.87 | - | - |
| 15% | 0.97 | 0.63 | 0.38 | 0.26 |
| 10% | - | 0.29 | - | - |
| 5% | - | 0.06 | - | - |

Expected session-level variance is not small. Portion mix is the dominant term: in the
reviewer's FACTS-rate simulation of the public-shaped session (next section), the same
workload gives B/A = +0.0% on a 5 MB/s home link (zero of 68 turns offload - portions are
mostly <2k) and +52% at 50 MB/s. Random owner sessions will scatter between these regimes
by portion mix alone; FACTS also shows run-level spread on this Mac protocol (e.g. 32-stream
32k repeat 48.9 vs 57.4 tok/s, -15%; 128k 16-stream anchor -7.5% between paired runs; M5
128k np4 repeat +15%). A per-session gain SD of 10-20% on the home series is the realistic
planning range, and any session whose portions stay under the routing threshold lands at
gain ~0 or slightly negative (proxy/reconciliation overhead, no offloads) - which alone
fails "lower bound > 0" for that comparison. On top of the statistical rule, qualification
requires every gate to pass in every arm (quality >=63/64 at 80 main-series checkpoints,
admission, accounting); finding 2 shows the current schedule gate cannot pass at all, and
even after that fix the all-gates conjunct further reduces the chance of a qualified claim.

So: a home-link main claim is unlikely (order 5-30% under optimistic assumptions, and
structurally blocked for B if sampled portions rarely offload at 3.8-6.0 MB/s); a fast-link
claim is possible if the median sampled session is prefill-dominated enough that prefill
savings exceed 10% of warm time including decode and waits, and no sampled session is a
no-offload session. This is what the prereg buys: a negative or unqualified result is a
valid, publishable outcome of this design, and the portion bins still localize where a gain
exists. I do not recommend changing the preregistered rule; I recommend the orchestrator
budget Mac time expecting the descriptive/bin results to be the deliverable, with a
headline claim as upside.

