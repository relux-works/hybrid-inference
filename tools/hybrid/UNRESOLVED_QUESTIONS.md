# Node state questions

The implemented configurable limits and idle LRU behavior have no pending product decision. The resident/host/disk architecture remains design-only; its open choices are listed in [NODE-STATE-DESIGN.md](NODE-STATE-DESIGN.md): pool capacity classes, fairness and all-pinned policy, retry/rollback acknowledgement, disk encryption and ownership, quotas/TTL, native-state migration, and calibrated CUDA/Metal thresholds. Resolve these with target-model measurements before tiering implementation.

SWA qualification: no Gemma 4 or Muse Glimmer GGUF is installed on this host, so real-model CPU/GPU logits and task quality remain unverified. Run the rented-GPU commands in `../hybrid-bench/RESULTS-hybrid-a-r16-261101.md` after keeper approval for model provisioning. Same-partition logit calibration requires the separate PR 9. No new design decision is pending for the implemented standard-window format.
