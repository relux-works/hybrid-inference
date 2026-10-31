# Node state questions

The implemented configurable limits and idle LRU behavior have no pending product decision. The resident/host/disk architecture remains design-only; its open choices are listed in [NODE-STATE-DESIGN.md](NODE-STATE-DESIGN.md): pool capacity classes, fairness and all-pinned policy, retry/rollback acknowledgement, disk encryption and ownership, quotas/TTL, native-state migration, and calibrated CUDA/Metal thresholds. Resolve these with target-model measurements before tiering implementation.
