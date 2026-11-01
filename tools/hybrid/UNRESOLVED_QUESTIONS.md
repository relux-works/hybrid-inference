# Node state questions

The implemented configurable limits and idle LRU behavior have no pending product decision. The resident/host/disk architecture remains design-only; its open choices are listed in [NODE-STATE-DESIGN.md](NODE-STATE-DESIGN.md): pool capacity classes, fairness and all-pinned policy, retry/rollback acknowledgement, disk encryption and ownership, quotas/TTL, native-state migration, and calibrated CUDA/Metal thresholds. Resolve these with target-model measurements before tiering implementation.

Import-speed r35: macOS must build the rebased head and pass auto/CommonCrypto and forced-portable codec/hash suites before landing. The [r35 report](RESULTS-hybrid-b-r34-261101.md#r35-rebase-after-pr-7) gives exact commands; agents1 CPU results do not close this platform gate.
