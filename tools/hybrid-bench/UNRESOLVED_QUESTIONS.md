# Qualification limits and open questions

- The inherited 16,384 context and 512 MiB retained-state/delta limits prevent long-depth 27B qualification even when GPU memory is sufficient. Extending the RAM store, bounded chunk transport and decoder import resource model is a separate design/review unit.
- CUDA loading, CUDA cache offload, Metal client build, cross-backend native state portability and the 27B file geometry need real target machines. CPU evidence cannot answer these questions.
- Cross-backend section-11 thresholds need calibration on real target pairs. The benchmark reports current reference thresholds and explicitly marks that calibration absent.
- The synthetic ladder is one growing conversation per mode, not the section-11 50-prompt matrix, multi-client contention, realistic networks or a production security audit.
- The default proxy policy uses estimated prefill/link rates. Operator calibration is required before performance/economic conclusions. Report the active policy with any comparative analysis.
- The capacity table uses explicit packing assumptions, not downloaded model sizes. Actual GGUF metadata and available memory take precedence.
