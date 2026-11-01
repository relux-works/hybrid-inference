# Qualification limits and open questions

- A node limits/LRU and B chunk transport/launcher are merged locally on `m1-integrated`. CPU integration evidence is in `RESULTS-hybrid-a-r11-261101.md`; 256k native state copying, memory amplification and target-GPU checker allocation still need qualification.
- Heterogeneous admission matches native layout and ordered vocabulary, not base-weight provenance. A local Q4_0-to-Q8_0 requantization tests compatibility, not increased model precision or production accuracy. Cross-quantization thresholds are uncalibrated.
- CUDA loading, CUDA cache offload, Metal client build, cross-backend native state portability and the 27B file geometry need real target machines. CPU evidence cannot answer these questions.
- Cross-backend section-11 thresholds need calibration on real target pairs. The benchmark reports current reference thresholds and explicitly marks that calibration absent.
- The synthetic ladder is one growing conversation per mode, not the section-11 50-prompt matrix, multi-client contention, realistic networks or a production security audit.
- The default proxy policy uses estimated prefill/link rates. Operator calibration is required before performance/economic conclusions. Report the active policy with any comparative analysis.
- The capacity table uses explicit packing assumptions, not downloaded model sizes. Actual GGUF metadata and available memory take precedence.
