# Unresolved questions

- Hybrid r19: Gemma 4 drops the bench's tool response without a preceding matching assistant tool call. The smallest proposed change is a matching assistant call using the same tool_call_id, retaining the response payload sizes and recording framing overhead separately. The user requested a proposal only; the workload is unchanged. See tools/hybrid-proxy/RESULTS-hybrid-a-r19-templates-261101.md for evidence and token-position implications.
- Real GPU qualification remains pending for Gemma 4 and Muse Glimmer: emitted stop tokens, complete sampled IDs, native checkpoint rollback, transfer identity and same-partition numerical equivalence must be measured on the actual host. CPU fixture replay establishes byte/token/frame contracts only.
