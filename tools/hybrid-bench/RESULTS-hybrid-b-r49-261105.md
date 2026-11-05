# R49 CPU session measurement prototype

Candidate parent: f9af64820d2a615dd282fea869cba4aaba742616 on m3-growing-r46. CPU only, existing Qwen3.5-0.8B-Q4_0 GGUF; no model downloads, GPU, Metal or target-model inference. Delivery is ready for review after the recorded CPU checks; production timing admission remains closed pending the owner export and shaped transport qualification. The previous forty-turn measurement remains a slowdown, not evidence of a session speedup.

## Delivered contract

Read packets/r49/DESIGN.md and RUNBOOK.md. Eight independent certificate-verified HTTPS range connections replace the single-stream payload path in the new opt-in entrypoint. Per-run credentials stay in private files, immutable envelopes are owner-scoped and bounded, and corrupt/short ranges refuse before import. TLS counters measure encrypted TCP payload, not retransmission or physical-interface bytes. The default accepted proxy and r46 session driver remain available unchanged in behavior.

The new session driver has local, bf16 hybrid with on-demand catch-up, and bf16 hybrid-background arms. f32 control labels remain distinct. A bounded worker only sends the durable evaluated journal, validates node prefix acknowledgements and preserves newer foreground commits. Foreground commits also preserve newer worker acknowledgements. Routing prices matching bf16 geometry and pending catch-up. Failed/missing node-time accounting cannot qualify silently. Native node compute clocks are wall proxies, not CUDA kernel counters.

The owner selector accepts only a pinned ordered export, chooses the first qualifying contiguous forty turns without consulting routing, and preserves incoming tokens, answer-length budgets, pauses and seeds across arms. The actual generated IDs remain arm-specific. Snapshots and artifact hashes are post-turn; fresh-reference quality checks run after all timed decoders stop, with identical 64 teacher inputs on the actual checkpoint journal. >=63/64 remains the gate; bf16 remains explicitly lossy/exit 5. The r46 ten exact controls and Qwen equality path are retained.

## Validation and boundaries

Evidence root: /home/agent/Werkstatt/hybrid-b/llama.cpp/.temp/hybrid-b-r49-261105. Logs preserve unsuccessful attempts, command exit codes and corrected runs. Commands use on-disk TMPDIR and PYTHONDONTWRITEBYTECODE=1. Native builds use nice 19, -j4 and the shared model-lane build lock after compiler/load/disk admission. No concurrent lane build was started.

| Check | Result and evidence |
|---|---|
| Benchmark Python suite | 166 tests, exit 0, seven explicit optional-fixture skips; bench-suite-final2.log |
| Proxy Python suite | 141 tests, exit 0, no skips; proxy-suite-final.log |
| Real TLS controls | Wrong token, untrusted certificate, ownership/range/TTL/size, short range, digest mismatch, opaque f32 identity and encrypted-byte accounting; test_parallel_transport.py |
| CPU native mechanics | native-04 exit 0: all three arms, four controlled turns plus cold per arm, real bf16 imports/background ACK injection, quality 64/64, mean KL 0.0000677591385, free divergence -1 within 64, lossy checker exit 5 |
| Native f32 state controls | test_prefill.py exit 0: f32 checksum/native snapshot identity, invalid boundary/token/model/ownership refusal, interleaved clients/sessions; prefill-controls-01.log |
| Existing exact/forcing controls | Checker --test-append-controls exit 0, --test-teacher-forcing on existing weights exit 0; append-controls-01.log and teacher-control-01.log |
| Import fitter | Delivered r44 NNLS helper --self-test exit 0; fitter-tests-01.log |
| Actual kernel shaping | Exit 1 at unshare uid_map setup, shaped/BLOCKED.json and namespace.log; no measured throughput and no substitute unshaped claim |

The benchmark skips cover optional native-header/no_alloc fixtures and SWA identity tests without their required local fixture. They are skipped, not green inference evidence. ty and editorconfig-checker are unavailable (exit 127). Python AST, shell syntax, diff whitespace and unchanged pure f32 codec/import source checks are the local substitutes for the applicable static scope, not equivalents to unavailable tools.

CMake was absent (first configure exit 127). A task-local pinned CMake 3.31.6 wheel was installed with python3 -m pip install --target .temp/r49/cmake --no-cache-dir --no-compile --only-binary=:all: --no-deps cmake==3.31.6. This installs only a build tool, not models. Configure flags: -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_METAL=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF. Targets: llama-server, llama-hybrid-prefill, llama-hybrid-bench-check. Build and tool directories are removed after retaining identities.

The initial native test failed because n_predict=0 emits one unused sample; cold handling now records/discards it and verifies the evaluated state is the original base. An initial journal fixture failed because Store.save intentionally does not persist slot binding; production and tests use set_slot. Both failures remain recorded. Self-review found and fixed the foreground stale-ACK race, with a deterministic regression and final native injection. TLS handshakes run in bounded request workers so a stalled handshake cannot serialize acceptance of the parallel range connections.

## Mutation controls

Every mutation below exits 1 at the named contract test; the handshake mutant times out the valid request while the planted peer withholds its handshake. None fails during setup. The nearby unmutated positive passes. No survivors are claimed.

| Mutant | Gate narrowed to | Named failing test |
|---|---|---|
| Remove committed-prefix bound | Allows a node ahead of the client's durable journal | test_background_catchup_never_runs_ahead_named_negative |
| Replace divergent teacher input before validation | Accepts unpaired checkpoint inputs | test_quality_teacher_pairing_and_lossy_negative |
| Run evidence before publishing timed completion | Allows blocked evidence to delay the measured result | test_evidence_never_blocks_a_timed_turn |
| Omit pending catch-up price | Routes as if stale node catch-up were free | test_pending_catchup_changes_route_and_is_not_free |
| Save stale foreground remote fields | Erases a newer background acknowledgement | test_foreground_commit_cannot_erase_background_ack |
| Serialize TLS handshake in accept loop | Allows a stalled peer to prevent valid parallel connections | test_stalled_handshake_does_not_serialize_connections |

No brief surface table was supplied. No requested contract is silently excluded. The kernel-shaped target, actual owner-log selection/estimates, CUDA/Metal/27B quality and rented-node/Mac execution are explicitly unverified. There are no new SWA claims. Changes outside tools/hybrid-bench are necessary transport integration (proxy.py, parallel_transport.py), atomic acknowledgement persistence (store.py), and two cheap native compute clocks (prefill.cpp). Pure hybrid-state.cpp/.h, llama-context.cpp and llama-kv-cache.cpp match the accepted parent.

## Remaining prerequisites and effort

The owner export plus trusted full hash is required before trace-specific traffic/time estimates can exist. An isolated authorized Linux runner must execute the six-row tc/netem matrix and establish >=80% at every path; agents1 cannot grant itself CAP_NET_ADMIN or alter the shared host qdisc. Allow roughly half a day after both inputs arrive for selection/planning, qualification and packet review; transport tuning is additional if a measured path misses target. No GPU time is authorized by a passing localhost mechanics test. Reserve a day per target Mac pending actual planning, with deferred local-reference checkpoints potentially taking several hours.
