# Spec-Dec-Opt — SGLang tree-policy porting notes

Working branch: `spec-dec-opt`. Goal: port the offline tree-drafting policies from
`ensemble_eval/pt_tree.py` (nucleus / controlled-expand / rerank / ensembles) into
SGLang's EAGLE path, to get real temp>0 rejection sampling + wall-clock speedup, and
to serve as the GPU-side reference for the Corsair split-deployment (see
`ensemble_eval/CORSAIR_IMPL.md`).

> Inspected SGLang at commit `3b1b512a9` (whatever HEAD was on clone). Line numbers
> are from that snapshot — re-grep if upstream moves.

## TL;DR (the two findings that matter)

1. **Your policies are pure Python/torch edits. No CUDA kernel needed** — including
   the "beat-eagle2" custom value function. The CUDA kernel (`sgl_build_tree_kernel_efficient`)
   only builds the tree *structure* (mask/positions/retrieve) from an **already-selected**
   set of nodes; it is policy-agnostic.
2. **SGLang's existing `adaptive_spec_params` does NOT do what nucleus does** — it only
   adjusts draft *depth* (`speculative_num_steps`) by an EMA of acceptance, and is
   **EAGLE/EAGLE3 + topk=1 only**. Width adaptation (nucleus) and global value rerank
   (`rerank-p`, beat-eagle2) are genuinely new, additive work.

## The pipeline & the two policy hooks (both in `python/sglang/srt/speculative/`)

EAGLE draft per block = N fixed steps of: draft forward → **EXPAND select** (per step)
→ … → **global RERANK to B** → build tree (CUDA) → verify → accept.

### Hook 1 — EXPAND (per-step beam): `spec_utils.py`
- `select_top_k_tokens` (`spec_utils.py:262`) dispatches to
  `_select_top_k_tokens_later` (`spec_utils.py:226`).
- The value is **cumulative PRODUCT probability**, linear domain:
  `expand_scores = scores.unsqueeze(2) * topk_p.view(-1, topk, topk)` (`:236`),
  then `fast_topk(expand_scores.flatten(1), topk)` (`:239`) keeps a **fixed top-`topk`**
  beam every step. This is EAGLE-2's fixed-width expand = our `greedy-w`/`fixedW`.
- **→ nucleus / controlled-expand goes here:** replace `fast_topk(..., topk)` with a
  cumulative-mass cut (cap at `topk`, pad — must stay fixed-shape, see constraints).
- ⚠️ `_select_top_k_tokens_later` is `@torch.compile(dynamic=True)` and `topk` is baked
  into CUDA graphs. Variable width must cap+pad to `topk` (exactly the `W`-cap+pad trick
  `pt_tree.py` already uses).

### Hook 2 — global RERANK (value function): `spec_utils.py`
- `organize_draft_results` (`spec_utils.py:77`): concatenates per-step scores into
  `score_list`, then **`top_scores = torch.topk(score_list, num_draft_token - 1)`**
  (`:85`) keeps the global best-B by score = EAGLE-2 global rerank.
- **The value function is THIS `torch.topk` on `score_list` — plain Python.** It is
  hardcoded as cumulative product prob, but trivially replaceable:
  - `rerank-p` (controlled expand + global rerank): just feed Hook-1's nucleus output
    through the same `torch.topk`.
  - **beat-eagle2** (expected-accepted-depth / calibrated P(accept) / decorrelation-
    folded value): transform `score_list` (or swap the `topk` objective) here. **No
    kernel.** Precompute your value as a tensor, hand it to `torch.topk`.

### The CUDA kernel is downstream and policy-agnostic
- `eagle_utils.py:113 build_tree_kernel_efficient` wraps `sgl_build_tree_kernel_efficient`
  (from `sgl_kernel`). It consumes `parent_list`, `top_scores_index`, `draft_tokens`
  (the already-selected nodes) and emits the tree mask / positions / retrieve_index.
  **You never touch it for a policy change.** Called from `eagle_worker_v2.py:550`.
- Verify + accept: `reject_sampling.py` (+ `eagle_info.py` EagleVerifyInput). Reused as-is;
  this is also where temp>0 rejection sampling already lives (the gap our offline harness
  couldn't test).

## Constraints discovered (the real gates, not kernels)
- **Static shape / CUDA graphs.** `topk`, `speculative_num_steps`, `num_draft_token` (=B)
  are fixed and graph-captured (`eagle_draft_cuda_graph_runner.py`). Keep variable *logic*,
  fixed *tensors* (cap+pad). This, not missing kernels, is the perf gate.
- **Scores are product-domain, not log.** SGLang multiplies probs (`scores * topk_p`).
  It gets away with no underflow because EAGLE depth is **shallow** (num_steps ~3–7).
  Our offline harness uses L=32 in **log domain** — if we push deep trees here, switch the
  accumulator to `Σ log p` (the CORSAIR_IMPL §4 numerics trap, now concretely relevant).
- **Existing adaptive is depth-only, topk=1-only** (`adaptive_spec_params.py:52,58`:
  rejects non-EAGLE and `eagle_topk != 1`). Don't build on top of it for width; it's a
  separate controller (turns spec depth up/down by load).
- **`eagle_worker_v2.py:193`**: "topk==1 only (select_top_k_tokens reorders rows,
  desyncing indices)" — there's a row-reorder subtlety to respect when editing select.

## Policy → edit map
| policy (pt_tree) | SGLang edit | kernel? |
|---|---|---|
| `greedy-w*` / `fixedW` | native (current `_select_top_k_tokens_later`) | no |
| `nucleus` / `greedy-p*` | Hook 1: mass-cut instead of `fast_topk(topk)` | no |
| `rerank-w16` (EAGLE-2) | native (current `organize_draft_results`) | no |
| `rerank-p*` (controlled) | Hook 1 (nucleus expand) + Hook 2 (existing topk) | no |
| beat-eagle2 value | Hook 2: transform `score_list` before `torch.topk` | no |
| `union` / `cascade` | run 2 draft forwards, merge candidate pools pre-Hook-1/2 | no* |

\* ensembles need orchestration in `eagle_worker_v2` (two draft streams), no kernel.

## How this helps the Corsair kernel-dev effort
The SGLang code is a **working, optimized reference for each of CORSAIR_IMPL.md's three
buckets** — the split deployment reuses the GPU side wholesale and mirrors the rest:
- **🟩 GPU verify is literally reusable.** SGLang's tree-mask verify + `reject_sampling.py`
  is exactly the Corsair deployment's GPU-side target verify (§C/§D, work-list #7/#8).
  The Corsair project does not need to write the verify; point dMatrix at this path.
- **🟨 Corsair-CTRL select/rerank/compaction has a reference.** `organize_draft_results`
  (sort/topk + gather) + `build_tree_kernel_efficient` (parent remap, mask rebuild) are a
  concrete spec for CORSAIR_IMPL §B2/§B3 (top-B + KV/parent compaction) — the control-core
  ops to port. Reading the CUDA kernel tells the Corsair team the *exact* data transform
  (parent_list → mask/positions/retrieve_index) they must reproduce on the sequencer.
- **The link payload matches.** SGLang ships `{draft_tokens, parent_list, scores}` into the
  tree-build; CORSAIR_IMPL §5's link is `{token ids, parent pointers}` — same contract, so
  the SGLang→kernel boundary is the natural Corsair↔GPU cut point.
- **Numerics lesson, confirmed.** SGLang's product-domain scores work only because trees are
  shallow. Corsair's deeper trees must use the log-domain accumulator — this is no longer
  hypothetical; it's the one place SGLang's approach would break if copied verbatim.
- **Value function is host/control-side, not silicon.** Because the rerank value is a Python
  `torch.topk`, the Corsair analogue lives on the **control core**, not the DIMC array — so
  "beat-eagle2" research (custom value) never requires a DIMC kernel on either platform.

## Independent draft + tree → use `STANDALONE` (NOT EAGLE)

EAGLE/EAGLE3 require a *trained head* that consumes the target's hidden states. For an
**independent draft model** (a normal small LM, e.g. Qwen 0.5B/1.5B drafting for Qwen-72B —
exactly the pt_tree / Corsair setup), SGLang has the **`STANDALONE`** speculative algorithm:
- `--speculative-algorithm STANDALONE`, draft = `--speculative-draft-model-path <small LM>`.
- `standalone_worker_v2.py`: `StandaloneDraftWorker(EagleDraftWorker)` /
  `StandaloneWorkerV2(EAGLEWorkerV2)` — **subclasses the EAGLE worker and reuses the whole
  tree machinery** (topk expand → `build_tree_kernel_efficient` → verify → accept). So you get
  **the tree on top of an independent draft for free**, and the tree policies (nucleus,
  rerank-p) added in `spec_utils.py`/`eagle_utils.py` apply to BOTH EAGLE and STANDALONE.
- `spec_info.py:130 carries_draft_hidden_states()` returns `is_eagle()` only → "STANDALONE's
  vanilla draft ignores [hidden states]" = genuinely independent.
- `StandaloneWorkerV2` does NOT override `draft()`/`verify()` → **the phase instrumentation
  below covers STANDALONE automatically.**
- Caveat: STANDALONE needs draft+target to **share a tokenizer/vocab** (standard spec decode).
  Cross-vocab (gpt-oss track) still needs the TLI port; same-tokenizer Qwen tracks work as-is.

Builtin algorithms: `EAGLE, EAGLE3, NEXTN, STANDALONE, NGRAM, DFLASH` (server_args.py:1488).

## Deliverables added this session
- **Bench:** `spec-dec-bench/src/sd_bench_sglang_offline.py` — SGLang sibling of the vLLM bench.
  Uses `sgl.Engine` offline; acceptance (accept_length==MAL, accept_rate, per-pos from
  `spec_correct_drafts_histogram`) comes free from per-request `meta_info`; throughput from
  wall time. Self-contained dataset loading (no vLLM). Sweeps num_steps×eagle_topk×num_draft_tokens.
- **Instrumentation (env-gated, zero overhead when off):**
  - `environ.py`: `SGLANG_DEBUG_SPEC_PHASE_TIMING` (EnvBool) + `SGLANG_DEBUG_SPEC_PHASE_TIMING_OUT`
    (EnvStr json path).
  - `speculative/spec_phase_timer.py`: CUDA-event `record(phase)` ctx mgr; folds totals;
    atexit dump to the json sink (worker is a subprocess, so file is how timing crosses back;
    rank-0 only).
  - `eagle_worker_v2.py` wraps 4 phases: `draft` (draft compute), `tree_build`
    (build_tree_kernel_efficient — the SELECT/rerank cost our policies change), `target_forward`
    (verify), `accept` (eagle_sample). Means are per verify step. Covers STANDALONE via inheritance.
- **Env:** `spec-dec-bench/requirements_sglang.txt` — editable local sglang (pulls torch 2.11.0,
  flashinfer[cu13] 0.6.12, sglang-kernel 0.4.4, transformers 5.8.1) + cu130 index + plotting libs.

## Install gotchas (env `sg`, editable)
- Editable build needs a **Rust toolchain** (current SGLang ships a Rust gRPC router):
  install rustup (`. "$HOME/.cargo/env"`) or you hit "can't find Rust compiler / build_rust".
- The Rust **gRPC router** (`sglang.srt.grpc._core`) build was failing (aho_corasick E0463) and
  is **not used by the offline `sgl.Engine`** (grpc._core is imported nowhere in the Python
  tree). Disabled it in `python/pyproject.toml` (commented the `[[tool.setuptools-rust.ext-modules]]`
  block, marked LOCAL DEV CHANGE) → editable install becomes pure-Python. Re-enable only if you
  need gRPC serving.

## Policy #1 implemented + validated: nucleus controlled-expand (rerank-p)
- Knob: `SGLANG_SPEC_EXPAND_P` (environ.py). 0 = native fixed-top-k expand; >0 = nucleus
  mass cut. Impl: `_select_top_k_tokens_later_nucleus` in `spec_utils.py` — zeros carried
  score of beams beyond mass `p` (capped at eagle_topk) so their children drop at the global
  rerank; this step's own nodes stay eligible leaves. Applies to EAGLE + STANDALONE.
- **Validated** (STANDALONE, Qwen2.5-1.5B target + 0.5B draft, humaneval, topk=8 steps=5 B=32):
  MAL responds monotonically to p — native 5.241 / p=0.5 5.023 / p=0.1 4.909 → wired through
  to the worker, narrowing reshapes the tree. (Startup logs "nucleus controlled-expand ACTIVE".)
- **Expected GPU behavior:** on fixed-shape CUDA graphs the beam width is constant, so zeroed
  beams still run the draft forward → draft compute is NOT saved (phase timing: draft ~108ms
  unchanged across p). So on GPU nucleus is a small MAL *cost* here. The controlled-expand WIN
  (≈same MAL at far less draft compute, pt_tree Finding 5) only materializes when draft compute
  is variable/cheap — i.e. `--disable-cuda-graph` (variable width) or the Corsair regime. The
  GPU value of nucleus is verify-budget reallocation at large B; measure with a proper sweep.
- Phase-timing instrumentation validated: draft ~108ms / target_forward ~92ms / tree_build
  ~0.9ms / accept ~1.2ms per 32 steps. Sink writes incrementally (atexit unreliable: worker
  killed by signal).

## Bench gotchas (resolved)
- Run with `conda activate sg` (or env bin on PATH) — SGLang JITs kernels via `ninja`.
- `meta_info["spec_correct_drafts_histogram"]` is a LIST (indexed by #correct), not a dict.

## Next steps
1. Stand up a baseline EAGLE run on the `sd`-sibling env (NOT `sd`) to get the golden
   acceptance/throughput numbers, temp>0.
2. Implement nucleus in Hook 1 (cap+pad), A/B vs native fixed-topk at matched B.
3. Implement `rerank-p` (Hook 1 + 2), reproduce the Finding-5 frontier on real wall-clock.
4. Prototype a beat-eagle2 value in Hook 2 (start: expected-accepted-depth).
5. Keep this file current as the discovery log for the branch.
