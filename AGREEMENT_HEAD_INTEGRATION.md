# Agreement-head rerank in SGLang — status and remaining work

Started 2026-09-12. Goal: run the corsair agreement-head tree rerank inside the
`spec-dec-opt` branch so it can be evaluated end to end, in **tokens/sec**
rather than MAL. That is the number the project actually optimises and the one
no offline harness can produce.

## Why this is worth building

At an identical 32-node verify budget, reranking EAGLE-2's candidate pool by a
frozen target-agreement head instead of cumulative draft log-prob is worth
**+1.36 MAL** on held-out RepoBench, **+1.41** on GSM8K and **+1.04** on
OpenCodeInstruct (each with a head trained on that corpus, 2026-09-12).

It also settles the open cost question. The 2026-09-12 depth sweep showed the
policy's margin over a greedy chain is roughly **flat in tree depth L**
(~+2.0 to +2.9 at every L tested) while draft forwards per cycle fall from 310
at L=32 to ~143 at L=10. So accepted-tokens-per-draft-FLOP is *best* at L=10,
but whether that is profitable depends on the 1.7B-forward vs 235B-verify cost
ratio — measurable only in a real serving run.

## What is DONE and verified

### `python/sglang/srt/speculative/agreement_head.py`

- `AgreementScorer` — ported verbatim from
  `corsair_spec_selector.agreement_model` so existing checkpoints load unchanged.
- `build_agreement_features()` — **the single shared feature contract**. Both the
  serving path and the feature-dump path must call it, so parity is structural
  rather than something re-tested after every change.
- `agreement_log_probs()` — per-edge `log P(target accepts)`.
- `load_agreement_head()` — reads `scripts/train_agreement.py` checkpoints
  (`model_state` key; `embedding.weight` is deliberately omitted there, so the
  caller supplies the DRAFT model's embedding and no second 620 MB copy exists).
- `AgreementTreeScorer` — threads cumulative log-agreement alongside EAGLE's
  cumulative draft score.

### Tests (all three PASS)

`test_agreement_parity.py` — SGLang head vs corsair head, same checkpoint, same
random inputs. **max|diff| = 0.000e+00** on candidate ids, candidate logprobs,
entropy, top1/top2 gap, the 17-way logits, and the consumed per-edge log
agreement. Also asserts the none-of-K class is withheld rather than
renormalised: that residual mass is the head's only way to express "this gate is
risky", and renormalising would destroy the cross-depth signal.

`test_agreement_tree.py` — beam-index recovery checked against the REAL
`_select_top_k_tokens_later`, not a reimplementation. Recovered indices
reproduce EAGLE's own returned scores and token ids exactly (0.000e+00), the
agreement cumulative attaches to the same parent the draft beam picked, and
ancestor closure holds.

`test_agreement_alignment.py` — the two things `draft_forward` relies on that
neither of the above covers: that per-candidate scores follow the TOKEN rather
than the column when the head's top-16 and EAGLE's top-k disagree, and that the
draft pool's `-1e30` drop sentinels survive into the agreement pool. Both have
controls asserting the naive version would have been wrong.

## Key design decisions (do not re-litigate)

**Frontier stays draft-scored; only the global rerank changes.** Pruning the
beam by agreement instead ("afrontier") was measured as a null: -0.39% MAL at
budget 32, replicated on two independent runs. So the beam search is untouched.
This is why `AgreementTreeScorer` only has to *mirror* the beam choice EAGLE
already made, recovering it from
`tree_info[2] = topk_cs_index + (topk*topk*(i-1) + topk)`. No change is needed
to any of the three `@torch.compile`'d selection kernels.

**`organize_draft_results` needs zero edits.** It builds its global top-B over
`torch.cat(score_list)`. Passing `agree_score_list` instead switches the node
value function. Its depth-alpha ancestor-closure repair will never fire, because
cumulative log-agreement is a sum of non-positive terms.

**Log domain is mandatory** (`SGLANG_SPEC_ENABLE_LOG_DOMAIN=1`). Agreement
per-edge probabilities are much smaller than draft ones (mean log -0.448 vs
-0.190), so product-domain accumulation underflows fp32 substantially earlier.

## What the engine run settled (2026-09-12, second session)

All six items below were previously listed as remaining work. They are now
implemented, and the first live run of the integrated path validated the two
that could only be checked on a real engine.

Validation config: **Qwen3-14B target / Qwen3-1.7B draft, STANDALONE, topk=8,
steps=4, budget=32, GSM8K, greedy**. Deliberately not the deployment config --
it loads in a minute and exercises every code path. The head
(`agreement_main9000.pt`) is off-distribution here by construction (trained
against a 235B target on RepoBench), so this leg is about plumbing and cost, not
quality.

### 1. Root logits, recomputed from the hidden state — WORKS, and is checked

`_root_draft_logits` mirrors `LogitsProcessor._compute_lm_head` plus the TP
gather and the pad-vocab truncation, matching corsair's
`sglang_runtime._selected_tokens_from_hidden`.

`SGLANG_SPEC_AGREEMENT_DEBUG=1` logs, once per worker, the fraction of the
tree's children that the head actually ranked, per level. This is the diagnostic
that distinguishes "the head does not help" from "the root logits are wrong":
a bad hidden state or lm_head drives the root level to ~0 while the levels that
use the draft forward's own logits stay at 1. Observed:

```
L0=1.000, L1=1.000, L2=1.000, L3=1.000      # tp=1
L0=1.000, L1=1.000, L2=1.000, L3=1.000      # tp=2, both ranks
```

Every root child EAGLE expanded is inside the head's top-16 of the recomputed
logits, on both ranks. Note the Python only runs in eager mode, so read this
under `--disable-cuda-graph` (same constraint as `SGLANG_DEBUG_SPEC_TREE_DEPTH`).

### 2. A STANDALONE draft carries NO hidden states — this was the real work

**The blocker nobody had costed.** EAGLE drafts consume the target's hidden
state, so it is plumbed everywhere. A STANDALONE draft is a plain LM that takes
token ids, so SGLang switches hidden states off end-to-end — and the agreement
head's largest feature IS the gate's hidden state. Five sites gate on
`is_standalone()`, and all five had to flip:

| Site | What it gates |
|---|---|
| `eagle_utils.get_draft_recurrent_hidden_state_spec` | the draft-decode graph's `hidden_states` buffer |
| `base_spec_worker.prepare_for_draft` | `CaptureHiddenMode` for the draft decode |
| `base_spec_worker.prepare_for_draft_extend` | `CaptureHiddenMode` for the decode-path draft extend |
| `eagle_worker_v2._draft_extend_for_prefill` | `CaptureHiddenMode` for the prefill-path draft extend |
| `eagle_draft_cuda_graph_runner.capture_one_shape` | the captured `EagleDraftInput`'s capture mode |

They now go through one predicate, `eagle_utils.draft_carries_hidden_states`,
which is `not is_standalone() or a head is set`. For every non-STANDALONE
algorithm it returns exactly what the old expression did, so EAGLE/EAGLE3 are
untouched.

**The sixth site is the one that fails silently**, and is worth reading twice:
`spec_utils.spec_need_hidden_states` gates whether the FutureMap *relays*
hidden states across iterations. With it off, `EagleDraftInput.merge_batch`
takes the `self.hidden_states is not None and spec_info.hidden_states is not
None` branch and simply does not concatenate — so a two-request decode batch
arrives at `draft_forward` with `hidden_states` of shape `(1, 2048)` against
`topk_index` of shape `(2, 8)`. Here that raised (the head's feature stack has
mismatched rows). It did not have to: a shape that happened to broadcast would
have scored both roots from one request's hidden state and produced a plausible,
wrong MAL. This is the single most dangerous line in the integration.

### 3. Column alignment between the head and the tree — NOT an identity

The head ranks its own top-16 of the RAW draft logits; EAGLE expands the top-k
of `renorm_draft_probs`, a different tensor, and `speculative_eagle_topk` need
not equal the checkpoint's `top_k`. Column j of one is not column j of the
other. `align_log_agreement` matches by token id — the same thing corsair's
runtime does when it looks the emitted token up in its own top-k — and floors
anything the head did not rank to `log(MIN_AGREEMENT)`. Covered by
`test_agreement_alignment.py`, including a control asserting a naive column
slice would have been wrong.

### 4. The rerank inherits the draft pool's structural pruning

`select_top_k_tokens` prunes by writing `-1e30` into the score tensor
(`SGLANG_SPEC_EXPAND_CHILDREN`, the nucleus mass cut). Those decisions live only
in the draft scores, so swapping the pool wholesale would resurrect every branch
EAGLE deleted — changing the tree's *shape*, not just its ranking.
`AgreementTreeScorer._inherit_drops` re-stamps the sentinel, which keeps the
rerank a pure change of value function. The test includes a control showing that
without the mask the capped branches do come back.

### 5. CUDA graph — captures and replays with the head inside

`Capture draft decode CUDA graph end. elapsed=8.71 s, mem usage=2.40 GB`
(vs 1.86 GB without the head), and the captured run produces byte-identical
greedy output to the eager run. Nothing in the head forward is dynamic-shape.

### 6. Startup rejections — all fire

Verified by launching each: `topk == 1` (no tree to rerank),
`SGLANG_SPEC_EXPAND_P > 0` (nucleus), missing `SGLANG_SPEC_ENABLE_LOG_DOMAIN`,
`SGLANG_SPEC_DEPTH_ALPHA != 0` (fights the head for the same problem and breaks
the ancestor closure the head otherwise gives for free), a reduced draft vocab
(`hot_token_id`), `--enable-dp-lm-head`, final logit softcapping, and
`speculative_eagle_topk > head.top_k`.

### Cost, measured

| | baseline | with head |
|---|---|---|
| accept length | 4.7287 | 4.7018 |
| output tok/s | 658.2 | 653.3 |
| free GPU memory after startup | 9.37 GB | 7.67 GB |

Both differences are **under 1%, i.e. below the threshold at which runs on this
harness are comparable at all** — read them as "the head's serving overhead does
not show up at this scale", not as a measurement. The head is off-distribution
here, so the MAL is not a quality result either way.

The memory is real and explainable: ~1.16 GB is `AgreementScorer.__init__`
calling `embedding_weight.float()`, which materialises an fp32 copy of the
draft's 151936 x 2048 embedding, and ~0.54 GB is the head's `(b*topk, vocab)`
fp32 intermediates pinned in the graph pool. The first is avoidable and
bit-identical to remove — gather in bf16 and cast the (B, K, H) result instead of
casting the whole matrix, since gather-then-cast equals cast-then-gather
elementwise. Not done, because `AgreementScorer` is pinned by the parity test and
1.7 GB was not in the way.

## TP > 1 needs the embedding passed explicitly

Under TP the draft's in-memory embedding is vocab-sharded (`75968 rows < vocab
151936` at tp=2), so `--speculative-agreement-embedding` must point at the saved
full matrix — `corsair-spec-selector/artifacts/qwen3_17b_embedding.pt` is the
one the head was trained with. Gathering the shards here would mean reproducing
`VocabParallelEmbedding`'s padding scheme, which is not worth it when a frozen
copy is already on disk. The error message says exactly this.

The recomputed root logits DO gather correctly under TP (the coverage diagnostic
reads 1.000 on both ranks at tp=2), via `tensor_model_parallel_all_gather`.

## Remaining work

1. ~~A deployment-config leg: 235B-GPTQ-Int4 at tp=4.~~ **Done 2026-09-13**, and
   the memory envelope did bite. At `--mem-fraction-static 0.85` the run OOMs
   during *target prefill* graph capture ("tried to allocate 480 MiB, 73 MiB
   free"). **0.76 works**; use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   as well. This is the config where the head's avoidable 1.16 GB fp32 embedding
   copy is worth removing -- at 14B it was 2% of the card, here it is a real
   share of the headroom left after 29.5 GB/GPU of int4 weights.
2. **A corpus-matched head**, which is the retraining below — collected with the
   EAGLE left-shift applied (see the parity section; this is a correction to the
   recipe as originally written). Until then any MAL number from this path is a
   number about a head that is off-distribution in two independent ways.
3. **PD-disaggregation is not covered.** `SpeculativeAlgorithm.
   carries_draft_hidden_states` (the prefill->decode transfer) is still
   `is_eagle()`-gated and was deliberately left alone; STANDALONE + head + disagg
   would drop the draft hidden states in transfer. Guard or fix it before anyone
   tries that combination.

## Serving-vs-collection feature parity, MEASURED (2026-09-13)

Run before committing to a retrain, on the reasoning that a retrain papers over
a feature skew rather than identifying it. It found one that a retrain on the
existing pipeline would have baked in.

**Method.** `SGLANG_SPEC_AGREEMENT_DUMP=<path>` writes the features the head
actually consumed on the first `draft_forward` (all four levels), produced by the
same `build_agreement_features` the head consumes, so this is a diff and not a
re-derivation. The collection leg reproduces `sglang_runtime.SGLangRuntime`'s
rollout path — plain Qwen3-1.7B, `return_hidden_states`, triton attention,
lm_head from the saved tied embedding — on the *identical* token sequence, taken
from `data/canonical/reason_eval_gsm8k.jsonl` so there is no re-tokenisation.
Scripts: `corsair-spec-selector/scripts/parity_serving_leg.py` and
`parity_collection_leg.py` (the latter takes `--eagle-shift` to reproduce the
rotation described below).

### THE finding: the draft's input is left-shifted, and collection never was

`_eagle_prefill_tail_tokens` rotates the draft-extend input: the draft consumes
**`prompt[1:] + [bonus]`**, not `prompt + [bonus]`. It never sees `prompt[0]`.
That is the EAGLE convention and STANDALONE inherits it.

corsair's collection fed the draft the **full** prefix. So every hidden state in
the training set was computed from a sequence the draft does not see at serving
time. Measured at the root gate of a 77-token prompt:

| collection variant | hidden rel-RMS vs serving | cosine | candidate logprob max\|diff\| |
|---|---|---|---|
| full prefix (what training used) | 6.70e-2 | 0.99776 | 0.6875 |
| EAGLE-shifted (`prompt[1:]`) | 2.51e-2 | 0.99969 | 0.1250 |

And in the consumed quantity — `log P(target accepts)` on the **top-8 edges the
tree actually expands**, matched by token id, across the root gate and the three
gates down the greedy chain:

| gate | full-prefix collection (mean / max) | EAGLE-shifted (mean / max) |
|---|---|---|
| 0 (root) | 0.086 / 0.258 | 0.023 / 0.044 |
| 1 | 0.147 / 0.524 | 0.031 / 0.053 |
| 2 | 0.048 / 0.079 | 0.017 / 0.041 |
| 3 | 0.050 / 0.152 | 0.026 / 0.075 |

**The scale that makes these numbers mean something: the draft overconfidence
the head exists to correct is ~0.26 nats per edge.** So collecting on the
unshifted prefix injects a skew worth 20-60% of the entire signal the head
carries, and it is systematic rather than noise. The shift reduces it to ~10%.

Match by token id, not by column, when repeating this. Comparing column j to
column j reports ~1.0 nats mean, which is an artefact: the two legs order the
far tail differently, and those candidates sit below -15 nats of draft logprob
(p < 3e-7) where the head's residual is meaningless.

### The non-finding: `context_length` was off by one, and it did not matter

Serving reported 77 where the collection path recorded 78 for the same gate.
Worth 1e-4 nats. It is also not actually an error once the shift above is
understood: `seq_lens` is the length of the sequence **the draft attended to**,
which is the right thing for a feature about the draft, and it is what a
shift-correct collection will record. Left as `seq_lens`, with the reasoning in
the code so nobody "fixes" it again.

This is the useful shape of the result: the off-by-one was the visible
discrepancy and was worth 1e-4 nats; the invisible one, in the draft's input
sequence, was worth 250x more.

### What is left, and why it is probably the floor

The EAGLE-shifted collection still differs from serving by ~0.025 nats/edge.
That is engine-level: triton vs flashinfer attention, a different prefill shape,
bf16 accumulation order. It cannot be removed by changing what tokens are fed —
only by collecting through the serving path itself.

**Caveat, stated plainly: n=1 prompt, 4 gates.** The *existence and direction* of
the shift is structural and certain (it is a line of code, not a measurement).
The magnitudes are from one prompt and should be re-measured across a sample
before anyone quotes them.

## Retraining on SGLang-produced features

Correct, and better than a workaround: a head calibrated on one feature
distribution and applied to another is miscalibrated, and miscalibration is
**worse than no head** — the RepoBench head scores -0.36 vs draft rerank on
GSM8K and -2.25 on MATH-500.

But retraining alone only *relocates* the risk. The invariant to enforce is that
collection, training and serving all call `build_agreement_features()`. Then
parity cannot be false rather than merely being tested.

Cheapest path: collect with SGLang's draft **teacher-forced along the canonicals
already generated on 2026-09-12** (`data/canonical/{reason,oci,bigcodebench}*.jsonl`
in corsair). Same feature code as serving, but a draft-only pass instead of full
235B serving, and the labels are already on disk.

**Corrected 2026-09-13 — this recipe is wrong as originally written.** It has to
feed the draft `prompt[1:] + [bonus]`, the rotation `_eagle_prefill_tail_tokens`
applies, not the plain prefix. Collecting on the plain prefix is what the
existing training set did, and the parity section above measures it at
0.05-0.15 nats/edge of systematic skew against a ~0.26 nats/edge signal. A
retrain on unshifted features would move the skew from "head trained on the
wrong corpus" to "head trained on the wrong sequence" and look like it had
fixed something.

Note the head is applied at inference to speculative prefixes, while training
labels exist only at canonical prefixes. That is not a defect: the head
estimates "if this prefix were accepted, would the target emit this token",
which is the correct conditional for path survival, and the corsair harness
already operates this way.

## Entropy cost, if profiling flags it

`build_agreement_features` does one `(B*topk, vocab)` reduction per step for
entropy. If that shows up, retrain without the entropy feature — it is 1 of 10
features, retraining takes minutes, and the offline quality cost can be measured
before committing.
