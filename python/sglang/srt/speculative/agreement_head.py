# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Frozen target-agreement head used as an EAGLE tree node-value function.

WHAT IT IS. A 525K-parameter 17-way classifier over the draft's top-16 next
tokens plus an explicit none-of-K class, trained with per-gate cross-entropy
against the TARGET model's actual next token at each canonical prefix. Because
cross-entropy is a proper scoring rule, the head is CALIBRATED: its output is an
estimate of P(target emits this token | prefix), not merely a ranking.

WHY THAT MATTERS FOR THE TREE. Path survival factorises by the chain rule into
per-edge conditionals, so a sum of calibrated per-edge log-probabilities is a
calibrated log-probability that the whole path survives. That quantity is
commensurable ACROSS DEPTHS, which is exactly what a global top-B rerank over a
candidate pool needs. The draft's own cumulative log-prob is a sum of log-probs
of the wrong event (what the draft would sample, not what the target accepts),
and it is overconfident by ~0.26 nats per edge -- an error that compounds with
depth, so deep paths are systematically over-valued and the budget degenerates
onto one deep chain.

Measured on Qwen3-1.7B draft / Qwen3-235B-A22B target at an identical 32-node
verify budget, reranking by this head instead of cumulative draft log-prob is
worth +1.36 MAL on held-out RepoBench, +1.41 on GSM8K and +1.04 on
OpenCodeInstruct (each with a head trained on that corpus).

CALIBRATION IS DISTRIBUTION-SPECIFIC -- THIS IS THE MAIN DEPLOYMENT HAZARD.
A head applied to a distribution it was not trained on is not merely weaker, it
can be WORSE THAN NO HEAD AT ALL: the RepoBench-trained head scores -0.36 vs
draft rerank on GSM8K and -2.25 on MATH-500. Train on the serving distribution,
and fall back to draft ordering when drift is detected.

FEATURE PARITY IS LOAD-BEARING. Every feature below must be produced exactly as
in training, or the head is silently miscalibrated and lands in the negative
regime above. Two specific hazards, both encoded in `build_agreement_features`:

  * candidate_logprobs comes from a PLAIN log_softmax over the raw draft logits.
    It is NOT SGLang's `renorm_draft_probs` output, which applies temperature /
    top-p sampling adjustments that training never saw.
  * top1_top2_gap is a difference of LOG-probabilities, not of probabilities.

`build_agreement_features` is the single shared entry point: the serving path and
the feature-dump path used to retrain the head must both call it, so parity is
structural rather than something to be re-tested after every change.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch import nn

from sglang.srt.environ import envs


class AgreementScorer(nn.Module):
    """Grouped top-K candidate scorer with an explicit none-of-K class.

    Ported verbatim from corsair_spec_selector.agreement_model so that
    checkpoints trained by scripts/train_agreement.py load unchanged. Do not
    "clean up" the arithmetic: the saved weights are tied to this exact form.

    The candidate score is a RESIDUAL on the draft's own log-prob
    (`candidate_logprobs + compatibility + aux`), which is why 525K parameters
    suffice -- the model only has to learn where and by how much the draft is
    wrong.
    """

    # Order is load-bearing: it fixes the column layout of the gate vector that
    # `none_head` and `candidate_aux` were trained against.
    LEGACY_GATE_FEATURES = ("entropy", "top1_top2_gap", "position", "context_length")
    # Features derivable from the RETAINED top-k alone, so they cost nothing beyond
    # the k exponentials the normaliser already needs and never touch the full vocab.
    # Added 2026-09-14 to replace `entropy`, which is the only feature that forces a
    # full-vocab softmax+exp (two 128x151936 fp32 materialisations per level).
    #
    # Measured on 43,802 real main9000 gates:
    #   * entropy is a near-perfect MONOTONE function of top1_top2_gap
    #     (Spearman -0.9977), and `candidate_aux`/`none_head` are MLPs that can learn
    #     any monotone transform of gap. Its marginal contribution to predicting
    #     P(target == draft top-1) is +0.00221 nats, 1.3% of what gap alone provides.
    #   * for the none-of-K class -- the one that has to absorb everything outside
    #     the top-k, and which grows from 2.16% at K=16 to 3.58% at K=8 --
    #     `spread` BEATS entropy outright: gap+spread scores 0.11668 nats against
    #     gap+entropy's 0.12744 (constant baseline 0.15443). Adding entropy on top
    #     of gap+eff_count+spread is worth +0.00013 nats, i.e. nothing.
    # So removing the full-vocab softmax does not weaken the none class, it
    # strengthens it.
    TOPK_GATE_FEATURES = ("spread", "eff_count")
    ALL_GATE_FEATURES = LEGACY_GATE_FEATURES + TOPK_GATE_FEATURES

    def __init__(
        self,
        embedding_weight: torch.Tensor,
        *,
        projection_dim: int = 128,
        top_k: int = 16,
        gate_features: Optional[Sequence[str]] = None,
        normalizer: str = "full",
    ) -> None:
        super().__init__()
        self.top_k = top_k
        # Recorded on the module (and in the checkpoint) rather than passed at each
        # call site, so the serving path cannot normalise differently from training.
        if normalizer not in ("full", "topk"):
            raise ValueError(f"unknown normalizer {normalizer!r}")
        self.normalizer = normalizer
        # `position` and `context_length` are opt-OUT rather than removed, because
        # every checkpoint trained before 2026-09-14 has them baked into the input
        # widths below. Default = legacy, so old checkpoints load unchanged.
        #
        # Why you would drop them (measured 2026-09-13 on 1024 real gates, in nats
        # of log P(target accepts), against a ~0.26 nats/edge signal):
        #   * context_length moves the output 0.02 over its whole 2k..25k range.
        #     It is inert.
        #   * position moves it 0.81 over its training range -- the largest single
        #     influence -- but TRAINING records it as index-into-generation (0..255,
        #     mean 121) while INFERENCE passes depth-below-root (0..L). Those are
        #     different quantities in the same slot. Worse, the effect is almost
        #     pure per-gate SHIFT (0.69 nats) rather than reordering (0.11), and a
        #     per-edge shift is exactly a depth-proportional bias on a path score --
        #     i.e. it lands on the one thing this head exists to calibrate.
        #     Conditional on the prefix being accepted (which is the event the label
        #     encodes), tree depth carries no information about the label, so the
        #     honest fix is to drop it rather than relabel it.
        self._gate_features = tuple(
            self.LEGACY_GATE_FEATURES if gate_features is None else gate_features
        )
        unknown = set(self._gate_features) - set(self.ALL_GATE_FEATURES)
        if unknown:
            raise ValueError(f"unknown gate features: {sorted(unknown)}")
        n_gate = len(self._gate_features)
        self.gate_feature_names = self._gate_features
        hidden_size = embedding_weight.shape[1]
        self.embedding = nn.Embedding.from_pretrained(
            embedding_weight.float(), freeze=True
        )
        self.hidden_projection = nn.Linear(hidden_size, projection_dim, bias=False)
        self.token_projection = nn.Linear(hidden_size, projection_dim, bias=False)
        self.candidate_aux = nn.Sequential(
            nn.Linear(2 + n_gate, 32), nn.SiLU(), nn.Linear(32, 1)
        )
        self.none_head = nn.Sequential(nn.Linear(n_gate, 32), nn.SiLU(), nn.Linear(32, 1))
        self.scale = math.sqrt(projection_dim)

    def gate_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # `spread` and `eff_count` are computed from candidate_logprobs, which the
        # caller has ALREADY sliced to top_k -- so they describe the retained
        # candidate set, which is exactly the set the none class is defined against.
        # Memoised: `spread` and `eff_count` each referenced this twice, so the
        # naive closure launched four redundant .float()+slice kernels per call and
        # made the scorer SLOWER at K=8 than the legacy head was at K=16 (0.590 vs
        # 0.463 ms). At this size the head is launch-bound, not FLOP-bound, so the
        # op COUNT is what matters. Still lazy: the legacy gate never touches it.
        _cache: Dict[str, torch.Tensor] = {}

        def _lp() -> torch.Tensor:
            if "lp" not in _cache:
                _cache["lp"] = batch["candidate_logprobs"].float()[:, : self.top_k]
            return _cache["lp"]

        columns = {
            "entropy": lambda: batch["entropy"],
            "top1_top2_gap": lambda: batch["top1_top2_gap"],
            "position": lambda: torch.log1p(batch["position"].float()),
            "context_length": lambda: torch.log1p(batch["context_length"].float()) / 10.0,
            # top1 - topK: how far the retained set spreads. Best single free
            # predictor of none-of-K.
            "spread": lambda: _lp()[:, 0] - _lp()[:, -1],
            # log of the effective number of candidates inside the top-k; a
            # TRUNCATED entropy (Spearman +0.9612 with the full-vocab entropy)
            # obtained from the k exps the normaliser already computes.
            "eff_count": lambda: torch.logsumexp(_lp(), dim=-1) - _lp()[:, 0],
        }
        return torch.stack([columns[name]() for name in self._gate_features], dim=-1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = batch["hidden"].float()
        candidate_ids = batch["candidate_ids"].long()
        candidate_logprobs = batch["candidate_logprobs"].float()
        gate = self.gate_features(batch)
        projected_hidden = self.hidden_projection(hidden)
        projected_tokens = self.token_projection(self.embedding(candidate_ids))
        compatibility = torch.einsum("bd,bkd->bk", projected_hidden, projected_tokens)
        compatibility = compatibility / self.scale
        ranks = torch.arange(
            1, self.top_k + 1, device=hidden.device, dtype=torch.float32
        )[None, :].expand_as(candidate_logprobs)
        expanded_gate = gate[:, None, :].expand(-1, self.top_k, -1)
        candidate_features = torch.cat(
            [candidate_logprobs[..., None], (ranks / self.top_k)[..., None], expanded_gate],
            dim=-1,
        )
        candidate_scores = (
            candidate_logprobs
            + compatibility
            + self.candidate_aux(candidate_features).squeeze(-1)
        )
        none_score = self.none_head(gate)
        return torch.cat([candidate_scores, none_score], dim=-1)


# Agreement probabilities are clamped before the log so a zero-probability edge
# becomes a large finite penalty rather than -inf, which would make every
# descendant incomparable under the cumulative sum. Matches
# corsair eagle_path._MIN_AGREEMENT.
MIN_AGREEMENT = 1e-30
LOG_MIN_AGREEMENT = math.log(MIN_AGREEMENT)

# spec_utils uses -1e30 as the "this branch is dropped" sentinel in the draft
# score tensors (children cap, nucleus mass cut). Cumulative sums keep such an
# entry astronomically negative but no longer exactly -1e30, so the test has to
# be a threshold. Real cumulative log-agreement is bounded below by
# depth * log(MIN_AGREEMENT) ~ -4.4e3 at depth 64, so -1e29 separates them by 25
# orders of magnitude.
DROP_SENTINEL_MAX = -1e29


@torch.no_grad()
def align_log_agreement(
    log_agree: torch.Tensor,
    head_ids: torch.Tensor,
    selected_ids: torch.Tensor,
) -> torch.Tensor:
    """Reorder per-candidate log-agreement onto the tree's own child ordering.

    The head scores ITS OWN top-k of the RAW draft logits. EAGLE expands the
    top-k of `renorm_draft_probs`, a different tensor (temperature / top-p
    adjusted) that can order or truncate differently, and `head.top_k` (16, set
    by the checkpoint) need not equal `speculative_eagle_topk`. Column j of one
    is therefore not column j of the other, and silently assuming it is would
    attach every score to the wrong edge -- which reads as "the head does not
    work in serving" rather than as a bug.

    Matching by token id is what corsair's runtime does
    (`sglang_runtime._selected_tokens_from_hidden` looks the emitted token up in
    its own top-k). A child outside the head's top-k gets the MIN_AGREEMENT
    floor: the head declined to rank it, which is exactly the "not in my
    candidate set" verdict its none-of-K class expresses.
    """
    match = head_ids.unsqueeze(1) == selected_ids.unsqueeze(2)  # (N, sel, K)
    found = match.any(-1)
    columns = match.int().argmax(-1)
    aligned = torch.gather(log_agree, 1, columns)
    return torch.where(found, aligned, aligned.new_full((), LOG_MIN_AGREEMENT))


@torch.no_grad()
def build_agreement_features(
    logits: torch.Tensor,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    context_lengths: torch.Tensor,
    top_k: int,
    *,
    need_entropy: bool = True,
    normalizer: str = "full",
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """THE single feature contract. Returns (batch, top_ids, top_log_probs).

    `logits` must be the RAW draft next-token logits, pre-sampling-adjustment.
    Passing `renorm_draft_probs` output here is the parity bug this function
    exists to prevent.

    NORMALIZER (added 2026-09-14). `full` is the legacy path: a log_softmax over
    the whole vocabulary, then topk. `topk` instead takes topk on the RAW logits
    and normalises over the retained k alone. The two differ by a per-row
    constant, `log(1/mass_topk)`, which on 43,802 real gates is 0.0000 nats at the
    median and 0.0140 at p99 for k=8, against the ~0.26 nats/edge signal the head
    calibrates. It is NOT free, though: the shift applies to the candidate scores
    but NOT to `none_score`, so it moves the candidate-vs-none-of-K balance. The
    checkpoint therefore records which normalizer it was trained with and this is
    driven from that -- a head trained under one and served under the other is
    miscalibrated exactly as the module docstring warns.

    WHY `topk` IS WORTH IT. `full` materialises TWO (rows x 151936) fp32 tensors
    per level -- 78 MB each at 128 rows -- for log_softmax and for the entropy
    `exp`, then reads them again for topk and the entropy sum. That bandwidth, not
    the arithmetic, is why features cost 0.714 ms of the head's 1.811 ms profile.
    Under `topk` with `need_entropy=False` nothing full-vocab is ever
    materialised: one streaming pass over the logits for topk (comparisons only),
    then k exponentials per row. For k=8 that is 8 exps instead of 2 x 151936, a
    ~38,000x reduction in transcendental ops, which matters on any target whose
    vectorised `exp` is weaker than its reductions.

    `topk(log_softmax(x)) == topk(x)` because log_softmax is strictly monotone, so
    taking topk on the raw logits changes no indices.
    """
    if normalizer not in ("full", "topk"):
        raise ValueError(f"unknown normalizer {normalizer!r}; expected 'full' or 'topk'")
    if normalizer == "full" or need_entropy:
        # entropy is the ONLY feature that needs the whole distribution, so it
        # forces the full-vocab path regardless of the normalizer setting.
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        top_log_probs, top_ids = torch.topk(log_probs, top_k, dim=-1)
        if normalizer == "topk":
            top_log_probs = top_log_probs - torch.logsumexp(
                top_log_probs, dim=-1, keepdim=True
            )
    else:
        top_logits, top_ids = torch.topk(logits.float(), top_k, dim=-1)
        top_log_probs = top_logits - torch.logsumexp(top_logits, dim=-1, keepdim=True)
    batch = {
        "hidden": hidden.float(),
        "candidate_ids": top_ids,
        "candidate_logprobs": top_log_probs,
        # LOG-space difference, matching training. Note the normaliser cancels
        # exactly here, so this column is identical under both modes.
        "top1_top2_gap": top_log_probs[:, 0] - top_log_probs[:, 1],
        "position": positions,
        "context_length": context_lengths,
    }
    if need_entropy:
        probabilities = log_probs.exp()
        batch["entropy"] = -(probabilities * log_probs).sum(-1)
    return batch, top_ids, top_log_probs


@torch.no_grad()
def agreement_log_probs(
    head: AgreementScorer,
    logits: torch.Tensor,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    context_lengths: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Per-edge log P(target accepts) for the draft's top-k.

    Returns (log_agree, top_ids, features). The feature dict is returned rather
    than rebuilt by callers that want to inspect it, so a dump is literally the
    tensors the head consumed.

    The softmax runs over top_k + 1 classes and only the first top_k are kept:
    the dropped none-of-K class is what lets the candidate mass sum to LESS than
    one, which is the head's only way to express "this gate is risky". Dropping
    the class instead of renormalising is therefore deliberate.
    """
    batch, top_ids, _ = build_agreement_features(
        logits,
        hidden,
        positions,
        context_lengths,
        head.top_k,
        # Structural parity: both flags come from the checkpoint, so a head that
        # was trained without entropy never pays for it, and one that was trained
        # with a given normaliser is always served with it.
        need_entropy="entropy" in head.gate_feature_names,
        normalizer=head.normalizer,
    )
    probs = torch.softmax(head(batch), dim=-1)[:, : head.top_k]
    return probs.clamp_min(MIN_AGREEMENT).log(), top_ids, batch


def load_agreement_head(
    path: str,
    embedding_weight: torch.Tensor,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> AgreementScorer:
    """Load a checkpoint written by corsair scripts/train_agreement.py.

    `embedding_weight` should be the DRAFT model's input embedding. Reusing it
    avoids carrying a second ~620 MB copy of the Qwen3 embedding matrix, and it
    is frozen in training so the values are identical by construction.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    # scripts/train_agreement.py writes {"model_state": ..., "config": ..., ...}
    # and deliberately OMITS embedding.weight (it is ~1.2 GB per checkpoint and
    # frozen anyway), which is why the caller supplies the draft embedding.
    weights = state.get("model_state", state.get("model", state))
    config = state.get("config", {}) if isinstance(state, dict) else {}
    top_k = int(config.get("top_k", 16))
    projection_dim = int(config.get("projection_dim", 128))
    # Absent => the legacy 4-feature gate, which is what every checkpoint before
    # 2026-09-14 is. Newer ones name their gate columns explicitly so the serving
    # side cannot silently build a differently-shaped head.
    gate_features = config.get("gate_features")
    # Absent => "full", i.e. every checkpoint trained before 2026-09-14.
    normalizer = config.get("normalizer", "full")
    head = AgreementScorer(
        embedding_weight,
        projection_dim=projection_dim,
        top_k=top_k,
        gate_features=gate_features,
        normalizer=normalizer,
    )
    # The frozen embedding is supplied by the caller, so a checkpoint that also
    # stored it must not overwrite the draft's copy.
    weights = {k: v for k, v in weights.items() if not k.startswith("embedding.")}
    missing, unexpected = head.load_state_dict(weights, strict=False)
    unexpected = [k for k in unexpected if not k.startswith("embedding.")]
    if unexpected:
        raise ValueError(f"unexpected keys in agreement checkpoint: {unexpected}")
    missing = [k for k in missing if not k.startswith("embedding.")]
    if missing:
        raise ValueError(f"missing keys in agreement checkpoint: {missing}")
    return head.to(device=device, dtype=dtype).eval()


class AgreementTreeScorer:
    """Threads cumulative log-agreement alongside EAGLE's cumulative draft score.

    DESIGN: the beam / frontier is still pruned by the DRAFT score; only the
    final global top-B rerank switches to agreement. That split is empirical --
    pruning the frontier by agreement instead ("afrontier") was measured as a
    null (-0.39% MAL at budget 32, replicated on two independent runs), so the
    beam search is deliberately left untouched. It also means this class never
    has to reimplement EAGLE's selection: it mirrors the beam choice that
    `select_top_k_tokens` already made.

    HOW THE MIRROR WORKS. `_select_top_k_tokens_later` returns
    ``tree_info[2] = topk_cs_index + (topk*topk*(i-1) + topk)``, so the beam
    indices it chose are recoverable exactly by subtracting that constant. We
    gather the agreement expansion at those same indices, which keeps the two
    cumulative scores aligned to the same tree without touching any of the three
    @torch.compile'd selection kernels.

    WHAT THE RERANK CONSUMES. `organize_draft_results` builds its global top-B
    over ``torch.cat(score_list)``. Feeding it `agree_score_list` instead of
    `score_list` switches the node value function with no change to that
    function -- and ancestor closure still holds for free, because cumulative
    log-agreement is a sum of non-positive terms, so no descendant can outrank
    its ancestor and the depth-alpha repair path never fires.
    """

    def __init__(self, head: AgreementScorer, topk: int):
        self.head = head
        self.topk = topk
        self.reset()

    def reset(self) -> None:
        self.cum: Optional[torch.Tensor] = None
        self.score_list: list[torch.Tensor] = []

    @torch.no_grad()
    def edge_log_agreement(
        self,
        next_token_logits: torch.Tensor,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        context_lengths: torch.Tensor,
        selected_ids: Optional[torch.Tensor] = None,
        out_features: Optional[dict] = None,
    ) -> torch.Tensor:
        """Per-edge log P(target accepts), on the caller's child ordering.

        `next_token_logits` must be the RAW draft logits. Do not pass
        `renorm_draft_probs` output -- see build_agreement_features.

        `selected_ids` is the tree's own top-k token ids (EAGLE's `topk_index`)
        for these gates. It is what the returned columns are aligned to; pass it
        whenever the result will be threaded through step_root / step_later.
        """
        log_agree, head_ids, features = agreement_log_probs(
            self.head, next_token_logits, hidden, positions, context_lengths
        )
        if out_features is not None:
            out_features.update(features)
        if selected_ids is None:
            return log_agree
        # IDENTITY SHORTCUT. align exists for exactly two reasons (see its
        # docstring): the head's K may differ from `speculative_eagle_topk`, and
        # the head ranks RAW logits while EAGLE expands `renorm_draft_probs`. The
        # second reason does not hold for this implementation --
        # `spec_utils.renorm_draft_probs` is `softmax(logits)` or
        # `softmax(logits / T)`, both strictly order-preserving with no top-p
        # truncation -- so when the widths also match, column j of head_ids IS
        # column j of selected_ids and align is an identity map over ~6 kernels.
        #
        # Skipping it is worth less than the eager profile suggests: the tensors
        # are tiny (N x sel x K), so almost all of align's 0.64 ms/level at 128
        # rows is launch overhead, and the head runs INSIDE the graph-captured
        # draft loop where replay already amortises that. Expect ~6 kernels x a
        # few us of GPU-side launch latency per level, not 0.64 ms.
        #
        # Guarded by an env check rather than trusted blindly, because being
        # wrong here silently attaches every score to the wrong edge -- which
        # reads as "the head does not work in serving" rather than as a bug.
        if selected_ids.shape[-1] == head_ids.shape[-1]:
            if envs.SGLANG_DEBUG_AGREEMENT_VERIFY_ALIGN.get():
                if not torch.equal(selected_ids, head_ids):
                    raise AssertionError(
                        "agreement align identity shortcut is invalid: head_ids "
                        "and selected_ids have equal width but differ. The "
                        "order-preservation assumption about renorm_draft_probs "
                        "no longer holds; remove the shortcut."
                    )
            return log_agree
        return align_log_agreement(log_agree, head_ids, selected_ids)

    @staticmethod
    def _inherit_drops(
        entry: torch.Tensor, draft_entry: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Carry the draft pool's drop sentinels into the agreement pool.

        `select_top_k_tokens` prunes structurally by writing -1e30 into the score
        tensor it returns: SGLANG_SPEC_EXPAND_CHILDREN caps each node's branching
        that way, and the nucleus path does the same for lines outside the mass
        cut. Those decisions live ONLY in the draft scores, so swapping the pool
        wholesale would quietly resurrect every branch they removed and change
        the tree's shape, not just its ranking. Re-stamping the sentinel keeps
        the rerank a pure change of value function.
        """
        if draft_entry is None:
            return entry
        return torch.where(
            draft_entry <= DROP_SENTINEL_MAX, draft_entry, entry
        )

    def step_root(
        self, log_agree: torch.Tensor, draft_entry: Optional[torch.Tensor] = None
    ) -> None:
        """Level 0. Every top-k root child is kept, so the beam is the identity."""
        self.cum = log_agree  # (b, topk)
        self.score_list.append(
            self._inherit_drops(log_agree.unsqueeze(1), draft_entry)
        )  # (b, 1, topk)

    def step_later(
        self,
        i: int,
        log_agree: torch.Tensor,
        parents: torch.Tensor,
        draft_entry: Optional[torch.Tensor] = None,
    ) -> None:
        """Level i >= 1. `parents` is tree_info[2] from select_top_k_tokens."""
        topk = self.topk
        expand = self.cum.unsqueeze(2) + log_agree.view(-1, topk, topk)
        self.score_list.append(self._inherit_drops(expand, draft_entry))
        beam = parents - (topk * topk * (i - 1) + topk)
        self.cum = torch.gather(expand.flatten(start_dim=1), 1, beam)

    def rerank_scores(self) -> list[torch.Tensor]:
        return self.score_list
