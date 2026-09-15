from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Optional

import torch
from huggingface_hub import snapshot_download

from sglang.srt.distributed.parallel_state import (
    GroupCoordinator,
    patch_tensor_parallel_group,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import set_mamba_track_indices_from_reqs
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.triton_ops.cache_locs import (
    align_evict_mask_to_page_size as align_evict_mask_to_page_size,
)
from sglang.srt.speculative.triton_ops.cache_locs import (
    assign_extend_cache_locs as assign_extend_cache_locs,
)
from sglang.srt.speculative.triton_ops.cache_locs import (
    assign_req_to_token_pool as assign_req_to_token_pool,
)
from sglang.srt.speculative.triton_ops.cache_locs import (
    assign_req_to_token_pool_func as assign_req_to_token_pool_func,
)
from sglang.srt.speculative.triton_ops.cache_locs import (
    filter_finished_cache_loc_kernel as filter_finished_cache_loc_kernel,
)
from sglang.srt.speculative.triton_ops.cache_locs import (
    generate_draft_decode_kv_indices as generate_draft_decode_kv_indices,
)
from sglang.srt.speculative.triton_ops.cache_locs import (
    get_src_tgt_cache_loc as get_src_tgt_cache_loc,
)
from sglang.srt.speculative.triton_ops.cache_locs import (
    get_target_cache_loc as get_target_cache_loc,
)
from sglang.srt.speculative.triton_ops.eagle import (
    fill_accept_out_cache_loc as fill_accept_out_cache_loc,
)
from sglang.srt.utils import is_cuda, is_hip, is_musa, is_npu, next_power_of_2
from sglang.srt.utils.async_probe import maybe_detect_oob
from sglang.srt.utils.nvtx_utils import profile_range

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_musa = is_musa()

# Experimental tree EXPAND policy (pt_tree port). 0.0 = native fixed-top-k beam.
# >0 = nucleus controlled-expand (keep minimal beam covering this path-prob mass each
# step, capped at eagle_topk). Read once at import; the worker subprocess inherits the
# env from the benchmark. See SGLANG_SPEC_EXPAND_P in environ.py.
_SPEC_EXPAND_P = float(envs.SGLANG_SPEC_EXPAND_P.get() or 0.0)
if _SPEC_EXPAND_P > 0.0:
    logging.getLogger(__name__).warning(
        "[spec] nucleus controlled-expand ACTIVE: SGLANG_SPEC_EXPAND_P=%.3f", _SPEC_EXPAND_P
    )

# Nucleus grace window: protect a branch that just failed the mass cut for k more
# levels before actually masking it. Only meaningful when nucleus is active. See
# SGLANG_SPEC_NUCLEUS_GRACE in environ.py.
_SPEC_NUCLEUS_GRACE = int(envs.SGLANG_SPEC_NUCLEUS_GRACE.get() or 0)
if _SPEC_NUCLEUS_GRACE > 0 and _SPEC_EXPAND_P > 0.0:
    logging.getLogger(__name__).warning(
        "[spec] nucleus grace window ACTIVE: SGLANG_SPEC_NUCLEUS_GRACE=%d "
        "(branches under the mass cut survive %d extra level(s) before dropping)",
        _SPEC_NUCLEUS_GRACE, _SPEC_NUCLEUS_GRACE,
    )

# Delayed nucleus onset: keep native fixed-topk expand (full width, no mass cut)
# through depth k, only starting the nucleus mass cut from depth k+1 on. Guarantees
# full exploration near the root/gate before reallocating budget toward the spine.
# Only meaningful when nucleus is active. See SGLANG_SPEC_NUCLEUS_DELAY in environ.py.
_SPEC_NUCLEUS_DELAY = int(envs.SGLANG_SPEC_NUCLEUS_DELAY.get() or 0)
if _SPEC_NUCLEUS_DELAY > 0 and _SPEC_EXPAND_P > 0.0:
    logging.getLogger(__name__).warning(
        "[spec] nucleus delayed onset ACTIVE: SGLANG_SPEC_NUCLEUS_DELAY=%d "
        "(native fixed-topk expand through depth %d, nucleus mass cut from depth %d on)",
        _SPEC_NUCLEUS_DELAY, _SPEC_NUCLEUS_DELAY, _SPEC_NUCLEUS_DELAY + 1,
    )

# Decouple the per-node branching factor (children/node) from the expand beam width.
# 0 = disabled (children = eagle_topk = beam width, native EAGLE-2). k>0 caps each node to
# its top-k children so a WIDE beam can explore over NARROW branching (pt_tree W>children).
# Only the expansion steps are capped; the root still seeds the full beam. See environ.py.
_SPEC_EXPAND_CHILDREN = int(envs.SGLANG_SPEC_EXPAND_CHILDREN.get() or 0)
if _SPEC_EXPAND_CHILDREN > 0:
    logging.getLogger(__name__).warning(
        "[spec] decoupled branching ACTIVE: SGLANG_SPEC_EXPAND_CHILDREN=%d "
        "(children/node; beam width stays speculative_eagle_topk)",
        _SPEC_EXPAND_CHILDREN,
    )

# Accumulate tree path scores as a sum of log-probs instead of a product of probs.
# The score is only a ranking key (topk), so log is a monotonic transform: identical
# selection to product domain but no underflow on deep trees (large num_steps). Read
# once at import; the worker subprocess inherits the env. See environ.py.
_SPEC_LOG_DOMAIN = bool(envs.SGLANG_SPEC_ENABLE_LOG_DOMAIN.get())
# clamp floor for log(prob): avoids log(0) = -inf; ~smallest normal float32.
_LOG_PROB_FLOOR = 1e-30
# "dropped beam" sentinel in log domain (product-domain uses 0.0). Far below any real
# cumulative log-prob so the node and its children sort last in the global rerank.
_LOG_DROP = -1e30
if _SPEC_LOG_DOMAIN:
    logging.getLogger(__name__).warning(
        "[spec] log-domain score accumulation ACTIVE: SGLANG_SPEC_ENABLE_LOG_DOMAIN=1"
    )

if TYPE_CHECKING:
    from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.eagle_info import EagleVerifyInput


if _is_cuda:
    from sgl_kernel import fast_topk
elif _is_hip:
    from sgl_kernel import fast_topk
else:
    from sglang.srt.utils.common import fast_topk


logger = logging.getLogger(__name__)


def fast_sample(probs: torch.Tensor, num_samples: int = 1):
    sample_index = torch.multinomial(probs, num_samples=num_samples)
    sample_p = probs.gather(1, sample_index)
    return sample_p, sample_index


def renorm_draft_probs(
    next_token_logits: torch.Tensor,
    sampling_info,
    use_rejection_sampling: bool,
) -> torch.Tensor:
    """Draft-side next-token distribution.

    Plain softmax, except under rejection sampling where logits are
    temperature-scaled so the draft proposal q tracks the target sampling
    temperature (higher acceptance; correctness holds for any q).

    SGLANG_SPEC_DRAFT_FP32_SOFTMAX=1: upcast logits to float32 before softmax.
    bf16 logits produce a flatter distribution (stronger competing branches at
    shallow depths fill the top-B verify budget and crowd out deep canonical
    paths). float32 sharpens the distribution, matching pt_tree's HF behaviour.
    """
    if not use_rejection_sampling or not next_token_logits.size(0):
        return torch.softmax(next_token_logits, dim=-1)
    return torch.softmax(next_token_logits / sampling_info.temperatures, dim=-1)


# Simulate acceptance length for benchmarking purposes
SIMULATE_ACC_LEN = envs.SGLANG_SIMULATE_ACC_LEN.get()  # turn off if < 0
SIMULATE_ACC_METHOD = envs.SGLANG_SIMULATE_ACC_METHOD.get()

TREE_TRAVERSE_TIME_THRESHOLD = 1  # TODO: set this properly
TREE_SPEC_KERNEL_AVAILABLE = (
    _is_cuda or _is_musa
)  # This kernel is only available for CUDA and MUSA now


def draft_kv_indices_buffer_width(
    num_seqs: int, topk: int, max_context_len: int
) -> int:
    """Per-step row width of the EAGLE draft-decode kv_indices buffer.

    num_seqs * topk branches each attend up to max_context_len KV slots; the topk
    factor is mandatory -- dropping it under-allocates and overflows the row (#27338, #27460).
    """
    assert (
        num_seqs * topk * max_context_len < 2**31
    ), "kv_indices flat offset would overflow int32; reduce batch/topk/context"
    return num_seqs * topk * max_context_len


def draft_kv_indices_used_len(
    seq_lens_sum: int, topk: int, bs: int, num_steps: int
) -> int:
    """kv_indices length used through num_steps draft-decode steps.

    bs = topk * num_seqs branches, one index appended per branch per step. Called with
    num_steps = i + 1 (per-step slice) and speculative_num_steps (capacity assert).
    """
    return seq_lens_sum * topk + bs * num_steps


def record_stream_each(tensors, stream):
    """Call record_stream(stream) on each cuda tensor in `tensors`, skipping
    non-tensor / non-cuda entries. Tells the caching allocator that the
    tensors are also used on `stream`, so memory is not recycled while
    queued work is still in flight after Python refs drop.
    """
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.is_cuda:
            t.record_stream(stream)


def record_stream_for_v2_verify(batch, verify_input, fwd_stream):
    """Mark pre-prepare SB / verify_input GPU tensors as used on `fwd_stream`.

    Spec V2 mutates SB mid-forward (`prepare_for_verify` rebinds
    `batch.input_ids` / `out_cache_loc`; `_draft_extend_for_decode` later
    replaces `batch.input_ids` again). Each rebind drops the only SB Python
    ref to the old tensor while the verify forward kernel may still be
    reading its memory on `fwd_stream`; `record_stream` tells the caching
    allocator to wait for `fwd_stream` before recycling the block.

    Covers pre-prepare tensors only; caller must also `record_stream_each`
    the post-prepare rebinds (new `batch.input_ids` / `out_cache_loc`).
    """
    candidates = [
        batch.seq_lens,
        batch.req_pool_indices,
        batch.input_ids,
        batch.out_cache_loc,
    ]
    if verify_input is not None:
        candidates.extend(
            [
                getattr(verify_input, attr, None)
                for attr in (
                    "draft_token",
                    "custom_mask",
                    "positions",
                    "retrieve_index",
                    "retrieve_next_token",
                    "retrieve_next_sibling",
                )
            ]
        )
    record_stream_each(candidates, fwd_stream)


def spec_need_hidden_states(server_args: Optional[ServerArgs] = None) -> bool:
    if server_args is None:
        server_args = get_global_server_args()

    # STANDALONE drafts don't consume `spec_info.hidden_states` (vanilla LLM).
    # multi_layer_eagle and DFLASH don't relay hidden_states through FutureMap.
    # TODO(lsyin): also skip when step == 1.
    if server_args.speculative_algorithm == "DFLASH":
        return False
    if server_args.speculative_algorithm == "STANDALONE":
        # ...except with --speculative-agreement-head, whose root gate reads the
        # draft's hidden state out of spec_info. Without the relay the merged
        # batch keeps whatever hidden_states the first request contributed, so a
        # batch of two decodes silently scores both roots from one request's
        # hidden state. Kept in step with eagle_utils.draft_carries_hidden_states,
        # which decides the matching CaptureHiddenMode.
        return server_args.speculative_agreement_head is not None
    return not server_args.enable_multi_layer_eagle


@torch.compile(dynamic=True, disable=_is_npu)
def create_num_accept_tokens_filter(
    num_correct_drafts: torch.Tensor,
    unfinished_index_device: torch.Tensor,
    seq_lens: torch.Tensor,
):
    num_accept_tokens_filter = torch.zeros_like(num_correct_drafts)
    num_accept_tokens_filter[unfinished_index_device] = (
        num_correct_drafts[unfinished_index_device] + 1
    )
    seq_lens.add_(num_correct_drafts + 1)
    return num_accept_tokens_filter


def _cap_children(
    expand_scores: torch.Tensor, topk: int, children: int, log_domain: bool
):
    """Cap each node's branching to its top-`children` children (beam width unchanged).

    expand_scores is (b, topk_beam, topk_children); the child dim is descending
    (topk_p came from fast_topk). Columns >= `children` are the lowest-prob children —
    replace them with the drop sentinel so they neither seed the beam (fast_topk over
    the flattened lines) nor survive the global rerank (they're stored in tree_info[0]).
    No-op unless 0 < children < topk."""
    if not (0 < children < topk):
        return expand_scores
    col = torch.arange(topk, device=expand_scores.device)
    drop = _LOG_DROP if log_domain else 0.0
    return torch.where(
        (col < children).view(1, 1, topk),
        expand_scores,
        expand_scores.new_full((), drop),
    )


def _select_top_k_tokens_first(
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: Optional[torch.Tensor],
    topk: int,
    log_domain: bool = False,
    grace: int = 0,
):
    input_ids = topk_index.flatten()
    if hidden_states is not None:
        hidden_states = hidden_states.repeat_interleave(topk, dim=0)

    # Level-0 path score. In log domain the whole tree accumulates as a sum of
    # log-probs, so seed the root scores (and score_list entry) with log p too —
    # every score_list entry must share one domain for the global rerank topk.
    scores = torch.log(topk_p.clamp_min(_LOG_PROB_FLOOR)) if log_domain else topk_p

    # Apply the same per-node children cap to the root that _cap_children applies
    # to subsequent expand steps. Without this, the root always seeds topk=beam
    # children into the global rerank pool regardless of SGLANG_SPEC_EXPAND_CHILDREN,
    # producing extra shallow (d1) candidates that crowd out deep canonical paths.
    # We zero scores (not just root_entry) so the -inf propagates through the
    # cumulative sums in all subsequent _select_top_k_tokens_later calls, which
    # also removes those branches from the frontier selection. input_ids is left
    # untouched (the draft model still runs on all topk root children; the extra
    # compute is a small cost for the diagnostic experiment).
    root_entry = scores.unsqueeze(1)  # (b, 1, topk)
    children = _SPEC_EXPAND_CHILDREN
    if 0 < children < topk:
        drop = _LOG_DROP if log_domain else 0.0
        col = torch.arange(topk, device=scores.device)
        mask = col < children  # (topk,)
        root_entry = torch.where(mask, root_entry, root_entry.new_full((), drop))
        scores = torch.where(mask, scores, scores.new_full((), drop))

    tree_info = (
        root_entry,  # (b, 1, topk)
        topk_index,  # (b, topk)
        torch.arange(-1, topk, dtype=torch.long, device=input_ids.device).expand(
            topk_p.shape[0], -1
        ),  # (b, topk + 1) — expand avoids the allocation of repeat
    )
    if grace > 0:
        # Root candidates are all "healthy" — full grace budget, nothing has failed
        # a mass cut yet (the root has no nucleus filtering).
        grace_left = torch.full(
            (topk_p.shape[0], topk), grace, dtype=torch.long, device=topk_p.device
        )
        tree_info = tree_info + (grace_left,)
    return input_ids, hidden_states, scores, tree_info


@torch.compile(dynamic=True, disable=_is_npu)
def _select_top_k_tokens_later(
    i: int,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
    log_domain: bool = False,
    children: int = 0,
):
    topk_sq = topk * topk

    if log_domain:
        # cumulative log-prob: parent log-score + child conditional log-prob.
        expand_scores = scores.unsqueeze(2) + torch.log(
            topk_p.view(-1, topk, topk).clamp_min(_LOG_PROB_FLOOR)
        )
    else:
        expand_scores = scores.unsqueeze(2) * topk_p.view(-1, topk, topk)
    # (b, topk, 1) [op] (b, topk, topk) -> (b, topk, topk)

    expand_scores = _cap_children(expand_scores, topk, children, log_domain)

    topk_cs_p, topk_cs_index = fast_topk(
        expand_scores.flatten(start_dim=1), topk, dim=-1
    )  # (b, topk)

    topk_index = topk_index.view(-1, topk_sq)
    input_ids = torch.gather(topk_index, 1, topk_cs_index).flatten()

    if hidden_states is not None and hidden_states.shape[0] > 0:
        flat_cs = topk_cs_index.flatten()
        batch_offsets = torch.arange(
            0, hidden_states.shape[0], step=topk, device=flat_cs.device
        )
        selected_input_index = flat_cs // topk + batch_offsets.repeat_interleave(topk)
        hidden_states = hidden_states[selected_input_index]

    tree_info = (
        expand_scores,  # (b, topk, topk)
        topk_index,  # (b, topk * topk)
        topk_cs_index + (topk_sq * (i - 1) + topk),  # (b, topk)
    )
    return input_ids, hidden_states, topk_cs_p, tree_info


@torch.compile(dynamic=True, disable=_is_npu)
def _select_top_k_tokens_later_nucleus(
    i: int,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
    p: float,
    log_domain: bool = False,
    children: int = 0,
    grace_left: Optional[torch.Tensor] = None,
    grace: int = 0,
):
    """Nucleus controlled-expand (pt_tree rerank-p): identical to _select_top_k_tokens_later
    but the next-step expand beam is narrowed to the minimal set of lines covering fraction
    `p` of the live path-prob mass (capped at topk). Lines beyond the cut get their carried
    score zeroed, so their children score 0 and are dropped by the global rerank
    (organize_draft_results) — the EXPAND beam breathes (~1 on a confident spine, up to topk
    at a gate) while the verify budget B is still allocated globally. This step's own nodes
    keep their real scores in score_list (tree_info[0]), so un-expanded siblings remain
    eligible verify leaves — exactly pt_tree's controlled-expand.

    grace (pt_tree port): the mass cut is memoryless — a line that fails it is masked
    immediately, and its children score ~-inf starting the VERY NEXT level (permanent
    death after one bad level). grace_left tracks a per-line countdown: a line that
    fails the mass cut but still has grace_left>0 (inherited from its parent) is kept
    anyway, and its budget decrements by one; a line that PASSES the mass cut has its
    budget refilled to `grace` (healthy again). 0 = disabled, native behavior."""
    topk_sq = topk * topk

    if log_domain:
        expand_scores = scores.unsqueeze(2) + torch.log(
            topk_p.view(-1, topk, topk).clamp_min(_LOG_PROB_FLOOR)
        )
    else:
        expand_scores = scores.unsqueeze(2) * topk_p.view(-1, topk, topk)
    expand_scores = _cap_children(expand_scores, topk, children, log_domain)
    topk_cs_p, topk_cs_index = fast_topk(
        expand_scores.flatten(start_dim=1), topk, dim=-1
    )  # (b, topk), descending (fast_topk -> torch.topk sorted)

    # nucleus mass cut over the (descending) kept lines: keep the minimal prefix whose
    # cumulative mass reaches p of the total; always keeps the top-1 line. The cut is a
    # PROBABILITY-mass cut, so in log domain exponentiate first (subtract the row max for
    # stability — the normalization constant cancels in the csum/total ratio).
    mass = (
        torch.exp(topk_cs_p - topk_cs_p.max(dim=-1, keepdim=True).values)
        if log_domain
        else topk_cs_p
    )
    total = mass.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    csum = torch.cumsum(mass, dim=-1)
    keep_by_mass = (csum - mass) < (p * total)

    new_grace_left = None
    if grace > 0 and grace_left is not None:
        parent_row = topk_cs_index // topk  # (b, topk) — which parent line each survivor came from
        parent_grace = grace_left.gather(1, parent_row)
        keep = keep_by_mass | (parent_grace > 0)
        new_grace_left = torch.where(
            keep_by_mass, torch.full_like(parent_grace, grace), (parent_grace - 1).clamp(min=0)
        )
    else:
        keep = keep_by_mass

    # "drop" = prob 0 (product domain) / -inf sentinel (log domain) so children rank last.
    drop = torch.full_like(topk_cs_p, _LOG_DROP if log_domain else 0.0)
    topk_cs_p = torch.where(keep, topk_cs_p, drop)

    topk_index = topk_index.view(-1, topk_sq)
    input_ids = torch.gather(topk_index, 1, topk_cs_index).flatten()

    if hidden_states is not None and hidden_states.shape[0] > 0:
        flat_cs = topk_cs_index.flatten()
        batch_offsets = torch.arange(
            0, hidden_states.shape[0], step=topk, device=flat_cs.device
        )
        selected_input_index = flat_cs // topk + batch_offsets.repeat_interleave(topk)
        hidden_states = hidden_states[selected_input_index]

    tree_info = (
        expand_scores,  # (b, topk, topk) — real candidate scores for the global rerank
        topk_index,  # (b, topk * topk)
        topk_cs_index + (topk_sq * (i - 1) + topk),  # (b, topk)
    )
    if new_grace_left is not None:
        tree_info = tree_info + (new_grace_left,)
    return input_ids, hidden_states, topk_cs_p, tree_info


def select_top_k_tokens(
    i: int,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
    grace_left: Optional[torch.Tensor] = None,
):
    if i == 0:
        return _select_top_k_tokens_first(
            topk_p, topk_index, hidden_states, topk, _SPEC_LOG_DOMAIN, _SPEC_NUCLEUS_GRACE,
        )
    if _SPEC_EXPAND_P > 0.0 and i >= _SPEC_NUCLEUS_DELAY:
        return _select_top_k_tokens_later_nucleus(
            i, topk_p, topk_index, hidden_states, scores, topk, _SPEC_EXPAND_P,
            _SPEC_LOG_DOMAIN, _SPEC_EXPAND_CHILDREN, grace_left, _SPEC_NUCLEUS_GRACE,
        )
    return _select_top_k_tokens_later(
        i, topk_p, topk_index, hidden_states, scores, topk, _SPEC_LOG_DOMAIN,
        _SPEC_EXPAND_CHILDREN,
    )


def _sample_simulated_acc_len(
    simulate_acc_len: float,
    simulate_acc_method: str,
    max_len: int,
) -> int:
    """Sample a simulated acceptance length in [1, max_len]."""
    if simulate_acc_method == "multinomial":
        simulated_values = torch.normal(
            mean=simulate_acc_len,
            std=1.0,
            size=(1,),
            device="cpu",
        )
        # clamp simulated values to be between 1 and max_len
        simulated_values = torch.clamp(simulated_values, min=1.0, max=max_len)
        simulate_acc_len = int(simulated_values.round().item())
    elif simulate_acc_method == "match-expected":
        # multinomial sampling does not match the expected length
        # we keep it for the sake of compatibility of existing tests
        # but it's better to use "match-expected" for the cases that need to
        # match the expected length, One caveat is that this will only sample
        # either round down or round up of the expected length
        simulate_acc_len = max(1.0, min(max_len, simulate_acc_len))
        lower = int(simulate_acc_len // 1)
        upper = lower + 1 if lower < max_len else lower
        if lower == upper:
            simulate_acc_len = lower
        else:
            weight_upper = simulate_acc_len - lower
            weight_lower = 1.0 - weight_upper
            probs = torch.tensor([weight_lower, weight_upper], device="cpu")
            sampled_index = torch.multinomial(probs, num_samples=1)
            simulate_acc_len = lower if sampled_index == 0 else upper
    else:
        raise ValueError(f"Invalid simulate_acc_method: {simulate_acc_method}")
    return int(simulate_acc_len)


def generate_simulated_accept_index(
    accept_index,
    predict,
    num_correct_drafts,
    bs,
    spec_steps,
    simulate_acc_len: float = SIMULATE_ACC_LEN,
    simulate_acc_method: str = SIMULATE_ACC_METHOD,
):
    assert simulate_acc_len > 0.0
    simulate_acc_len = _sample_simulated_acc_len(
        simulate_acc_len, simulate_acc_method, spec_steps + 1
    )

    accept_indx_first_col = accept_index[:, 0].view(-1, 1)
    sim_accept_index = torch.full(
        (bs, spec_steps + 1), -1, dtype=torch.int32, device="cuda"
    )
    sim_accept_index[:, :simulate_acc_len] = accept_indx_first_col + torch.arange(
        simulate_acc_len, device=accept_index.device
    )
    num_correct_drafts.fill_(simulate_acc_len - 1)
    predict.fill_(100)  # some legit token id
    return sim_accept_index


def traverse_tree(
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    draft_tokens: torch.Tensor,
    grammar: BaseGrammarObject,
    allocate_token_bitmask: torch.Tensor,
    vocab_size: Optional[int] = None,
):
    """
    Traverse the tree constructed by the draft model to generate the logits mask.
    """
    assert (
        retrieve_next_token.shape == retrieve_next_sibling.shape == draft_tokens.shape
    )

    def dfs(
        curr: int,
        retrieve_next_token: torch.Tensor,
        retrieve_next_sibling: torch.Tensor,
        parent_pos: int,
    ):
        if curr == 0:
            # the first token generated by the target model, and thus it is always
            # accepted from the previous iteration
            is_accepted = True
        else:
            parent_bitmask = allocate_token_bitmask[parent_pos]
            current_token = draft_tokens[curr]
            if vocab_size and current_token >= vocab_size:
                is_accepted = False
            else:
                # 32 boolean bitmask values are packed into 32-bit integers
                is_accepted = (
                    parent_bitmask[current_token // 32] & (1 << (current_token % 32))
                ) != 0

        if is_accepted:
            if curr != 0:
                # Accept the current token
                grammar.accept_token(int(draft_tokens[curr]))
            if not grammar.is_terminated():
                # Generate the bitmask for the current token
                grammar.fill_vocab_mask(allocate_token_bitmask, curr)
                if retrieve_next_token[curr] != -1:
                    # Visit the child node
                    dfs(
                        int(retrieve_next_token[curr]),
                        retrieve_next_token,
                        retrieve_next_sibling,
                        curr,
                    )

            if curr != 0:
                # Rollback the current token
                grammar.rollback(1)

        if retrieve_next_sibling[curr] != -1:
            # Visit the sibling node
            dfs(
                int(retrieve_next_sibling[curr]),
                retrieve_next_token,
                retrieve_next_sibling,
                parent_pos,
            )

    dfs(0, retrieve_next_token, retrieve_next_sibling, -1)


def generate_token_bitmask(
    reqs: List[Req],
    verify_input: EagleVerifyInput,
    retrieve_next_token_cpu: torch.Tensor,
    retrieve_next_sibling_cpu: torch.Tensor,
    draft_tokens_cpu: torch.Tensor,
    vocab_size: int,
):
    """
    Generate the logit mask for structured output.
    Draft model's token can be either valid or invalid with respect to the grammar.
    We need to perform DFS to
    1. figure out which tokens are accepted by the grammar.
    2. if so, what is the corresponding logit mask.
    """

    num_draft_tokens = draft_tokens_cpu.shape[-1]

    allocate_token_bitmask = None
    assert len(reqs) == retrieve_next_token_cpu.shape[0]
    grammar = None
    for i, req in enumerate(reqs):
        if req.grammar is not None:
            if allocate_token_bitmask is None:
                allocate_token_bitmask = req.grammar.allocate_vocab_mask(
                    vocab_size=vocab_size,
                    batch_size=draft_tokens_cpu.numel(),
                    device="cpu",
                )
            grammar = req.grammar
            s = time.perf_counter()
            traverse_tree(
                retrieve_next_token_cpu[i],
                retrieve_next_sibling_cpu[i],
                draft_tokens_cpu[i],
                req.grammar,
                allocate_token_bitmask[
                    i * num_draft_tokens : (i + 1) * num_draft_tokens
                ],
                vocab_size=vocab_size,
            )
            tree_traverse_time = time.perf_counter() - s
            if tree_traverse_time > TREE_TRAVERSE_TIME_THRESHOLD:
                logger.warning(
                    f"Bit mask generation took {tree_traverse_time} seconds with "
                    f"grammar: {req.grammar}"
                )

    verify_input.grammar = grammar
    return allocate_token_bitmask


def load_token_map(token_map_path: str) -> List[int]:
    if not os.path.exists(token_map_path):
        repo_id = os.path.dirname(token_map_path)
        file_name = os.path.basename(token_map_path)

        cache_dir = None
        if envs.SGLANG_USE_MODELSCOPE.get():
            from modelscope.utils.file_utils import get_model_cache_root

            cached_repo_path = os.path.join(get_model_cache_root(), repo_id)
            if os.path.exists(cached_repo_path):
                cache_dir = cached_repo_path

        if cache_dir is None:
            if envs.SGLANG_USE_MODELSCOPE.get():
                from modelscope.hub.snapshot_download import (
                    snapshot_download as download_func,
                )
            else:
                download_func = snapshot_download
            cache_dir = download_func(
                repo_id,
                ignore_patterns=["*.bin", "*.safetensors"],
            )

        token_map_path = os.path.join(cache_dir, file_name)
    hot_token_id = torch.load(token_map_path, weights_only=True)
    return torch.tensor(hot_token_id, dtype=torch.int64)


@contextmanager
def draft_tp_context(tp_group: GroupCoordinator):
    # Draft model doesn't use dp and has its own tp group.
    # We disable mscclpp now because it doesn't support 2 comm groups.
    with patch_tensor_parallel_group(tp_group):
        yield


def spec_stage_span(name: str):
    """Profiler span for a coarse speculative-decoding stage (``draft`` /
    ``draft_extend`` / ``verify``).
    """
    return profile_range(name)


def move_accept_tokens_to_target_kvcache(
    batch: ScheduleBatch,
    accept_index: torch.Tensor,
    num_correct_drafts: torch.Tensor,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
):
    """
    Move accepted tokens (drafts + bonus) to the target KV cache.

    Args:
        batch: The batch to run.
        accept_index: The index of the accepted tokens (incl. bonus).
        num_correct_drafts: Per-req count of correct drafts (excludes bonus);
            seq_lens is advanced by ``num_correct_drafts + 1`` to cover the bonus slot.
    """
    bs = len(batch.seq_lens)
    device = batch.seq_lens.device
    # accept_index element count, NOT bs * num_draft_tokens: for topk > 1 the
    # tree exceeds the accepted chain, over-reading accept_index (illegal memory).
    size = bs * accept_index.shape[1]

    # fill_accept_out_cache_loc reads out_cache_loc[accept_index]; -1 sentinel ok.
    maybe_detect_oob(
        accept_index,
        -1,
        batch.out_cache_loc.size(0),
        "spec v2 move_accept_tokens accept_index",
    )

    tgt_cache_loc = torch.zeros(
        size,
        dtype=torch.int64,
        device=device,
    )
    accept_out_cache_loc = torch.zeros(size, dtype=torch.int64, device=device)
    assign_extend_cache_locs[(bs,)](
        batch.req_pool_indices,
        batch.req_to_token_pool.req_to_token,
        batch.seq_lens,
        batch.seq_lens + num_correct_drafts + 1,
        tgt_cache_loc,
        batch.req_to_token_pool.req_to_token.shape[1],
        next_power_of_2(bs),
    )
    fill_accept_out_cache_loc[(size,)](
        accept_index,
        batch.out_cache_loc,
        accept_out_cache_loc,
        next_power_of_2(size),
    )
    token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
        tgt_cache_loc, accept_out_cache_loc
    )


def prepare_mamba_track_for_verify(batch: ScheduleBatch) -> None:
    """Rebuild mamba track indices from reqs before a TARGET_VERIFY forward.

    Spec batches skip the refresh in prepare_for_decode, and filter/merge
    null these fields, so they must be rebuilt right before verify. Clearing
    the mask also keeps a stale extend-time mask from triggering in-forward
    tracking during TARGET_VERIFY; tracking is done in
    commit_mamba_states_after_verify instead.
    """
    if not get_global_server_args().enable_mamba_extra_buffer():
        return
    set_mamba_track_indices_from_reqs(batch)
    batch.mamba_track_mask = None
    batch.mamba_track_seqlens = None


def commit_mamba_states_after_verify(
    target_worker: TpModelWorker,
    batch: ScheduleBatch,
    accept_lens: torch.Tensor,
    accept_index: torch.Tensor,
    draft_token_num: int,
) -> None:
    """Commit accepted per-step mamba states into the persistent caches.

    During TARGET_VERIFY, hybrid linear attention backends keep per-step
    states in intermediate caches instead of advancing the persistent
    conv/ssm caches. After acceptance, the state of each request's last
    accepted step is committed back, plus the interval-crossing state used
    for prefix-cache tracking (mamba extra_buffer mode).

    No-op for models without mamba-style state or backends without the
    commit hook.
    """
    model_runner = target_worker.model_runner
    if model_runner.mambaish_config is None:
        return
    attn_backend = model_runner.attn_backend
    if not hasattr(attn_backend, "update_mamba_state_after_mtp_verify"):
        return

    bs = accept_lens.shape[0]
    # `accept_lens` already includes the bonus token (drafts + 1 per req).
    if not batch.forward_mode.is_idle() and accept_index.numel() > 0:
        accept_indices_offset = torch.arange(
            0,
            bs * draft_token_num,
            step=draft_token_num,
            dtype=accept_lens.dtype,
            device=accept_lens.device,
        )
        req_idx = torch.arange(bs, dtype=torch.int64, device=accept_lens.device)
        # Per-req tree step of the last accepted node, i.e. the step whose
        # mamba state to commit; reduces to accept_lens - 1 for topk == 1.
        last_correct_step_indices = (
            accept_index[req_idx, (accept_lens - 1).to(torch.int64)]
            - accept_indices_offset
        )

        if batch.mamba_track_indices is not None:
            # If after verify, the request's seq_lens has crossed a mamba track interval,
            # we need to update the mamba state for the request at the crossing point.
            seq_lens_pre_verify = batch.seq_lens
            seq_lens_post_verify = batch.seq_lens + accept_lens
            mamba_track_interval = get_global_server_args().mamba_track_interval
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != seq_lens_post_verify // mamba_track_interval
            )
            tracking_point = (
                seq_lens_post_verify // mamba_track_interval * mamba_track_interval
            )
            to_track_ith = torch.clamp(
                tracking_point - seq_lens_pre_verify - 1, min=0
            ).to(torch.int64)
            candidate_track_steps = (
                accept_index[req_idx, to_track_ith] - accept_indices_offset
            )
            mamba_steps_to_track = torch.where(
                to_track_mask,
                candidate_track_steps,
                torch.full_like(candidate_track_steps, -1),
            )
        else:
            mamba_steps_to_track = None

        attn_backend.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=model_runner.model,
        )


def spec_prepare_for_decode(batch: ScheduleBatch) -> None:
    """eagle/ngram share a stateless free function; dflash keeps stateful
    prep on its draft input -- the dispatcher routes.
    """
    if batch.spec_algorithm.is_dflash():
        batch.spec_info.prepare_for_decode(batch)
    else:
        from sglang.srt.speculative.eagle_utils import eagle_prepare_for_decode

        eagle_prepare_for_decode(batch)
