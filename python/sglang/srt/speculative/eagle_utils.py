from __future__ import annotations

import logging
import math
from enum import IntEnum
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_alloc_reserve_per_decode,
    get_last_loc,
)
from sglang.srt.environ import envs
from sglang.srt.utils import is_cuda, is_hip, is_musa, is_npu
from sglang.srt.utils.async_probe import maybe_detect_oob

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.eagle_info import EagleVerifyInput

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_musa = is_musa()

if _is_cuda or _is_hip or _is_musa:
    from sgl_kernel import (
        build_tree_kernel_efficient as sgl_build_tree_kernel_efficient,
    )


_SPEC_DEPTH_ALPHA: float = envs.SGLANG_SPEC_DEPTH_ALPHA.get()
if _SPEC_DEPTH_ALPHA != 0.0:
    logging.getLogger(__name__).warning(
        "[spec] depth-normalized rerank ACTIVE: SGLANG_SPEC_DEPTH_ALPHA=%.3f "
        "(score/depth^alpha; ancestor-closure repaired post-topk)",
        _SPEC_DEPTH_ALPHA,
    )

_SPEC_ENABLE_SINGLE_BRANCH_VERIFY: bool = (
    envs.SGLANG_SPEC_ENABLE_SINGLE_BRANCH_VERIFY.get()
)
if _SPEC_ENABLE_SINGLE_BRANCH_VERIFY:
    logging.getLogger(__name__).warning(
        "[spec] single-branch verify ACTIVE: SGLANG_SPEC_ENABLE_SINGLE_BRANCH_VERIFY=1 "
        "(global rerank keeps only the single highest-value branch; target verifies "
        "one plain chain, not the tree)"
    )

# ---- debug: verify-budget depth allocation (SGLANG_DEBUG_SPEC_TREE_DEPTH) --------------
# Accumulate a {depth: count} histogram of the nodes the global rerank keeps, so we can
# see whether the B verify nodes go shallow-bushy or deep-narrow vs the pt_tree harness.
_DEPTH_HIST: dict = {}
_DEPTH_CALLS = 0


def _record_retained_depths(top_scores_index: torch.Tensor, topk: int, nlevels: int):
    """top_scores_index: (b, B-1) indices into the flattened score_list. Map each to its
    tree depth and fold into a global histogram; dump incrementally (worker is a subprocess).
    Flattened layout after cat+flatten(1): dim1 has 1 + topk*(nlevels-1) rows (row 0 = depth 1,
    then topk rows per subsequent level), each row topk wide -> col c has row = c // topk."""
    global _DEPTH_CALLS
    if topk <= 0:
        return
    # This does D2H syncs (.tolist) + dynamic-shape ops (unique) — both illegal while a CUDA
    # graph is capturing, and during graph *replay* this Python never runs. So it only yields
    # real data in eager mode (--disable-cuda-graph); skip entirely under capture to be safe.
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    idx = top_scores_index.reshape(-1)
    row = torch.div(idx, topk, rounding_mode="floor")
    depth = torch.where(row == 0, row.new_ones(()), torch.div(row - 1, topk, rounding_mode="floor") + 2)
    vals, counts = torch.unique(depth, return_counts=True)
    for v, c in zip(vals.tolist(), counts.tolist()):
        _DEPTH_HIST[v] = _DEPTH_HIST.get(v, 0) + int(c)
    _DEPTH_CALLS += 1
    if _DEPTH_CALLS % 32 == 0:
        _dump_retained_depths()


def _dump_retained_depths():
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
    except Exception:
        pass
    out = envs.SGLANG_DEBUG_SPEC_TREE_DEPTH_OUT.get()
    if not out:
        return
    tot = sum(_DEPTH_HIST.values()) or 1
    mean_depth = sum(d * c for d, c in _DEPTH_HIST.items()) / tot
    try:
        import json

        with open(out, "w") as f:
            json.dump(
                {"calls": _DEPTH_CALLS, "mean_retained_depth": mean_depth,
                 "hist": {str(d): _DEPTH_HIST[d] for d in sorted(_DEPTH_HIST)}}, f)
    except Exception:
        pass


def per_step_draft_out_cache_loc(
    out_cache_loc: torch.Tensor,
    batch_size: int,
    topk: int,
    num_steps: int,
) -> torch.Tensor:
    """Per-step slice of the multi-step EAGLE draft out_cache_loc buffer.

    Single source of truth for the layout shared by EagleWorkerV2.draft_forward
    (per-step write target) and DeepseekV4AttnBackend (per-step compression
    write target baked into metadata).
    """
    expected = batch_size * topk * num_steps
    assert out_cache_loc.shape[0] == expected, (
        f"out_cache_loc.shape[0]={out_cache_loc.shape[0]} != "
        f"batch_size * topk * num_steps = {batch_size}*{topk}*{num_steps}={expected}"
    )
    return (
        out_cache_loc.view(batch_size, topk, num_steps)
        .permute(2, 0, 1)
        .reshape(num_steps, -1)
    )


def _eagle_prefill_tail_tokens(
    batch: ScheduleBatch, next_token_ids: torch.Tensor
) -> torch.Tensor:
    """Per-seq tail token for EAGLE prefill rotation; uses next prompt token for
    non-final chunks (chunked-prefill chain consistency, see PR #26329)."""
    tail_tokens = next_token_ids.to(batch.input_ids.dtype)
    next_prompt_token = batch.chunked_req_next_prompt_token
    if next_prompt_token is not None:
        for i, r in enumerate(batch.reqs):
            if r is batch.chunked_req:
                tail_tokens = tail_tokens.clone()
                tail_tokens[i] = next_prompt_token
                break
    return tail_tokens


def _enforce_ancestor_closure(
    top_scores_index: torch.Tensor,
    parent_list: torch.Tensor,
    rerank_scores: torch.Tensor,
    topk: int,
    nlevels: int,
) -> torch.Tensor:
    """Repair the top-B selection to be ancestor-closed — pure GPU tensor ops.

    Depth normalization can select a deep node whose parent was not selected
    (child normalized score > parent normalized score for easy tokens). The CUDA
    tree kernel requires every selected node's parent to also be selected.

    parent_list encoding (from the CUDA kernel): parent of flat index p is
    parent_list[b, p // topk].  Row 0 maps to -1 (root = bonus token, always
    valid). Rows 1.. map to flat indices of depth-1, depth-2, ... parents.

    Phase 1 — add missing parents: up to nlevels passes, each pass finds all
    selected nodes whose parent is absent and scatter-adds the parents into the
    selection mask.  No GPU→CPU sync (no .any() branch).
    Phase 2 — trim excess: remove the lowest-scoring leaves (nodes not
    referenced as parents) until the count is back to B-1.  Also no sync.
    """
    b, B_minus_1 = top_scores_index.shape
    total = rerank_scores.shape[1]
    device = top_scores_index.device

    # parent_flat_all[b, p] = parent_list[b, p // topk] — parent flat index for every node
    all_pos = torch.arange(total, device=device)
    parent_flat_all = parent_list[:, all_pos // topk]  # (b, total)
    parent_flat_clamped = parent_flat_all.clamp(min=0)  # -1 → 0 (harmless sentinel)
    is_root_child = parent_flat_all < 0  # depth-1 nodes whose parent is the bonus token

    # Represent the current selection as a bool mask (b, total)
    sel_mask = torch.zeros(b, total, dtype=torch.bool, device=device)
    sel_mask.scatter_(1, top_scores_index, True)

    # Pre-allocate scratch tensors reused across iterations
    add_count = torch.zeros(b, total, dtype=torch.long, device=device)
    in_use = torch.zeros(b, total, dtype=torch.long, device=device)

    # Phase 1: add missing ancestors (up to nlevels passes, no CPU sync)
    for _ in range(nlevels):
        parent_in_sel = sel_mask.gather(1, parent_flat_clamped)
        violators = sel_mask & ~is_root_child & ~parent_in_sel
        # scatter_add_ to parents of violators; non-violators point to position 0
        # (harmless: position-0 nodes are depth-1 root children, already valid)
        add_count.fill_(0)
        add_count.scatter_add_(
            1,
            torch.where(violators, parent_flat_clamped, torch.zeros_like(parent_flat_clamped)),
            violators.long(),
        )
        sel_mask = sel_mask | add_count.bool()

    # Phase 2: trim excess back to B-1 (at most nlevels passes, no CPU sync)
    for _ in range(nlevels):
        # A leaf is a selected node that is not the parent of any other selected node
        in_use.fill_(0)
        in_use.scatter_add_(
            1,
            torch.where(
                sel_mask & ~is_root_child,
                parent_flat_clamped,
                torch.zeros_like(parent_flat_clamped),
            ),
            (sel_mask & ~is_root_child).long(),
        )
        is_leaf = sel_mask & (in_use == 0)

        # needs_trim[b] is True when that batch item still has excess nodes
        needs_trim = sel_mask.long().sum(dim=1) > B_minus_1  # (b,)

        # For items that need trimming, find the lowest-scoring leaf; for others
        # point to position 0 (will produce a spurious remove masked out below)
        leaf_scores = torch.where(
            is_leaf & needs_trim.unsqueeze(1),
            rerank_scores,
            rerank_scores.new_full((), float("inf")),
        )
        worst_leaf = leaf_scores.argmin(dim=1)  # (b,)

        remove = torch.zeros(b, total, dtype=torch.bool, device=device)
        remove.scatter_(1, worst_leaf.unsqueeze(1), needs_trim.unsqueeze(1))
        sel_mask = sel_mask & ~remove

    # Convert the repaired mask back to sorted indices
    _, top_scores_index_new = sel_mask.float().topk(B_minus_1, dim=1, sorted=False)
    return torch.sort(top_scores_index_new).values


def _select_single_branch(
    rerank_scores: torch.Tensor,
    last_level_width: int,
    parent_list: torch.Tensor,
    topk: int,
    nlevels: int,
) -> torch.Tensor:
    """Restrict the global rerank to the SINGLE highest-value branch (one node
    per level) instead of the top-(B-1) nodes anywhere in the tree. Lets the
    draft EXPAND wide (topk>1) while VERIFY only ever sees a plain chain: the
    target does standard (non-tree) speculative verification over one
    candidate sequence, not a tree-masked forward over many.

    The single best branch is the ancestor chain of the highest-scoring node
    at the FINAL expand level. This is provably the true best-value path end
    to end: value is non-increasing along any path, and every expand step's
    fast_topk already performs a value-sorted cut across the FLATTENED
    (all-parents-pooled) candidates -- so the running max-score node is always
    carried into the next level's frontier, and no other branch's descendant
    can ever overtake it later. Returns (b, nlevels) sorted indices -- exactly
    fills a nlevels-1 draft-token budget (see the num_draft_token assert at
    the call site), no padding needed."""
    b, total = rerank_scores.shape
    device = rerank_scores.device

    offset = total - last_level_width
    leaf = offset + rerank_scores[:, offset:].argmax(dim=1)  # (b,) global flat index

    # parent_flat_all[b, p] = flat index of p's parent (or <0 sentinel for a
    # depth-1 / root-child node) -- same lookup `_enforce_ancestor_closure` uses.
    # nlevels==1 has no ancestor table at all (parent_list is empty): every
    # candidate is trivially a root child, so skip the lookup entirely.
    if parent_list.numel() > 0:
        all_pos = torch.arange(total, device=device)
        parent_flat_all = parent_list[:, all_pos // topk]  # (b, total)
        is_root_child = parent_flat_all < 0
        parent_flat_clamped = parent_flat_all.clamp(min=0)
    else:
        is_root_child = torch.ones(b, total, dtype=torch.bool, device=device)
        parent_flat_clamped = torch.zeros(b, total, dtype=torch.long, device=device)

    sel_mask = torch.zeros(b, total, dtype=torch.bool, device=device)
    cur = leaf
    for _ in range(nlevels):
        sel_mask.scatter_(1, cur.unsqueeze(1), True)
        at_root = is_root_child.gather(1, cur.unsqueeze(1)).squeeze(1)
        parent = parent_flat_clamped.gather(1, cur.unsqueeze(1)).squeeze(1)
        cur = torch.where(at_root, cur, parent)

    _, top_scores_index = sel_mask.float().topk(nlevels, dim=1, sorted=False)
    return torch.sort(top_scores_index).values


def organize_draft_results(
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    num_draft_token: int,
):
    # topk per step = last dim of the level-0 score entry (b, 1, topk).
    _dbg_topk = score_list[0].shape[-1] if score_list else 0
    _dbg_nlevels = len(score_list)
    if _SPEC_DEPTH_ALPHA != 0.0:
        # Depth-normalize: divide each level's cumulative score by depth^alpha.
        # score_list[i] covers depth (i+1) nodes. Dividing rewards deeper paths
        # whose raw cumulative score is suppressed by the length of the chain.
        # In log domain (SGLANG_SPEC_ENABLE_LOG_DOMAIN=1) with alpha=1 this is
        # the average log-prob per step (geometric mean), giving each depth equal
        # standing in the global rerank. The indices into ss_token_list are
        # unchanged — only the ranking key changes.
        rerank_scores = torch.cat(
            [s / ((i + 1) ** _SPEC_DEPTH_ALPHA) for i, s in enumerate(score_list)],
            dim=1,
        ).flatten(1)
    else:
        rerank_scores = torch.cat(score_list, dim=1).flatten(1)
    ss_token_list = torch.cat(token_list, dim=1)

    # Build parent_list up front -- both the ancestor-closure repair and
    # single-branch selection need the ancestor-lookup table.
    if len(parents_list) > 1:
        parent_list = torch.cat(parents_list[:-1], dim=1)
    else:
        batch_size = parents_list[0].shape[0]
        parent_list = torch.empty(
            batch_size, 0, dtype=torch.long, device=parents_list[0].device
        )

    if _SPEC_ENABLE_SINGLE_BRANCH_VERIFY:
        assert num_draft_token - 1 == _dbg_nlevels, (
            "SGLANG_SPEC_ENABLE_SINGLE_BRANCH_VERIFY requires "
            "speculative_num_draft_tokens == speculative_num_steps + 1 (the single "
            f"chosen branch has exactly one node per level), got num_draft_token="
            f"{num_draft_token} speculative_num_steps={_dbg_nlevels}"
        )
        last_level_width = score_list[-1].flatten(1).shape[1]
        top_scores_index = _select_single_branch(
            rerank_scores, last_level_width, parent_list, _dbg_topk, _dbg_nlevels
        )
    else:
        top_scores = torch.topk(rerank_scores, num_draft_token - 1, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values

        # Depth normalization can select deep nodes whose parent was not selected
        # (child normalized score > parent normalized score for easy tokens). Fix
        # before passing to the CUDA tree kernel which requires ancestor-closure.
        if _SPEC_DEPTH_ALPHA != 0.0 and parent_list.numel() > 0:
            top_scores_index = _enforce_ancestor_closure(
                top_scores_index, parent_list, rerank_scores, _dbg_topk, _dbg_nlevels
            )
            top_scores_index = torch.sort(top_scores_index).values

    if envs.SGLANG_DEBUG_SPEC_TREE_DEPTH.get():
        _record_retained_depths(top_scores_index, _dbg_topk, _dbg_nlevels)
    maybe_detect_oob(
        top_scores_index,
        0,
        ss_token_list.shape[1],
        "organize_draft_results: top_scores_index OOB for gather on ss_token_list",
    )
    draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

    return parent_list, top_scores_index, draft_tokens


class TreeMaskMode(IntEnum):
    FULL_MASK = 0
    QLEN_ONLY = 1
    QLEN_ONLY_BITPACKING = 2


def build_tree_kernel_efficient(
    bonus_tokens: torch.Tensor,
    parent_list: List[torch.Tensor],
    top_scores_index: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_sum: int,
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
    tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK,
    tree_mask_buf: Optional[torch.Tensor] = None,
    position_buf: Optional[torch.Tensor] = None,
):
    draft_tokens = torch.cat((bonus_tokens.unsqueeze(1), draft_tokens), dim=1).flatten()

    # seq_lens_sum == sum(seq_lens); seq_lens: sequence length without draft tokens
    bs = seq_lens.numel()
    device = seq_lens.device
    # e.g. for bs=1, tree_mask: num_draft_token, seq_lens_sum + num_draft_token (flattened)
    # where each row indicates the attending pattern of each draft token
    # if use_partial_packed_tree_mask is True, tree_mask: num_draft_token (flattened, packed)
    if tree_mask_buf is not None:
        tree_mask = tree_mask_buf
        if tree_mask_mode == TreeMaskMode.QLEN_ONLY:
            tree_mask.fill_(True)
        elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
            tree_mask.fill_(0)
        elif tree_mask_mode == TreeMaskMode.FULL_MASK:
            tree_mask.fill_(True)
        else:
            raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY:
        tree_mask = torch.full(
            (num_verify_tokens * bs * num_verify_tokens,),
            True,
            dtype=torch.bool,
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
        packed_dtypes = [torch.uint8, torch.uint16, torch.uint32]
        packed_dtype_idx = int(math.ceil(math.log2((num_verify_tokens + 7) // 8)))
        tree_mask = torch.zeros(
            (num_verify_tokens * bs,),
            dtype=packed_dtypes[packed_dtype_idx],
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.FULL_MASK:
        tree_mask = torch.full(
            (
                seq_lens_sum * num_verify_tokens
                + num_verify_tokens * num_verify_tokens * bs,
            ),
            True,
            device=device,
        )
    else:
        raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")

    # TODO: make them torch.empty and fuse them into `sgl_build_tree_kernel`
    retrieve_buf = torch.full(
        (3, bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    retrieve_index, retrieve_next_token, retrieve_next_sibling = retrieve_buf
    # position: where each token belongs to
    # e.g. if depth of each draft token is [0, 1, 1, 2] and the prompt length is 7
    # then, positions = [7, 8, 8, 9]
    if position_buf is not None:
        positions = position_buf
    else:
        positions = torch.empty(
            (bs * num_verify_tokens,), device=device, dtype=torch.long
        )

    if _is_npu:
        torch.ops.npu.build_tree_kernel_efficient(
            parent_list.to(dtype=torch.int64),
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    else:
        sgl_build_tree_kernel_efficient(
            parent_list,
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    return (
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        draft_tokens,
    )


def verify_tree_greedy_func(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
    topk: int = -1,
):
    if _is_cuda or _is_hip or _is_musa:
        from sgl_kernel import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=accept_token_num,  # mutable
            candidates=candidates,
            # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )

    elif _is_npu:
        from sgl_kernel_npu.sample.verify_tree_greedy import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )
    return predicts, accept_index, accept_token_num


def get_draft_input_from_target_hidden_dim(model_runner: ModelRunner) -> int:
    """Width of the target hidden states fed into the draft model.

    This is the single source of truth and is derived entirely from config: for
    EAGLE3 aux mode the draft consumes `num_aux` concatenated target layers
    (each `target_hidden_size` wide); every other arch consumes the per-layer
    `spec_hidden_size`.

    Do NOT read this off a draft projection's `in_features` (e.g. an `fc`
    layer): that width is arch-specific.

    Note: read entirely from the *draft* `model_runner`'s config. The non-aux
    branch assumes the draft's `spec_hidden_size` equals the target hidden width
    fed to the draft (true for standard EAGLE, where the draft mirrors the
    target hidden size); aux mode reads the explicit `target_hidden_size`.
    """
    model_config = model_runner.model_config
    hf_config = model_config.hf_config
    eagle_config = getattr(hf_config, "eagle_config", None) or {}
    get_eagle_config = (
        eagle_config.get
        if isinstance(eagle_config, dict)
        else lambda key, default=None: getattr(eagle_config, key, default)
    )
    use_aux = get_eagle_config("use_aux_hidden_state", True)
    spec_algorithm = model_runner.spec_algorithm

    if not (spec_algorithm is not None and spec_algorithm.is_eagle3() and use_aux):
        return model_config.spec_hidden_size

    target_hidden = getattr(hf_config, "target_hidden_size", None)
    if target_hidden is None:
        target_hidden = model_config.hidden_size
    num_aux = getattr(hf_config, "num_aux_hidden_states", None)
    if num_aux is None:
        layer_ids = get_eagle_config("eagle_aux_hidden_state_layer_ids", None)
        if layer_ids is None:
            layer_ids = getattr(hf_config, "eagle_aux_hidden_state_layer_ids", None)
        num_aux = len(layer_ids) if layer_ids else 3
    return target_hidden * num_aux


def draft_carries_hidden_states(model_runner: ModelRunner) -> bool:
    """Whether the draft loop has to carry the draft model's hidden states.

    EAGLE-family drafts consume them as input, so they always do. A STANDALONE
    draft is a plain LM that takes token ids and normally discards its hidden
    states end-to-end -- except that the agreement head's largest feature IS the
    gate's hidden state, so --speculative-agreement-head turns the capture back
    on. The model computes the tensor either way; what capturing costs is the
    graph buffer and the copy.
    """
    if not model_runner.spec_algorithm.is_standalone():
        return True
    return model_runner.server_args.speculative_agreement_head is not None


def get_draft_recurrent_hidden_state_spec(
    model_runner: ModelRunner,
) -> tuple[Optional[int], Optional[torch.dtype]]:
    """Return hidden_states width/dtype carried between draft decode steps."""
    if not draft_carries_hidden_states(model_runner):
        return None, None
    return model_runner.model_config.spec_hidden_size, model_runner.model_config.dtype


def eagle_prepare_for_verify(
    verify_input: EagleVerifyInput,
    req_to_token_pool: ReqToTokenPool,
    batch: ScheduleBatch,
    target_worker: TpModelWorker,
):
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardBatch,
        ForwardMode,
    )
    from sglang.srt.speculative.spec_utils import prepare_mamba_track_for_verify
    from sglang.srt.speculative.triton_ops.cache_locs import (
        assign_extend_cache_locs_func,
    )

    if not batch.forward_mode.is_idle():
        # Assign cache locations
        bs = len(batch.req_pool_indices)
        batch.input_ids = verify_input.draft_token
        maybe_detect_oob(
            batch.input_ids,
            0,
            batch.model_config.vocab_size,
            "v2 prepare_for_verify input_ids",
        )
        device = batch.device
        batch.out_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=req_to_token_pool.req_to_token,
            start_offset=batch.seq_lens,
            end_offset=batch.seq_lens + verify_input.draft_token_num,
            batch_size=bs,
            draft_token_num=verify_input.draft_token_num,
            device=device,
        )

        prepare_mamba_track_for_verify(batch)

        # TBO's split_spec_info reads these; no-verify-sync leaves both None.
        verify_input.seq_lens_cpu = batch.seq_lens_cpu
        verify_input.seq_lens_sum = (
            int(batch.seq_lens_cpu.sum()) if batch.seq_lens_cpu is not None else None
        )

    # Get a forward batch
    batch.forward_mode = (
        ForwardMode.IDLE if batch.forward_mode.is_idle() else ForwardMode.TARGET_VERIFY
    )
    capture_mode = (
        CaptureHiddenMode.NULL
        if target_worker.model_runner.spec_algorithm.is_standalone()
        else CaptureHiddenMode.FULL
    )
    batch.capture_hidden_mode = capture_mode
    verify_forward_batch = ForwardBatch.init_new(batch, target_worker.model_runner)

    # Run attention backend plan and cuda graph preparation
    can_run_cuda_graph = bool(
        target_worker.model_runner.decode_cuda_graph_runner
        and target_worker.model_runner.decode_cuda_graph_runner.can_run_graph(
            verify_forward_batch
        )
    )
    if can_run_cuda_graph:
        target_worker.model_runner.decode_cuda_graph_runner.load_batch(
            verify_forward_batch
        )
        verify_forward_batch.mark_forward_metadata_ready()
    # Non-cuda-graph: defer init to forward_extend, which runs after
    # `_forward_raw -> prepare_mlp_sync_batch` pads the batch. Initing
    # here would use pre-pad shapes and trip DSv4 indexer shape match.

    return verify_forward_batch, can_run_cuda_graph


def eagle_sample(
    verify_input: EagleVerifyInput,
    batch: ScheduleBatch,
    logits_output: LogitsProcessorOutput,
    vocab_mask: torch.Tensor = None,
):
    """
    Verify and find accepted tokens based on logits output and batch
    (which contains spec decoding information).
    """
    import torch.nn.functional as F

    from sglang.srt.distributed import get_tp_group
    from sglang.srt.layers.dp_attention import (
        get_attention_tp_group,
        is_dp_attention_enabled,
    )
    from sglang.srt.sampling.penaltylib.repetition_penalty import (
        apply_scaling_penalties,
    )
    from sglang.srt.server_args import get_global_server_args
    from sglang.srt.speculative.spec_utils import (
        SIMULATE_ACC_LEN,
        generate_simulated_accept_index,
    )
    from sglang.srt.utils.async_probe import maybe_detect_nan, sanitize_nan_logits

    device = batch.device
    if batch.forward_mode.is_idle():
        predict = torch.empty(0, dtype=torch.int32, device=device)
        num_correct_drafts = torch.empty(0, dtype=torch.int32, device=device)
        accept_index = torch.empty(0, dtype=torch.int32, device=device)
        return predict, num_correct_drafts, accept_index

    bs = len(batch.seq_lens)
    sampling_info = batch.sampling_info
    next_token_logits = logits_output.next_token_logits

    sanitize_nan_logits(next_token_logits, "verify: target model logits")

    # Apply penalty
    # This is a relaxed version of penalties for speculative decoding.
    if sampling_info.acc_additive_penalties is not None:
        next_token_logits.add_(
            torch.repeat_interleave(
                sampling_info.acc_additive_penalties,
                verify_input.draft_token_num,
                dim=0,
            )
        )
    if sampling_info.acc_scaling_penalties is not None:
        apply_scaling_penalties(
            next_token_logits,
            torch.repeat_interleave(
                sampling_info.acc_scaling_penalties, verify_input.draft_token_num, dim=0
            ),
        )
    if sampling_info.logit_bias is not None:
        next_token_logits.add_(
            torch.repeat_interleave(
                sampling_info.logit_bias, verify_input.draft_token_num, dim=0
            )
        )

    # Apply grammar mask if provided
    if vocab_mask is not None:
        assert verify_input.grammar is not None
        verify_input.grammar.apply_vocab_mask(
            logits=next_token_logits, vocab_mask=vocab_mask
        )

    candidates = verify_input.draft_token.reshape(bs, verify_input.draft_token_num)
    predict_shape = list(next_token_logits.shape)[:-1]
    predict = torch.zeros(predict_shape, dtype=torch.int32, device=device).flatten()
    accept_index = torch.full(
        (bs, verify_input.max_tree_depth), -1, dtype=torch.int32, device=device
    )
    num_correct_drafts = torch.empty((bs,), dtype=torch.int32, device=device)

    # Sample tokens
    if sampling_info.is_all_greedy or _is_npu or _is_hip:
        target_predict = torch.argmax(next_token_logits, dim=-1)
        target_predict = target_predict.reshape(bs, verify_input.draft_token_num)
        predict, accept_index, num_correct_drafts = verify_tree_greedy_func(
            predicts=predict,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=num_correct_drafts,  # mutable
            candidates=candidates,
            retrieve_index=verify_input.retrieve_index,
            retrieve_next_token=verify_input.retrieve_next_token,
            retrieve_next_sibling=verify_input.retrieve_next_sibling,
            target_predict=target_predict,
            topk=verify_input.tree_topk,
        )
    else:
        from sgl_kernel import (
            top_k_renorm_prob,
            top_p_renorm_prob,
            tree_speculative_sampling_target_only,
        )

        from sglang.srt.speculative.reject_sampling import (
            chain_speculative_sampling_triton,
        )

        use_rejection_sampling = (
            get_global_server_args().speculative_use_rejection_sampling
        )

        # Apply temperature and get target probs
        expanded_temperature = torch.repeat_interleave(
            sampling_info.temperatures, verify_input.draft_token_num, dim=0
        )  # (bs * num_draft_tokens, 1)

        target_probs = F.softmax(
            next_token_logits / expanded_temperature, dim=-1
        )  # (bs * num_draft_tokens, vocab_size)
        maybe_detect_nan(target_probs, "v2 verify: target_probs after softmax")
        target_probs = top_k_renorm_prob(
            target_probs,
            torch.repeat_interleave(
                sampling_info.top_ks, verify_input.draft_token_num, dim=0
            ),
        )  # (bs * num_draft_tokens, vocab_size)
        maybe_detect_nan(target_probs, "v2 verify: target_probs after top_k_renorm")
        target_probs = top_p_renorm_prob(
            target_probs,
            torch.repeat_interleave(
                sampling_info.top_ps, verify_input.draft_token_num, dim=0
            ),
        )
        maybe_detect_nan(target_probs, "v2 verify: target_probs after top_p_renorm")
        target_probs = target_probs.reshape(bs, verify_input.draft_token_num, -1)
        draft_probs = (
            verify_input.draft_probs
            if use_rejection_sampling
            else torch.zeros_like(target_probs)
        )
        # Defense-in-depth behind the spec_hook startup allowlist: validate the
        # actual kernel inputs (catches draft_probs plumbing regressions or a
        # startup guard bypassed by a worker subclass) before the Triton kernel.
        if use_rejection_sampling and (
            draft_probs is None or draft_probs.shape[-1] != target_probs.shape[-1]
        ):
            raise ValueError(
                "Rejection sampling requires a target-vocab draft proposal "
                "distribution; the current speculative algorithm/draft worker "
                "does not produce one (draft_probs missing or vocab-mismatched)."
            )

        # coins for rejection sampling
        coins = torch.rand_like(candidates, dtype=torch.float32, device=device)
        # coins for final sampling
        coins_for_final_sampling = torch.rand((bs,), dtype=torch.float32, device=device)

        sampling_fn = (
            chain_speculative_sampling_triton
            if use_rejection_sampling
            else tree_speculative_sampling_target_only
        )
        sampling_fn(
            predicts=predict,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=num_correct_drafts,  # mutable
            candidates=candidates,
            # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
            retrive_index=verify_input.retrieve_index,
            retrive_next_token=verify_input.retrieve_next_token,
            retrive_next_sibling=verify_input.retrieve_next_sibling,
            uniform_samples=coins,
            uniform_samples_for_final_sampling=coins_for_final_sampling,
            target_probs=target_probs,
            draft_probs=draft_probs,
            threshold_single=get_global_server_args().speculative_accept_threshold_single,
            threshold_acc=get_global_server_args().speculative_accept_threshold_acc,
            deterministic=True,
        )

        # Sync sampling results across TP ranks: different GPUs may
        # produce slightly different target_probs due to floating-point
        # non-determinism in softmax/top_k/top_p, causing different
        # sampled tokens. Broadcast from rank 0 to ensure consistency.
        tp_group = (
            get_attention_tp_group() if is_dp_attention_enabled() else get_tp_group()
        )
        if tp_group.world_size > 1:
            tp_group.broadcast(predict, src=0)
            tp_group.broadcast(accept_index, src=0)
            tp_group.broadcast(num_correct_drafts, src=0)

    if SIMULATE_ACC_LEN > 0:
        # Do simulation. The helper builds (and returns) a replacement
        # accept_index of width spec_steps + 1, so pass max_tree_depth - 1
        # to keep the simulated width identical to the real one.
        accept_index = generate_simulated_accept_index(
            accept_index=accept_index,
            predict=predict,  # mutable
            num_correct_drafts=num_correct_drafts,  # mutable
            simulate_acc_len=SIMULATE_ACC_LEN,
            bs=bs,
            spec_steps=verify_input.max_tree_depth - 1,
        )

    # `num_correct_drafts` stays drafts-only inside this function; the returned
    # tensor includes the trailing/bonus token via out-of-place +1 so the
    # name no longer flips semantics mid-function (naming doc C2).
    return predict, num_correct_drafts + 1, accept_index


def eagle_prepare_for_decode(batch: ScheduleBatch):
    batch.maybe_evict_swa()

    from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

    bs = batch.batch_size()

    # Accumulate penalty
    # This is a relaxed version of penalties for speculative decoding.
    if batch.sampling_info.penalizer_orchestrator.is_required:
        batch.cumulate_penalty_output_tokens()

    page_size = batch.token_to_kv_pool_allocator.page_size
    double_alloc = get_alloc_reserve_per_decode()

    cur_kv_lens = [0] * bs
    nxt_kv_lens = [0] * bs
    num_needed_tokens = 0
    for i, r in enumerate(batch.reqs):
        cur = r.kv_allocated_len
        # max(cur, ...) clamps so adaptive downswitch cannot make nxt < cur.
        # kv_committed_len is honest (bonus committed in resolve, not here),
        # so it lags batch.seq_lens by ~1 verify in overlap; 2*alloc absorbs.
        nxt = max(cur, r.kv_committed_len + double_alloc)
        cur_kv_lens[i] = cur
        nxt_kv_lens[i] = nxt
        num_needed_tokens += nxt - cur
        r.kv_allocated_len = nxt
        r.decode_batch_idx += 1

    cur_kv_lens_cpu = torch.tensor(cur_kv_lens, dtype=torch.int32, device="cpu")
    nxt_kv_lens_cpu = torch.tensor(nxt_kv_lens, dtype=torch.int32, device="cpu")

    # Fail fast if the page>1 + topk>1 draft over-allocation
    # (get_alloc_reserve_per_decode) outgrows the req_to_token row: the write below
    # would OOB and free would leak KV. The row is widened to hold it in _init_pools
    # (PR #26972); fail here with a clear error, not on a later cryptic CUDA assert.
    from sglang.srt.server_args import get_global_server_args

    if page_size > 1 and (get_global_server_args().speculative_eagle_topk or 1) > 1:
        max_alloc_len = int(nxt_kv_lens_cpu.max())
        row_width = batch.req_to_token_pool.req_to_token.shape[1]
        assert max_alloc_len <= row_width, (
            f"spec v2 page>1 topk>1 draft over-allocation ({max_alloc_len}) exceeds "
            f"req_to_token row width ({row_width}); page_size={page_size}. Widen the "
            f"row to hold committed + get_alloc_reserve_per_decode (PR #26972)."
        )

    # non_blocking H2D: a blocking .to() syncs the schedule stream, which the WAR
    # barrier has chained to the prev forward -> host stalls a full forward.
    cur_kv_lens_device = cur_kv_lens_cpu.to(device=batch.device, non_blocking=True)
    nxt_kv_lens_device = nxt_kv_lens_cpu.to(device=batch.device, non_blocking=True)
    if page_size == 1:
        out_cache_loc = alloc_token_slots(batch.tree_cache, num_needed_tokens)
    else:
        last_loc = get_last_loc(
            batch.req_to_token_pool.req_to_token,
            batch.req_pool_indices,
            cur_kv_lens_device,
        )
        out_cache_loc = alloc_paged_token_slots_extend(
            batch.tree_cache,
            cur_kv_lens_device,
            cur_kv_lens_cpu,
            nxt_kv_lens_device,
            nxt_kv_lens_cpu,
            last_loc,
            num_needed_tokens,
        )

    assign_req_to_token_pool_func(
        batch.req_pool_indices,
        batch.req_to_token_pool.req_to_token,
        cur_kv_lens_device,
        nxt_kv_lens_device,
        out_cache_loc,
        bs,
    )
