#!/usr/bin/env python3
"""The two things draft_forward relies on that the earlier tests do not cover.

1. COLUMN ALIGNMENT. The head ranks its own top-16 of the RAW draft logits;
   EAGLE expands the top-k of `renorm_draft_probs`. Those are different tensors
   and `speculative_eagle_topk` need not equal the checkpoint's top_k, so column
   j of one is not column j of the other. Getting this wrong attaches every
   agreement score to the wrong edge, which does not crash and does not look
   like a bug -- it looks like "the head does not help in serving".

2. DROP INHERITANCE. select_top_k_tokens prunes structurally by writing -1e30
   into the score tensor (SGLANG_SPEC_EXPAND_CHILDREN, nucleus mass cut). Those
   decisions exist ONLY in the draft scores, so swapping the pool wholesale
   would resurrect branches EAGLE deleted -- changing the tree's shape, not just
   its ranking.

Run:  python test_agreement_alignment.py
"""
from __future__ import annotations
import sys
import torch

sys.path.insert(0, "/home/ubuntu/pratap/GithubProjects/sglang/python")
from sglang.srt.speculative.agreement_head import (
    AgreementTreeScorer,
    LOG_MIN_AGREEMENT,
    align_log_agreement,
)
from sglang.srt.speculative.spec_utils import _cap_children, _select_top_k_tokens_later

torch.manual_seed(0)
ok = True


def check(name, cond):
    global ok
    print(f"  {'OK  ' if cond else 'FAIL'} {name}")
    if not cond:
        ok = False


# --- 1. alignment -------------------------------------------------------
print("column alignment:")
N, K = 5, 16
head_ids = torch.stack([torch.randperm(1000)[:K] for _ in range(N)])
log_agree = torch.randn(N, K)

# a) identity: selecting the head's own ids in its own order is a no-op.
same = align_log_agreement(log_agree, head_ids, head_ids)
check("identity selection is a no-op", torch.equal(same, log_agree))

# b) a permuted / truncated selection follows the TOKEN, not the column.
perm = torch.stack([torch.randperm(K)[:8] for _ in range(N)])
sel = torch.gather(head_ids, 1, perm)
got = align_log_agreement(log_agree, head_ids, sel)
check("permuted topk=8 selection tracks the token", torch.equal(got, torch.gather(log_agree, 1, perm)))
check("a naive column slice would have been wrong", not torch.equal(got, log_agree[:, :8]))

# c) a token the head never ranked is floored, not silently given column 0.
sel_missing = sel.clone()
sel_missing[:, 0] = 99999  # not in any row's top-k
got = align_log_agreement(log_agree, head_ids, sel_missing)
check("unranked token gets the MIN_AGREEMENT floor",
      bool((got[:, 0] == LOG_MIN_AGREEMENT).all()))
check("its neighbours are untouched", torch.equal(got[:, 1:], torch.gather(log_agree, 1, perm)[:, 1:]))

# --- 2. drop inheritance ------------------------------------------------
print("\ndrop inheritance (SGLANG_SPEC_EXPAND_CHILDREN=2, topk=4):")
B, TOPK, HID, CHILDREN, i = 3, 4, 8, 2, 1
scores = torch.log(torch.rand(B, TOPK))
topk_p = torch.rand(B * TOPK, TOPK)
topk_p = topk_p / topk_p.sum(-1, keepdim=True)
topk_index = torch.randint(0, 1000, (B * TOPK, TOPK))
hidden = torch.randn(B * TOPK, HID)

_, _, _, tree_info = _select_top_k_tokens_later(
    i, topk_p, topk_index, hidden, scores, TOPK, True, CHILDREN
)
draft_entry, parents = tree_info[0], tree_info[2]
dropped = draft_entry <= -1e29
check("the capped children really are dropped in the draft pool",
      bool(dropped.any()) and bool(dropped[:, :, :CHILDREN].sum() == 0))


class _FakeHead:
    top_k = TOPK


sc = AgreementTreeScorer.__new__(AgreementTreeScorer)
sc.head, sc.topk = _FakeHead(), TOPK
sc.reset()
root_entry = torch.log(torch.rand(B, 1, TOPK))
root_capped = _cap_children(root_entry, TOPK, CHILDREN, True)
sc.step_root(torch.log(torch.rand(B, TOPK)), root_capped)
check("root: dropped columns stay dropped in the agreement pool",
      bool((sc.score_list[0][root_capped <= -1e29] <= -1e29).all()))
check("root: surviving columns are agreement values, not draft ones",
      bool((sc.score_list[0][root_capped > -1e29] > -1e29).all())
      and not torch.equal(sc.score_list[0], root_capped))

sc.step_later(i, torch.log(torch.rand(B * TOPK, TOPK)), parents, draft_entry)
agree_entry = sc.score_list[1]
check("level 1: every draft-dropped node is dropped in the agreement pool",
      bool((agree_entry[dropped] <= -1e29).all()))
check("level 1: no surviving node was dropped by accident",
      bool((agree_entry[~dropped] > -1e29).all()))
check("level 1: survivors are ranked by agreement, not by the draft score",
      not torch.equal(agree_entry[~dropped], draft_entry[~dropped]))

# Without the mask the capped branches come back -- this is the bug being fenced.
sc2 = AgreementTreeScorer.__new__(AgreementTreeScorer)
sc2.head, sc2.topk = _FakeHead(), TOPK
sc2.reset()
sc2.step_root(torch.log(torch.rand(B, TOPK)))
sc2.step_later(i, torch.log(torch.rand(B * TOPK, TOPK)), parents)
check("control: omitting draft_entry DOES resurrect them (mask is load-bearing)",
      bool((sc2.score_list[1][dropped] > -1e29).all()))

print("\nALIGNMENT + DROPS", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
