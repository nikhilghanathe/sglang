#!/usr/bin/env python3
"""Unit-test the beam-index recovery and cumulative threading.

The only non-obvious arithmetic in the integration is recovering EAGLE's beam
choice from tree_info[2]. If that is off, the agreement cumulative silently
attaches to the WRONG parent and the head is scoring a different tree than the
one being built -- which would look like "the head does not work in serving"
rather than like a bug. So it is checked against the real
_select_top_k_tokens_later, not against a reimplementation.
"""
from __future__ import annotations
import sys
import torch

sys.path.insert(0, "/home/ubuntu/pratap/GithubProjects/sglang/python")
from sglang.srt.speculative.spec_utils import _select_top_k_tokens_later
from sglang.srt.speculative.agreement_head import AgreementTreeScorer

torch.manual_seed(0)
B, TOPK, HID = 3, 4, 8
ok = True

# Level i>=1: b*topk parents, each with topk children.
i = 1
scores = torch.randn(B, TOPK)                      # cumulative draft
topk_p = torch.rand(B * TOPK, TOPK)                # child conditionals
topk_p = topk_p / topk_p.sum(-1, keepdim=True)
topk_index = torch.randint(0, 1000, (B * TOPK, TOPK))
hidden = torch.randn(B * TOPK, HID)

input_ids, hid_out, new_scores, tree_info = _select_top_k_tokens_later(
    i, topk_p, topk_index, hidden, scores, TOPK, True, 0
)
expand_draft, tok, parents = tree_info[0], tree_info[1], tree_info[2]

# Recover the beam EAGLE chose, and verify by re-deriving its own outputs.
beam = parents - (TOPK * TOPK * (i - 1) + TOPK)
regathered = torch.gather(expand_draft.flatten(start_dim=1), 1, beam)
d = (regathered - new_scores).abs().max().item()
print(f"  beam index recovery -> draft scores match: max|diff| = {d:.3e}")
if d > 1e-5:
    ok = False
    print("  FAIL beam indices do not reproduce EAGLE's own returned scores")

regathered_ids = torch.gather(tok, 1, beam).flatten()
same = bool((regathered_ids == input_ids).all())
print(f"  beam index recovery -> token ids match: {same}")
if not same:
    ok = False

# Now the agreement path threads through the SAME beam.
class _FakeHead:
    top_k = TOPK
sc = AgreementTreeScorer.__new__(AgreementTreeScorer)
sc.head, sc.topk = _FakeHead(), TOPK
sc.reset()

root_log_agree = torch.log(torch.rand(B, TOPK).clamp_min(1e-6))
sc.step_root(root_log_agree)
assert sc.score_list[0].shape == (B, 1, TOPK), sc.score_list[0].shape
assert sc.cum.shape == (B, TOPK)

log_agree = torch.log(torch.rand(B * TOPK, TOPK).clamp_min(1e-6))
sc.step_later(i, log_agree, parents)
print(f"  agreement pool entry shape {tuple(sc.score_list[1].shape)} (expect {(B,TOPK,TOPK)})")
if tuple(sc.score_list[1].shape) != (B, TOPK, TOPK):
    ok = False

# Cumulative agreement must attach to the same parent the draft beam picked.
expect = root_log_agree.unsqueeze(2) + log_agree.view(B, TOPK, TOPK)
expect_cum = torch.gather(expect.flatten(start_dim=1), 1, beam)
d = (sc.cum - expect_cum).abs().max().item()
print(f"  cumulative agreement aligned to draft beam: max|diff| = {d:.3e}")
if d > 1e-6:
    ok = False

# Ancestor closure: every edge term must be <= 0 so no child outranks its parent.
child_max = expect.max().item()
parent_max = root_log_agree.max().item()
print(f"  non-positive edges -> closure holds: child_max {child_max:.3f} <= parent_max {parent_max:.3f}")
if child_max > parent_max + 1e-6:
    ok = False

print("\nTREE THREADING", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
