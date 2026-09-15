#!/usr/bin/env python3
"""Parity test: SGLang's agreement head must match corsair's bit-for-bit.

A miscalibrated head is WORSE than no head (measured: -0.36 MAL on GSM8K,
-2.25 on MATH-500 when the head is applied off-distribution), and a feature
mismatch produces exactly that failure silently -- the numbers still look
plausible. So this compares the two implementations on the SAME random inputs
with the SAME real checkpoint, and requires agreement to ~1e-5.

Run:  python test_agreement_parity.py
"""
from __future__ import annotations
import sys
import torch

sys.path.insert(0, "/home/ubuntu/pratap/GithubProjects/corsair-spec-selector/src")
sys.path.insert(0, "/home/ubuntu/pratap/GithubProjects/sglang/python")

from corsair_spec_selector.agreement_model import AgreementScorer as CorsairScorer
from sglang.srt.speculative.agreement_head import (
    AgreementScorer as SGLangScorer,
    build_agreement_features,
    agreement_log_probs,
    load_agreement_head,
)

CKPT = "/home/ubuntu/pratap/GithubProjects/corsair-spec-selector/artifacts/agreement_main9000.pt"
EMB = "/home/ubuntu/pratap/GithubProjects/corsair-spec-selector/artifacts/qwen3_17b_embedding.pt"

torch.manual_seed(0)
dev = "cuda:0" if torch.cuda.is_available() else "cpu"

emb = torch.load(EMB, map_location="cpu", weights_only=True)["weight"]
vocab, hidden_size = emb.shape
print(f"embedding: vocab={vocab} hidden={hidden_size}")

# --- build both heads from the SAME checkpoint ---------------------------
sg = load_agreement_head(CKPT, emb, dev)

state = torch.load(CKPT, map_location="cpu", weights_only=True)
cors = CorsairScorer(
    emb,
    projection_dim=state["config"]["projection_dim"],
    top_k=state["config"]["top_k"],
)
cors.load_state_dict(state["model_state"], strict=False)
cors = cors.to(dev).float().eval()

# --- identical random draft state ---------------------------------------
B, TOPK = 7, state["config"]["top_k"]
logits = torch.randn(B, vocab, device=dev) * 3.0
hid = torch.randn(B, hidden_size, device=dev)
pos = torch.randint(0, 250, (B,), device=dev)
ctx = torch.randint(500, 30000, (B,), device=dev)

# --- path A: SGLang shared feature builder ------------------------------
batch_sg, top_ids_sg, top_lp_sg = build_agreement_features(logits, hid, pos, ctx, TOPK)
out_sg = sg(batch_sg)

# --- path B: corsair's construction, transcribed from sglang_runtime.py --
log_probs = torch.log_softmax(logits.float(), dim=-1)
top_lp_c, top_ids_c = torch.topk(log_probs, TOPK, dim=-1)
probs = log_probs.exp()
entropy = -(probs * log_probs).sum(-1)
gap = top_lp_c[:, 0] - top_lp_c[:, 1]
batch_c = {
    "hidden": hid.float(), "candidate_ids": top_ids_c,
    "candidate_logprobs": top_lp_c, "entropy": entropy,
    "top1_top2_gap": gap, "position": pos, "context_length": ctx,
}
out_c = cors(batch_c)

ok = True
def check(name, a, b, tol):
    global ok
    d = (a.float() - b.float()).abs().max().item()
    status = "OK " if d <= tol else "FAIL"
    if d > tol:
        ok = False
    print(f"  {status} {name:26} max|diff| = {d:.3e}  (tol {tol:g})")

print("\nfeature parity:")
check("candidate_ids", top_ids_sg, top_ids_c, 0)
check("candidate_logprobs", top_lp_sg, top_lp_c, 0)
check("entropy", batch_sg["entropy"], entropy, 0)
check("top1_top2_gap", batch_sg["top1_top2_gap"], gap, 0)

print("\nhead output parity (17-way logits):")
check("logits", out_sg, out_c, 1e-4)

print("\nconsumed quantity (per-edge log agreement):")
la_sg, _, feat_sg = agreement_log_probs(sg, logits, hid, pos, ctx)
la_c = torch.softmax(out_c, dim=-1)[:, :TOPK].clamp_min(1e-30).log()
check("log_agreement", la_sg, la_c, 1e-4)

# The features handed back for dumping must BE the ones the head consumed --
# that is the whole basis of the serving-vs-collection parity test.
for key in ("candidate_ids", "candidate_logprobs", "entropy", "top1_top2_gap"):
    check(f"returned feature {key}", feat_sg[key], batch_c[key], 0)

# The none-of-K class must be DROPPED, not renormalised: the residual mass is
# the head's only way to say "this gate is risky", and renormalising would
# destroy exactly the cross-depth signal the tree rerank consumes.
mass = torch.softmax(out_sg, dim=-1)[:, :TOPK].sum(-1)
print(f"\n  candidate mass per gate (must be < 1): min {mass.min():.4f} max {mass.max():.4f}")
if mass.max() >= 1.0:
    ok = False
    print("  FAIL none-of-K class is not being withheld")

print("\nPARITY", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
