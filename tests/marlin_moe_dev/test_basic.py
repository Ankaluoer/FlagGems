"""
Phase-1 sanity test for fused_marlin_moe (FlagGems wrapper).
Compares against a naive PyTorch SwiGLU MoE reference at FP16.

Note: FlagGems' fused_experts_impl currently dequantizes INT4/INT8 weights
to FP16 before the GEMM (see comment in fused_moe.py). So for Phase 1 we
test correctness by feeding *already FP16* weights through the W4A16 path
with unit scales — this exercises the wrapper plumbing without hitting
true Marlin layout.
"""
import torch

from flag_gems.fused.fused_marlin_moe import (
    fused_marlin_moe,
    QUANT_TYPE_UINT4B8,
)


def reference_swiglu_moe(
    hidden_states: torch.Tensor,   # (M, K)
    w1: torch.Tensor,              # (E, 2N, K)  gate+up packed along dim 1
    w2: torch.Tensor,              # (E, K, N)
    topk_weights: torch.Tensor,    # (M, topk)
    topk_ids: torch.Tensor,        # (M, topk)
) -> torch.Tensor:
    """Naive but obviously-correct SwiGLU MoE."""
    M, K = hidden_states.shape
    E, two_N, _ = w1.shape
    N = two_N // 2
    topk = topk_ids.shape[1]
    out = torch.zeros_like(hidden_states)
    for m in range(M):
        for k in range(topk):
            e = topk_ids[m, k].item()
            w_topk = topk_weights[m, k]
            x = hidden_states[m]                    # (K,)
            gate_up = w1[e] @ x                     # (2N,)
            gate = gate_up[:N]
            up   = gate_up[N:]
            act  = torch.nn.functional.silu(gate) * up   # (N,)
            y    = w2[e] @ act                      # (K,)
            out[m] += w_topk.to(y.dtype) * y
    return out


def main():
    torch.manual_seed(0)
    device = 'cuda'
    dtype = torch.bfloat16

    M, K, N, E, topk = 64, 128, 256, 8, 2

    hidden = torch.randn(M, K, dtype=dtype, device=device) * 0.1
    w1 = torch.randn(E, 2 * N, K, dtype=dtype, device=device) * 0.1
    w2 = torch.randn(E, K, N,    dtype=dtype, device=device) * 0.1

    # Routing
    topk_weights = torch.softmax(
        torch.randn(M, topk, device=device), dim=-1
    ).to(torch.float32)
    topk_ids = torch.randint(0, E, (M, topk), dtype=torch.long, device=device)

    # Reference
    ref = reference_swiglu_moe(hidden, w1, w2, topk_weights, topk_ids)

    # Wrapper call. Phase-1 quirk: fused_experts_impl will dequant w1/w2
    # via `w * scale.unsqueeze(-1)`; using unit scales keeps math identical.
    w1_scale = torch.ones(E, 2 * N, dtype=dtype, device=device)
    w2_scale = torch.ones(E, K, dtype=dtype, device=device)

    got = fused_marlin_moe(
        hidden_states=hidden,
        w1=w1, w2=w2,
        bias1=None, bias2=None,
        w1_scale=w1_scale, w2_scale=w2_scale,
        topk_weights=topk_weights, topk_ids=topk_ids,
        quant_type_id=QUANT_TYPE_UINT4B8,  # treat as W4A16 path
    )

    print(f"ref  shape={tuple(ref.shape)}  dtype={ref.dtype}")
    print(f"got  shape={tuple(got.shape)}  dtype={got.dtype}")
    print(f"max abs diff: {(ref - got).abs().max().item():.6f}")
    print(f"mean abs diff: {(ref - got).abs().mean().item():.6f}")

    if torch.allclose(ref, got, rtol=1e-2, atol=1e-2):
        print("[PASS] outputs match within FP16 tolerance")
    else:
        print("[FAIL] outputs differ beyond tolerance")


if __name__ == "__main__":
    main()
