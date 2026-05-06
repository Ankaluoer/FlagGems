# SPDX-License-Identifier: Apache-2.0
"""
Fused Marlin MoE for FlagGems.

Aligns the interface of vLLM v0.20.0:
    vllm/model_executor/layers/fused_moe/fused_marlin_moe.py :: fused_marlin_moe

PHASE 1 (this file): wrapper on top of FlagGems' existing
`fused_experts_impl`. Correctness-focused. Performance is bounded by the
underlying FP16 GEMM path (Triton currently dequants INT4/INT8 to FP16 and
runs an FP16 GEMM, see fused_experts_impl in fused_moe.py).

PHASE 2 (TODO): replace the dequant-then-FP16-GEMM path with a true
Marlin-style fused dequant + tensor-core GEMM Triton kernel for real
W4A16 / W8A16 speedup.

MVP scope:
  - quant_type: GPTQ uint4b8 (INT4) and uint8b128 (INT8)
  - activation: SwiGLU / SiLU
  - act_order:  NOT supported (g_idx / sort_indices must be None)
  - FP8 input:  NOT supported
  - LoRA, clamp_limit, expert_map: NOT supported
"""
from typing import Any, Callable, List, Optional

import torch

from flag_gems.fused.fused_moe import fused_experts_impl


# ----------------------------------------------------------------------------
# quant_type_id constants — mirror a subset of vLLM scalar_types ids.
# vLLM uses ScalarType.from_id(quant_type_id); during integration we should
# resolve the exact int values. For a self-contained MVP we accept either
# the well-known vLLM ids or our own constants.
# ----------------------------------------------------------------------------
# GPTQ INT4 (weight stored as w + 8, dequant subtracts 8)
QUANT_TYPE_UINT4B8 = 0
# INT8 (weight stored as w + 128)
QUANT_TYPE_UINT8B128 = 1

_QUANT_TYPE_INT4 = {QUANT_TYPE_UINT4B8}
_QUANT_TYPE_INT8 = {QUANT_TYPE_UINT8B128}
_SUPPORTED_QUANT_TYPES = _QUANT_TYPE_INT4 | _QUANT_TYPE_INT8


def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: Optional[torch.Tensor],
    bias2: Optional[torch.Tensor],
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type_id: int,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    activation: Any = None,
    activation_func: Optional[Callable] = None,
    moe_sum: Optional[Callable] = None,
    expert_map: Optional[torch.Tensor] = None,
    input_global_scale1: Optional[torch.Tensor] = None,
    input_global_scale2: Optional[torch.Tensor] = None,
    global_scale1: Optional[torch.Tensor] = None,
    global_scale2: Optional[torch.Tensor] = None,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    intermediate_cache13: Optional[torch.Tensor] = None,
    intermediate_cache2: Optional[torch.Tensor] = None,
    is_k_full: bool = True,
    output: Optional[torch.Tensor] = None,
    input_dtype: Optional[torch.dtype] = None,
    inplace: bool = False,
    clamp_limit: Optional[float] = None,
) -> torch.Tensor:
    """
    Phase-1 wrapper: route a Marlin-style call into FlagGems'
    `fused_experts_impl`. See module docstring for scope and limitations.
    """
    # ---- MVP guardrails --------------------------------------------------
    if quant_type_id not in _SUPPORTED_QUANT_TYPES:
        raise NotImplementedError(
            f"MVP supports quant_type_id in {_SUPPORTED_QUANT_TYPES}, "
            f"got {quant_type_id}"
        )
    if g_idx1 is not None or g_idx2 is not None:
        raise NotImplementedError("act_order (g_idx) not yet supported in MVP")
    if sort_indices1 is not None or sort_indices2 is not None:
        raise NotImplementedError(
            "act_order (sort_indices) not yet supported in MVP"
        )
    if input_dtype is not None:
        raise NotImplementedError("FP8 / INT8 input quantization not supported")
    if clamp_limit is not None:
        raise NotImplementedError("clamp_limit (GLM-4 swiglu) not supported")
    # bias / global_scale / input_global_scale: tolerated only when None.
    if input_global_scale1 is not None or input_global_scale2 is not None:
        raise NotImplementedError("input_global_scale not supported in MVP")
    if global_scale1 is not None or global_scale2 is not None:
        raise NotImplementedError("global_scale not supported in MVP")
    if workspace is not None:
        # Marlin uses workspace for atomic-add reduction; not used here.
        pass
    # `intermediate_cache13` / `intermediate_cache2` are buffer-reuse hints.
    # `fused_experts_impl` allocates its own caches — silently ignore.
    # `output` / `inplace` are honored below via `inplace=` to fused_experts_impl.

    # ---- quant_type_id -> use_int4_w4a16 / use_int8_w8a16 flags ---------
    use_int4_w4a16 = quant_type_id in _QUANT_TYPE_INT4
    use_int8_w8a16 = quant_type_id in _QUANT_TYPE_INT8

    # ---- Activation: MVP only "silu" / SwiGLU ----------------------------
    # vLLM passes a MoEActivation enum; FlagGems takes the string "silu".
    # When `activation` is an enum, defer to its .value or .name; otherwise
    # default to "silu" (only path supported by fused_experts_impl asserts).
    activation_str = "silu"
    if activation is not None:
        # Best-effort: support enum-like or string-like inputs.
        for attr in ("value", "name"):
            v = getattr(activation, attr, None)
            if isinstance(v, str):
                activation_str = v.lower()
                break
        if isinstance(activation, str):
            activation_str = activation.lower()
    if activation_str != "silu":
        raise NotImplementedError(
            f"MVP only supports SiLU/SwiGLU activation, got {activation_str}"
        )

    # ---- Inplace / output handling --------------------------------------
    # vLLM `inplace=True` writes back into hidden_states; if `output` is
    # given, that buffer is used. We honor inplace via fused_experts_impl;
    # `output` is not directly forwarded (Phase-2 will optimize this).
    if inplace and output is not None:
        raise ValueError("Cannot pass both inplace=True and output")

    # ---- Delegate to fused_experts_impl ---------------------------------
    result = fused_experts_impl(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=inplace,
        activation=activation_str,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_int4_w4a16=use_int4_w4a16,
        use_int8_w8a16=use_int8_w8a16,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=w1_zeros,
        w2_zp=w2_zeros,
        w1_bias=bias1,
        w2_bias=bias2,
        # Marlin uses per-group scales; group_size is encoded as
        # block_shape=[0, group_size]. MVP defaults to 128.
        # block_shape=[0, min(128, hidden_states.size(-1))],
    )

    # ---- Optional explicit output buffer (vLLM convention) --------------
    if output is not None:
        output.copy_(result)
        return output
    return result


__all__ = ["fused_marlin_moe", "QUANT_TYPE_UINT4B8", "QUANT_TYPE_UINT8B128"]
