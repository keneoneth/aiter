# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib
from dataclasses import dataclass, fields

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import fused_moe, torch_moe
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.moe.moe_op_gemm_a16w4 import moe_gemm_torch
from aiter.ops.triton.moe.moe_op_gemm_a16w16 import (
    build_routing_from_topk,
    get_topk_align_config,
    is_triton_a16w16_available,
    moe_gemm_a16w16,
)
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.utils._triton import arch_info
from aiter.test_common import checkAllclose
from op_tests.triton_tests.moe.moe_test_utils import assert_close

_GLUON_SUPPORTED_ARCHS = ("gfx1250",)
_RTOL = 1e-2
_ATOL = 128
_POLICY_SELECTED_TOKENS = [
    1024,
    1025,
    1136,
    1152,
    1543,
    1544,
    1678,
    1724,
    1728,
    1799,
    1803,
    1815,
    1837,
    1944,
    1945,
    2039,
    2048,
    2304,
    2446,
    2560,
    3072,
    3584,
    3840,
    4096,
]
_POLICY_REJECTED_TOKENS = [1023, 4097]


def _on_triton_a16w16_configured_arch() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    try:
        get_gfx()
        return is_triton_a16w16_available() and bool(get_topk_align_config())
    except Exception:  # noqa: BLE001
        return False


def _on_gfx1201() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    try:
        return get_gfx() == "gfx1201"
    except Exception:  # noqa: BLE001
        return False


def init_routing_data(m, n_expts_tot, n_expts_act, do_gather, do_scatter, device):
    logits = torch.randn((m, n_expts_tot), dtype=torch.float16, device=device)
    routing_data, gather_idx, scatter_idx = routing(logits, n_expts_act)
    routing_data.gate_scal = None
    return (
        routing_data,
        gather_idx if do_gather else None,
        scatter_idx if do_scatter else None,
    )


def init_compute_data(
    m, n, k, n_expts_tot, n_expts_act, gindx, has_y_gammas, device, has_bias=True
):
    in_m = m * (n_expts_act if gindx is None else 1)
    x = torch.randn((in_m, k), dtype=torch.bfloat16, device=device)
    w = torch.randn((n_expts_tot, k, n), dtype=torch.bfloat16, device=device)
    bias = (
        torch.randn((n_expts_tot, n), dtype=torch.float32, device=device)
        if has_bias
        else None
    )
    if has_y_gammas:
        gammas = 2 ** torch.randint(
            -5, 0, (m * n_expts_act,), device=device, dtype=torch.float32
        )
    else:
        gammas = None
    return x, w, bias, gammas


def _make_large_m_a16w16_case(tokens: int):
    torch.manual_seed(2026 + tokens)
    model_dim = 2048
    inter_dim = 128
    expert = 256
    topk = 8
    hidden = torch.randn((tokens, model_dim), dtype=dtypes.bf16, device="cuda")
    w1 = torch.randn(
        (expert, 2 * inter_dim, model_dim), dtype=dtypes.bf16, device="cuda"
    )
    w2 = torch.randn((expert, model_dim, inter_dim), dtype=dtypes.bf16, device="cuda")
    ids = torch.randint(0, expert, (tokens, topk), dtype=torch.int32, device="cuda")
    weights = torch.rand((tokens, topk), dtype=torch.float32, device="cuda")
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return (
        hidden,
        w1.contiguous(),
        w2.contiguous(),
        weights.contiguous(),
        ids.contiguous(),
    )


def _make_case_gfx1201(tokens: int):
    return _make_large_m_a16w16_case(tokens)


def _large_m_gate_inputs(tokens: int):
    fused_moe_mod = importlib.import_module("aiter.fused_moe")
    hidden, w1, w2, weights, ids = _make_case_gfx1201(tokens)
    return {
        "hidden_states": hidden,
        "w1": w1,
        "w2": w2,
        "topk_weight": weights,
        "topk_ids": ids,
        "activation": aiter.ActivationType.Silu,
        "quant_type": aiter.QuantType.No,
        "dtype": dtypes.bf16,
        "q_dtype_a": dtypes.bf16,
        "q_dtype_w": dtypes.bf16,
        "is_g1u1": True,
        "doweight_stage1": False,
        "w1_scale": None,
        "w2_scale": None,
        "a1_scale": None,
        "a2_scale": None,
        "num_local_tokens": None,
        "moe_sorting_dispatch_policy": 0,
        "expert_mask": None,
        "hidden_pad": 0,
        "intermediate_pad": 0,
        "bias1": None,
        "bias2": None,
        "swiglu_limit": None,
        "beta": None,
        "linear_beta": None,
        "gate_mode": fused_moe_mod.GateMode.SEPARATED.value,
        "stage2_scatter": None,
    }


def _large_m_gate(**kwargs):
    fused_moe_mod = importlib.import_module("aiter.fused_moe")
    return fused_moe_mod._can_use_bf16_g1u1_gfx1201_triton_large_m(**kwargs)


def _force_validated_gfx1201_target(monkeypatch):
    fused_moe_mod = importlib.import_module("aiter.fused_moe")
    monkeypatch.setattr(fused_moe_mod, "get_gfx_runtime", lambda: "gfx1201")
    monkeypatch.setattr(fused_moe_mod, "get_cu_num", lambda: 64)


@dataclass
class Case:
    m: int
    n: int
    k: int
    n_expts_tot: int = 1
    n_expts_act: int = 1
    has_bias: bool = True
    out_dtype: torch.dtype = torch.bfloat16


@pytest.mark.parametrize(
    ", ".join(f.name for f in fields(Case)),
    [
        tuple(getattr(case, f.name) for f in fields(Case))
        for case in [
            Case(4, 128, 128, 2, 1),
            Case(16, 128, 128, 2, 1),
            Case(16, 256, 256, 128, 4),
            Case(64, 256, 256, 8, 2),
            Case(1024, 3072, 512, 128, 4),
            Case(256, 512, 512, 8, 2),
            Case(512, 256, 256, 8, 1),
            Case(300, 400, 768, 8, 4),
            Case(1000, 704, 768, 8, 2),
            Case(1024, 512, 512, 8, 2),
            Case(4096, 256, 256, 128, 4),
            Case(64, 256, 320, 8, 2),
            Case(300, 256, 400, 8, 4),
            Case(256, 256, 320, 8, 2),
            Case(16, 128, 128, 2, 1, has_bias=False),
            Case(256, 512, 512, 8, 2, has_bias=False),
            Case(256, 512, 512, 8, 2, out_dtype=torch.float16),
            Case(64, 256, 256, 8, 2, has_bias=False, out_dtype=torch.float16),
        ]
    ],
)
@pytest.mark.parametrize(
    "do_gather, do_scatter",
    [(False, False), (True, False), (False, True), (True, True)],
)
@pytest.mark.parametrize("has_y_gammas", [False, True])
@pytest.mark.parametrize("apply_swiglu", [False, True])
def test_op(
    m,
    n,
    k,
    n_expts_tot,
    n_expts_act,
    has_bias,
    out_dtype,
    do_gather,
    do_scatter,
    has_y_gammas,
    apply_swiglu,
    device="cuda",
):
    if arch_info.get_arch() not in _GLUON_SUPPORTED_ARCHS:
        pytest.skip(f"moe_gemm_a16w16 is only implemented for {_GLUON_SUPPORTED_ARCHS}")

    torch.manual_seed(0)

    rdata, gindx, sindx = init_routing_data(
        m, n_expts_tot, n_expts_act, do_gather, do_scatter, device
    )
    x, w, bias, gammas = init_compute_data(
        m, n, k, n_expts_tot, n_expts_act, gindx, has_y_gammas, device, has_bias
    )

    ref_y = moe_gemm_torch(
        x, w, bias, rdata, gindx, sindx, gammas, apply_swiglu, alpha=1.0, limit=1.0
    )
    tri_y = moe_gemm_a16w16(
        x,
        w,
        bias=bias,
        routing_data=rdata,
        gather_indx=gindx,
        scatter_indx=sindx,
        gammas=gammas,
        apply_swiglu=apply_swiglu,
        alpha=1.0,
        limit=1.0,
        out_dtype=out_dtype,
    )

    assert tri_y.dtype == out_dtype
    maxtol = 3e-2 if out_dtype == torch.float16 else None
    assert_close(
        ref_y.to(out_dtype),
        tri_y,
        maxtol=maxtol,
        description=f"a16w16 m={m} n={n} k={k} bias={has_bias} out={out_dtype}",
    )


@pytest.mark.skipif(
    not _on_triton_a16w16_configured_arch(),
    reason="configured Triton A16W16 large-M backend",
)
def test_moe_gemm_a16w16_triton_backend_direct():
    hidden, w1, w2, weights, ids = _make_large_m_a16w16_case(1152)

@pytest.mark.skipif(not _on_gfx1201(), reason="gfx1201 selected large-M coverage")
@pytest.mark.parametrize("tokens", [1024, 1152, 1544, 1728, 1944, 2048, 2560, 4096])
def test_public_fused_moe_bf16_g1u1_selected_large_m(monkeypatch, tokens):
    _force_validated_gfx1201_target(monkeypatch)
    hidden, w1, w2, weights, ids = _make_case_gfx1201(tokens)

    actual = fused_moe(
        hidden,
        w1,
        w2,
        weights,
        ids,
        activation=aiter.ActivationType.Silu,
        quant_type=aiter.QuantType.No,
        dtype=dtypes.bf16,
    )
    expected = torch_moe(
        hidden,
        w1,
        w2,
        weights,
        ids,
        activation=aiter.ActivationType.Silu,
    )
    assert checkAllclose(actual, expected, rtol=_RTOL, atol=_ATOL) == 0


    route_cfg = get_topk_align_config()
    routing_data, gather_idx, scatter_idx, sorted_weights = build_routing_from_topk(
        ids, weights, num_experts=w1.shape[0], block_m=route_cfg["block_m"]
    )

    stage1 = moe_gemm_a16w16(
        hidden.contiguous(),
        w1,
        routing_data=routing_data,
        gather_indx=gather_idx,
        out_dtype=torch.bfloat16,
        apply_swiglu=True,
        backend="triton",
    )
    actual = moe_gemm_a16w16(
        stage1.contiguous(),
        w2,
        routing_data=routing_data,
        scatter_indx=scatter_idx,
        gammas=sorted_weights,
        out_dtype=torch.bfloat16,
        backend="triton",
    )
    expected = torch_moe(
        hidden,
        w1,
        w2,
        weights,
        ids,
        activation=aiter.ActivationType.Silu,
    )
    assert checkAllclose(actual, expected, rtol=_RTOL, atol=_ATOL) == 0


@pytest.mark.skipif(not _on_gfx1201(), reason="gfx1201 selected large-M coverage")
def test_public_fused_moe_bf16_g1u1_selected_large_m_output_buffer(monkeypatch):
    _force_validated_gfx1201_target(monkeypatch)
    hidden, w1, w2, weights, ids = _make_case_gfx1201(1152)
    expected = torch_moe(
        hidden,
        w1,
        w2,
        weights,
        ids,
        activation=aiter.ActivationType.Silu,
    )
    output = torch.empty_like(expected)
    actual = fused_moe(
        hidden,
        w1,
        w2,
        weights,
        ids,
        activation=aiter.ActivationType.Silu,
        quant_type=aiter.QuantType.No,
        dtype=dtypes.bf16,
        output=output,
    )
    assert actual is output
    assert checkAllclose(output, expected, rtol=_RTOL, atol=_ATOL) == 0


@pytest.mark.skipif(not _on_gfx1201(), reason="gfx1201 selected large-M coverage")
def test_public_fused_moe_large_m_block_size_override_skips_shortcut(monkeypatch):
    _force_validated_gfx1201_target(monkeypatch)
    fused_moe_mod = importlib.import_module("aiter.fused_moe")
    hidden, w1, w2, weights, ids = _make_case_gfx1201(1152)
    sentinel = torch.empty_like(hidden)

    def fail_large_m_gate(**_kwargs):
        raise AssertionError("large-M gate helper should not run with block_size_M")

    def fake_generic_path(**kwargs):
        assert kwargs["block_size_M"] == 64
        return sentinel

    monkeypatch.setattr(
        fused_moe_mod,
        "_can_use_bf16_g1u1_gfx1201_triton_large_m",
        fail_large_m_gate,
    )
    monkeypatch.setattr(fused_moe_mod, "fused_moe_", fake_generic_path)

    actual = fused_moe(
        hidden,
        w1,
        w2,
        weights,
        ids,
        activation=aiter.ActivationType.Silu,
        quant_type=aiter.QuantType.No,
        dtype=dtypes.bf16,
        block_size_M=64,
    )

    assert actual is sentinel


@pytest.mark.skipif(not _on_gfx1201(), reason="gfx1201 selected large-M coverage")
@pytest.mark.parametrize("tokens", _POLICY_SELECTED_TOKENS)
def test_large_m_gate_accepts_validated_policy_rows(monkeypatch, tokens):
    _force_validated_gfx1201_target(monkeypatch)
    assert _large_m_gate(**_large_m_gate_inputs(tokens))


@pytest.mark.skipif(not _on_gfx1201(), reason="gfx1201 selected large-M coverage")
@pytest.mark.parametrize("tokens", _POLICY_REJECTED_TOKENS)
def test_large_m_gate_rejects_unvalidated_policy_rows(monkeypatch, tokens):
    _force_validated_gfx1201_target(monkeypatch)
    assert not _large_m_gate(**_large_m_gate_inputs(tokens))


@pytest.mark.skipif(not _on_gfx1201(), reason="gfx1201 selected large-M coverage")
def test_large_m_gate_rejects_unsupported_modes(monkeypatch):
    _force_validated_gfx1201_target(monkeypatch)
    fused_moe_mod = importlib.import_module("aiter.fused_moe")
    case = _large_m_gate_inputs(1152)

    monkeypatch.setattr(fused_moe_mod, "get_gfx_runtime", lambda: "gfx950")
    assert not _large_m_gate(**case)
    monkeypatch.setattr(fused_moe_mod, "get_gfx_runtime", lambda: "gfx1201")

    monkeypatch.setattr(fused_moe_mod, "get_cu_num", lambda: 60)
    assert not _large_m_gate(**case)
    monkeypatch.setattr(fused_moe_mod, "get_cu_num", lambda: 64)

    monkeypatch.setattr(
        fused_moe_mod,
        "get_cu_num",
        lambda: (_ for _ in ()).throw(AssertionError("mixed CUs")),
    )
    assert not _large_m_gate(**case)
    monkeypatch.setattr(fused_moe_mod, "get_cu_num", lambda: 64)

    assert not _large_m_gate(**{**case, "quant_type": aiter.QuantType.per_Token})
    assert not _large_m_gate(**{**case, "is_g1u1": False})
    assert not _large_m_gate(**{**case, "expert_mask": torch.ones(256, device="cuda")})
    assert not _large_m_gate(**{**case, "stage2_scatter": object()})

    shuffled_w1 = case["w1"].clone()
    shuffled_w1.is_shuffled = True
    assert not _large_m_gate(**{**case, "w1": shuffled_w1})

    noncontig_w1 = case["w1"].transpose(1, 2)
    assert not _large_m_gate(**{**case, "w1": noncontig_w1})
