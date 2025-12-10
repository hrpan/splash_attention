import pytest
import torch
from torch.testing import assert_close
from splash_attention import sparse_attention_naive
from splash_attention.splash_attention import sparse_attn_fwd_debug, sparse_attn_bwd_debug


EPS_FP32 = 1e-2
EPS_BF16 = 1e-1

IN_SHAPE = (1, 4, 16, 16)


@pytest.mark.fwd
@pytest.mark.parametrize('dtype', ['BF16', 'FP32'])
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('sample', [False])
@pytest.mark.parametrize('bias', [0., 1.])
def test_fwd(dtype, causal, sample, bias):

    if dtype == 'BF16':
        _dtype = torch.bfloat16
        _eps = EPS_BF16
    elif dtype == 'FP32':
        _dtype = torch.float32
        _eps = EPS_FP32
    q, k, v = torch.rand(3, *IN_SHAPE, device='cuda', dtype=_dtype).unbind(0)

    print(f'\n###\n### CAUSAL={causal} SAMPLE={sample} FORWARD ({dtype})\n###')
    if sample:
        with torch.no_grad():
            _ = sparse_attn_fwd_debug(q, k, v, bias, causal, sample, True)
        return

    with torch.no_grad():
        out, p_mask, adj, _ = sparse_attn_fwd_debug(q, k, v, bias, causal, sample, True)
        gold_out, gold_p_mask, gold_adj = sparse_attention_naive(q.double(), k.double(), v.double(), bias, causal, sample, True)
    out_diff = (out.double() - gold_out).abs().max().item()
    assert_close(out, gold_out, atol=_eps, rtol=_eps, check_dtype=False, msg=f'out failed abs max: {out_diff:.4f}')
    print('out passed with abs diff:', out_diff)
    mask_diff = (p_mask.double() - gold_p_mask).abs().max().item()
    assert_close(p_mask, gold_p_mask, atol=_eps, rtol=_eps, check_dtype=False, msg=f'mask failed abs max: {mask_diff:.4f}')
    print('expected mask passed with abs diff:', mask_diff)
    mask_pass_rate = (adj == gold_adj).float().mean().item()
    print(f'mask pass rate: {100 * mask_pass_rate:.4f}%')


@pytest.mark.bwd
@pytest.mark.parametrize('dtype', ['BF16', 'FP32'])
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('sample', [False])
@pytest.mark.parametrize('bias', [0., 1.])
@pytest.mark.parametrize('weight', [0., 0.5, 1.])
def test_bwd(dtype, causal, sample, bias, weight):

    if dtype == 'BF16':
        _dtype = torch.bfloat16
        _eps = EPS_BF16
    elif dtype == 'FP32':
        _dtype = torch.float32
        _eps = EPS_FP32
    q, k, v = torch.rand(3, *IN_SHAPE, device='cuda', dtype=torch.float64).unbind(0)
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True

    out, p_mask, adj, lse = sparse_attn_fwd_debug(
        q.to(dtype=_dtype),
        k.to(dtype=_dtype),
        v.to(dtype=_dtype),
        bias, causal, sample, True
    )
    grad_out = torch.full_like(out, 1 - weight)
    grad_p_mask = torch.full_like(p_mask, weight)
    go = (grad_out[..., None, :] @ out[..., None]).squeeze([-1, -2])
    grad_q, grad_k, grad_v = sparse_attn_bwd_debug(
        grad_out, grad_p_mask, go,
        q.to(dtype=_dtype),
        k.to(dtype=_dtype),
        v.to(dtype=_dtype),
        out, lse, bias,
        causal, False, 0
    )

    gold_out, gold_p_mask, _ = sparse_attention_naive(q, k, v, bias, causal, sample, False)
    ((1 - weight) * gold_out.sum() + weight * gold_p_mask.sum()).backward()

    print(f'\n###\n### CAUSAL={causal} SAMPLE={sample} BACKWARD ({dtype})\n###')

    grad_diff = (grad_q.double() - q.grad).abs().max().item()
    assert_close(grad_q, q.grad, atol=_eps, rtol=_eps, check_dtype=False, msg=f'q.grad failed abs max: {grad_diff:.4f}')
    print('q.grad passed with abs diff:', grad_diff)
    grad_diff = (grad_k.double() - k.grad).abs().max().item()
    assert_close(grad_k, k.grad, atol=_eps, rtol=_eps, check_dtype=False, msg=f'k.grad failed abs max: {grad_diff:.4f}')
    print('k.grad passed with abs diff:', grad_diff)
    grad_diff = (grad_v.double() - v.grad).abs().max().item()
    assert_close(grad_v, v.grad, atol=_eps, rtol=_eps, check_dtype=False, msg=f'v.grad failed abs max: {grad_diff:.4f}')
    print('v.grad passed with abs diff:', grad_diff)
