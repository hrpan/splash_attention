import pytest
import torch
from splash_attention import sparse_attention_naive, splash_attention


EPS_FP32 = 1e-2
EPS_BF16 = 1e-1

IN_SHAPE = (4, 8, 64, 32)


@pytest.mark.fwd
@pytest.mark.parametrize('dtype', ['BF16', 'FP32'])
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('sample', [False, True])
@pytest.mark.parametrize('bias', [0., 1., 2.])
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
            _ = splash_attention(q, k, v, bias, causal, sample, True)
        return

    with torch.no_grad():
        out, p_mask, adj = splash_attention(q, k, v, bias, causal, sample, True)
        gold_out, gold_p_mask, gold_adj = sparse_attention_naive(q, k, v, bias, causal, sample, True)
    out_diff = (out - gold_out).abs().max().item()
    assert torch.allclose(out, gold_out, atol=_eps, rtol=_eps), f'out failed abs max: {out_diff:.4f}'
    print('out passed with abs diff:', out_diff)
    mask_diff = (p_mask - gold_p_mask).abs().max().item()
    assert torch.allclose(p_mask, gold_p_mask, atol=_eps, rtol=_eps), f'mask failed abs max: {mask_diff:.4f}'
    print('expected mask passed with abs diff:', mask_diff)
    mask_pass_rate = (adj == gold_adj).float().mean().item()
    print(f'mask pass rate: {100 * mask_pass_rate:.4f}%')


@pytest.mark.bwd
@pytest.mark.parametrize('dtype', ['BF16', 'FP32'])
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('sample', [False, True])
@pytest.mark.parametrize('bias', [0., 1.])
@pytest.mark.parametrize('weight', [0., 1., 2.])
def test_bwd(dtype, causal, sample, bias, weight):

    if dtype == 'BF16':
        _dtype = torch.bfloat16
        _eps = EPS_BF16
    elif dtype == 'FP32':
        _dtype = torch.float32
        _eps = EPS_FP32
    q, k, v = torch.rand(3, *IN_SHAPE, device='cuda', dtype=_dtype).unbind(0)
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    q2 = q.detach().clone()
    k2 = k.detach().clone()
    v2 = v.detach().clone()
    q2.requires_grad = True
    k2.requires_grad = True
    v2.requires_grad = True

    print(f'\n###\n### CAUSAL={causal} SAMPLE={sample} BACKWARD ({dtype})\n###')
    if sample:
        out, p_mask, _ = splash_attention(q, k, v, bias, causal, sample, False)
        (out.sum() + weight * p_mask.sum()).backward()
        return

    out, p_mask, _ = splash_attention(q, k, v, bias, causal, sample, False)
    (out.sum() + weight * p_mask.sum()).backward()
    gold_out, gold_p_mask, _ = sparse_attention_naive(q2, k2, v2, bias, causal, sample, False)
    (gold_out.sum() + weight * gold_p_mask.sum()).backward()

    grad_diff = (q.grad - q2.grad).abs().max().item()
    assert torch.allclose(q.grad, q2.grad, atol=_eps, rtol=_eps), f'q.grad failed abs max: {grad_diff:.4f}'
    print('q.grad passed with abs diff:', grad_diff)
    grad_diff = (k.grad - k2.grad).abs().max().item()
    assert torch.allclose(k.grad, k2.grad, atol=_eps, rtol=_eps), f'k.grad failed abs max: {grad_diff:.4f}'
    print('k.grad passed with abs diff:', grad_diff)
    grad_diff = (v.grad - v2.grad).abs().max().item()
    assert torch.allclose(v.grad, v2.grad, atol=_eps, rtol=_eps), f'v.grad failed abs max: {grad_diff:.4f}'
    print('v.grad passed with abs diff:', grad_diff)
