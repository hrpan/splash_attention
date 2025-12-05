import torch
import torch.nn.functional as F

import helion
import helion.language as hl

import math

import random

import sparse_attention


@helion.kernel(
    autotune_effort="none",
    static_shapes=False,
    ignore_warnings=[helion.exc.TensorOperationInWrapper]
)
def _sparse_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seed: int,
    causal: hl.constexpr = False,
    sample: hl.constexpr = False,
    return_map: hl.constexpr = False
):

    B, nh, T, hs = q.shape

    q = q.view(-1, T, hs)
    k = k.view(-1, T, hs)
    v = v.view(-1, T, hs)

    scale = 1 / math.sqrt(q.size(-1))
    out = torch.zeros((B * nh, T, hs), device=q.device, dtype=torch.float32)

    if return_map:
        adj = torch.zeros((B * nh, T, T), device=q.device, dtype=torch.bool)
    else:
        adj = None

    max_logits = torch.full((B * nh, T), float('-inf'), device=q.device, dtype=torch.float32)
    se = torch.full((B * nh, T), 0, device=q.device, dtype=torch.float32)

    # count number of non-masked elements for averaging sparsity (trivial for now, but can be useful for future)
    if causal:
        count = float(T * (T + 1) // 2)
    else:
        count = float(T * T)

    p_mask_avg = torch.zeros((B * nh, ), device=q.device, dtype=torch.float32)

    for tile_b in hl.tile(B * nh):
        for tile_q in hl.tile(T):
            _max_logits = max_logits[tile_b, tile_q]
            _se = se[tile_b, tile_q]
            qs = q[tile_b, tile_q, :].float()
            for tile_k in hl.tile(T):
                ks = k[tile_b, tile_k, :].float()
                logits = (qs @ ks.transpose(-1, -2) * scale)

                if causal:
                    causal_mask = tile_q.index[:, None] >= tile_k.index[None, :]
                    logits = torch.where(causal_mask, logits, float('-inf'))
                p_mask_avg[tile_b] += logits.sigmoid().sum(dim=-1).sum(dim=-1)

                new_max_logits = torch.maximum(_max_logits, logits.amax(dim=-1))
                old_se = _se * torch.exp(_max_logits - new_max_logits)
                exp_weights = torch.exp(logits - new_max_logits[:, :, None])
                new_se = old_se + exp_weights.sum(dim=-1)

                if sample:
                    rand = torch.logit(hl.rand(logits.shape, seed=seed, device=q.device))
                    sparse_mask = logits + rand > 0
                else:
                    sparse_mask = logits > 0

                if return_map:
                    if causal:
                        adj[tile_b, tile_q, tile_k] = sparse_mask & causal_mask
                    else:
                        adj[tile_b, tile_q, tile_k] = sparse_mask
                weights = torch.where(sparse_mask, exp_weights, 0) / new_se[:, :, None]
                out_old = (old_se / new_se)[:, :, None] * out[tile_b, tile_q, :]
                out[tile_b, tile_q, :] = torch.baddbmm(out_old, weights, v[tile_b, tile_k, :].float())
                _max_logits = new_max_logits
                _se = new_se
            se[tile_b, tile_q] = _se
            max_logits[tile_b, tile_q] = _max_logits
    p_mask_avg.div_(count)
    return (
        out.view(B, nh, T, hs).to(dtype=q.dtype),
        p_mask_avg.view(B, nh).to(dtype=q.dtype),
        adj,
        max_logits,
        se
    )


@helion.kernel(
    autotune_effort="none",
    static_shapes=False,
    ignore_warnings=[helion.exc.TensorOperationInWrapper]
)
def _sparse_attn_bwd(
    grad_out: torch.Tensor,
    grad_mask: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    max_logits: torch.Tensor,
    lse: torch.Tensor,
    seed: int,
    causal: hl.constexpr = False,
    sample: hl.constexpr = False,
):

    B, nh, T, hs = q.shape

    out = out.view(-1, T, hs)

    q = q.view(-1, T, hs)
    k = k.view(-1, T, hs)
    v = v.view(-1, T, hs)

    grad_out = grad_out.view(-1, T, hs)
    grad_mask = grad_mask.view(-1)

    scale = 1 / math.sqrt(q.size(-1))
    grad_q = torch.zeros_like(q, dtype=torch.float32)
    grad_k = torch.zeros_like(k, dtype=torch.float32)
    grad_v = torch.zeros_like(v, dtype=torch.float32)

    if causal:
        count = float(T * (T + 1) // 2)
    else:
        count = float(T * T)

    for tile_b in hl.tile(B * nh):
        for tile_q in hl.tile(T):
            _max_logits = max_logits[tile_b, tile_q]
            _lse = lse[tile_b, tile_q]
            qs = q[tile_b, tile_q, :].float()

            for tile_k in hl.tile(T):
                ks = k[tile_b, tile_k, :].float()
                logits = (qs @ ks.transpose(-1, -2) * scale)

                if causal:
                    causal_mask = tile_q.index[:, None] >= tile_k.index[None, :]
                    logits = torch.where(causal_mask, logits, float('-inf'))

                if sample:
                    rand = torch.logit(hl.rand(logits.shape, seed=seed, device=q.device))
                    sparse_mask = logits + rand > 0
                    gate = (logits + rand).sigmoid()
                    gate_prob = logits.sigmoid()
                else:
                    sparse_mask = logits > 0
                    gate = logits.sigmoid()
                    gate_prob = gate

                attn_weights = torch.exp(logits - _max_logits[:, :, None]) / _lse[:, :, None]
                final_weights = torch.where(sparse_mask, attn_weights, 0)

                grad_v[tile_b, tile_k, :] = torch.baddbmm(
                    grad_v[tile_b, tile_k, :],
                    final_weights.transpose(-1, -2),
                    grad_out[tile_b, tile_k, :].float(),
                )

                gvt = torch.matmul(
                    grad_out[tile_b, tile_q, :].float(),
                    v[tile_b, tile_k, :].transpose(-1, -2).float()
                )
                go = (
                    grad_out[tile_b, tile_q, :].float() * out[tile_b, tile_q, :].float()
                ).sum(dim=-1)[:, :, None]

                grad_score = attn_weights * (((gate * (1 - gate)) + sparse_mask.to(q.dtype)) * gvt - go) * scale

                # sparsity bwd
                grad_sigmoid = grad_mask[tile_b, None, None].float() * gate_prob * (1 - gate_prob) * scale / count

                grad_both = grad_score + grad_sigmoid

                grad_q[tile_b, tile_q, :] = torch.baddbmm(grad_q[tile_b, tile_q, :], grad_both, ks)
                grad_k[tile_b, tile_k, :] = torch.baddbmm(grad_k[tile_b, tile_k, :], grad_both.transpose(-1, -2), qs)

    return (
        grad_q.view(B, nh, T, hs).to(dtype=q.dtype),
        grad_k.view(B, nh, T, hs).to(dtype=q.dtype),
        grad_v.view(B, nh, T, hs).to(dtype=q.dtype)
    )


class SplashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal: bool = False, sample: bool = False, return_map: bool = False):

        assert q.shape == k.shape and k.shape == v.shape

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        seed = random.randint(0, 2 ** 31)
        ctx.seed = seed
        ctx.causal = causal
        ctx.sample = sample

        out, p_mask, adj, max_logits, lse = _sparse_attn_fwd(q, k, v, seed, causal, sample, return_map)

        ctx.save_for_backward(q, k, v, out, p_mask, max_logits, lse)

        if isinstance(adj, torch.Tensor):
            ctx.mark_non_differentiable(adj)

        return out, p_mask, adj

    @staticmethod
    def backward(ctx, grad_out, grad_p_mask, dummy):

        q, k, v, out, p_mask, max_logits, lse = ctx.saved_tensors
        seed = ctx.seed
        causal = ctx.causal
        sample = ctx.sample

        grad_q, grad_k, grad_v = _sparse_attn_bwd(
            grad_out.contiguous(),
            grad_p_mask.contiguous(),
            q, k, v, out, max_logits, lse,
            seed, causal, sample
        )

        return grad_q, grad_k, grad_v, None, None, None


splash_attention = SplashAttention.apply

if __name__ == '__main__':

    eps = 1e-2  # from helion puzzle

    q, k, v = torch.randn(3, 1, 2, 10, 16, device='cuda', dtype=torch.float64).unbind(0)
    mask_size = q.size(-3) * q.size(-2) * q.size(-2)

    # causal attention test
    with torch.no_grad():
        out, p_mask, adj = splash_attention(q, k, v, True, False, True)
        gold_out, gold_p_mask, gold_adj = sparse_attention._sparse_attention_torch(q, k, v, True, False, True)
    print('### causal=True sample=False forward test: ')
    out_diff = (out - gold_out).abs().max().item()
    assert torch.allclose(out, gold_out, atol=eps, rtol=eps), f'out failed abs max: {out_diff:.4f}'
    print('out passed with abs diff:', out_diff)
    mask_diff = (p_mask - gold_p_mask).abs().max().item()
    assert torch.allclose(p_mask, gold_p_mask, atol=eps, rtol=eps), f'mask failed abs max: {mask_diff:.4f}'
    print('expected mask passed with abs diff:', mask_diff)
    mask_same_count = (adj == gold_adj).sum()
    print(f'mask pass rate: {100 * mask_same_count/mask_size:.4f}%')

    # causal attention sample test
    # no gold comparison because rng
    with torch.no_grad():
        out, p_mask, adj = splash_attention(q, k, v, True, True, True)
    print('### causal sample forward passed')

    # noncausal attention test
    with torch.no_grad():
        out, p_mask, adj = splash_attention(q, k, v, False, False, True)
        gold_out, gold_p_mask, gold_adj = sparse_attention._sparse_attention_torch(q, k, v, False, False, True)
    print('### noncausal forward test: ')
    out_diff = (out - gold_out).abs().max().item()
    assert torch.allclose(out, gold_out, atol=eps, rtol=eps), f'out failed abs max: {out_diff:.4f}'
    print('out passed with abs diff:', (gold_out - out).abs().max())
    mask_diff = (p_mask - gold_p_mask).abs().max().item()
    assert torch.allclose(p_mask, gold_p_mask, atol=eps, rtol=eps), f'mask failed abs max: {mask_diff:.4f}'
    print('expected mask passed with abs diff:', mask_diff)
    mask_same_count = (adj == gold_adj).sum()
    print(f'mask pass rate: {100 * mask_same_count/mask_size:.4f}%')

    # noncausal attention sample test
    # no gold comparison because rng
    with torch.no_grad():
        out, p_mask, adj = splash_attention(q, k, v, False, True, True)
    print('### causal sample forward passed')

    # backward tests
    q, k, v = torch.randn(3, 1, 2, 10, 16, device='cuda', dtype=torch.float32).unbind(0)
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    q2 = q.detach().clone()
    k2 = k.detach().clone()
    v2 = v.detach().clone()
    q2.requires_grad = True
    k2.requires_grad = True
    v2.requires_grad = True
    out, p_mask, adj = splash_attention(q, k, v, False, False, False)
    gold_out, gold_p_mask, gold_adj = sparse_attention._sparse_attention_torch(q2, k2, v2, False, False, False)
    (out.sum() + p_mask.sum()).backward()
    (gold_out.sum() + gold_p_mask.sum()).backward()
    print('### noncausal backward test: ')
    grad_diff = (q.grad - q2.grad).abs().max()
    assert torch.allclose(q.grad, q2.grad, atol=eps, rtol=eps), f'q.grad failed abs max: {grad_diff:.4f}'
    print('q.grad passed with abs diff:', grad_diff)
    grad_diff = (k.grad - k2.grad).abs().max()
    assert torch.allclose(k.grad, k2.grad, atol=eps, rtol=eps), f'k.grad failed abs max: {grad_diff:.4f}'
    print('k.grad passed with abs diff:', grad_diff)
    grad_diff = (v.grad - v2.grad).abs().max()
    assert torch.allclose(v.grad, v2.grad, atol=eps, rtol=eps), f'v.grad failed abs max: {grad_diff:.4f}'
    print('v.grad passed with abs diff:', grad_diff)

    q.grad.zero_()
    k.grad.zero_()
    v.grad.zero_()
    q2.grad.zero_()
    k2.grad.zero_()
    v2.grad.zero_()

    out, p_mask, adj = splash_attention(q, k, v, True, False, False)
    gold_out, gold_p_mask, gold_adj = sparse_attention._sparse_attention_torch(q2, k2, v2, True, False, False)
    (out.sum() + p_mask.sum()).backward()
    (gold_out.sum() + gold_p_mask.sum()).backward()
    print('### causal backward test: ')
    grad_diff = (q.grad - q2.grad).abs().max()
    assert torch.allclose(q.grad, q2.grad, atol=eps, rtol=eps), f'q.grad failed abs max: {grad_diff:.4f}'
    print('q.grad passed with abs diff:', grad_diff)
    grad_diff = (k.grad - k2.grad).abs().max()
    assert torch.allclose(k.grad, k2.grad, atol=eps, rtol=eps), f'k.grad failed abs max: {grad_diff:.4f}'
    print('k.grad passed with abs diff:', grad_diff)
    grad_diff = (v.grad - v2.grad).abs().max()
    assert torch.allclose(v.grad, v2.grad, atol=eps, rtol=eps), f'v.grad failed abs max: {grad_diff:.4f}'
    print('v.grad passed with abs diff:', grad_diff)
