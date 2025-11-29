import torch
import torch.nn.functional as F

import helion
import helion.language as hl

import math

import sparse_attention


@helion.kernel(autotune_effort="none", ignore_warnings=[helion.exc.TensorOperationInWrapper])
def _sparse_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: hl.constexpr = False,
    sample: hl.constexpr = False,
    return_map: hl.constexpr = False
):

    B, nh, T, hs = q.shape

    q = q.view(-1, T, hs)
    k = k.view(-1, T, hs)
    v = v.view(-1, T, hs)

    scale = 1 / math.sqrt(q.size(-1))
    out = torch.empty((B * nh, T, hs), device=q.device, dtype=q.dtype)

    if return_map:
        adj = torch.zeros((B * nh, T, T), device=q.device, dtype=torch.bool)
    else:
        adj = None

    max_logits = torch.full((B * nh, T), float('-inf'), device=q.device, dtype=torch.float32)
    lse = torch.full((B * nh, T), 0, device=q.device, dtype=torch.float32)

    rands = torch.empty((B * nh, T, T), device=q.device, dtype=torch.float32)

    # count number of non-masked elements for averaging sparsity (trivial for now, but can be useful for future)
    count = torch.zeros((B * nh, ), device=q.device, dtype=torch.float32)

    p_mask_avg = torch.zeros((B * nh, ), device=q.device, dtype=torch.float32)

    for tile_b in hl.tile(B * nh):
        for tile_q in hl.tile(T):
            _max_logits = max_logits[tile_b, tile_q]
            _lse = lse[tile_b, tile_q]
            qs = q[tile_b, tile_q, :]
            for tile_k in hl.tile(T):
                ks = k[tile_b, tile_k, :]
                logits = (qs @ ks.transpose(-1, -2) * scale)

                if causal:
                    causal_mask = tile_q.index[:, None] >= tile_k.index[None, :]
                    logits = torch.where(causal_mask, logits, float('-inf'))
                count[tile_b] += torch.where(logits == float('-inf'), 0, 1).sum(dim=-1).sum(dim=-1)
                p_mask_avg[tile_b] += logits.sigmoid().sum(dim=-1).sum(dim=-1)

                new_max_logits = torch.maximum(_max_logits, logits.amax(dim=-1))
                ratio = torch.exp(_max_logits - new_max_logits)
                exp_weights = torch.exp(logits - new_max_logits[:, :, None])
                curr_lse = exp_weights.sum(dim=-1)
                new_lse = _lse * ratio + curr_lse

                if sample:
                    rand = torch.logit(torch.rand_like(logits))
                    rands[tile_b, tile_q, tile_k] = rand
                    sparse_mask = logits + rand > 0
                else:
                    sparse_mask = logits > 0

                if return_map:
                    if causal:
                        adj[tile_b, tile_q, tile_k] = (sparse_mask & causal_mask).float()
                    else:
                        adj[tile_b, tile_q, tile_k] = sparse_mask.float()
                weights = torch.where(sparse_mask, exp_weights, 0)
                curr_out = torch.matmul(weights, v[tile_b, tile_q, :])
                out_old = _lse[:, :, None] * out[tile_b, tile_q, :] * ratio[:, :, None]
                out[tile_b, tile_q, :] = (out_old + curr_out) / new_lse[:, :, None]

                _max_logits = new_max_logits
                _lse = new_lse
            lse[tile_b, tile_q] = _lse
            max_logits[tile_b, tile_q] = _max_logits
    p_mask_avg.div_(count)
    return out.view(B, nh, T, hs), p_mask_avg.view(B, nh), adj, max_logits, lse


@helion.kernel(autotune_effort="none", ignore_warnings=[helion.exc.TensorOperationInWrapper])
def _sparse_attn_mask_bwd(grad_mask: torch.Tensor, q: torch.Tensor, k: torch.Tensor, causal: hl.constexpr = False):

    # reference impl. for mask bwd

    B, nh, T, hs = q.shape

    q = q.view(-1, T, hs)
    k = k.view(-1, T, hs)

    scale = 1 / math.sqrt(q.size(-1))
    grad_q = torch.zeros_like(q)
    grad_k = torch.zeros_like(k)

    if causal:
        count = float(T * (T + 1) // 2)
    else:
        count = float(T * T)

    for tile_b in hl.tile(B * nh):
        for tile_q in hl.tile(T):
            qs = q[tile_b, tile_q, :]
            for tile_k in hl.tile(T):

                ks = k[tile_b, tile_k, :]
                logits = (qs @ ks.transpose(-1, -2) * scale)

                prob = logits.sigmoid()
                grad_sigmoid = grad_mask[tile_b, None, None] * prob * (1 - prob) * scale / count

                if causal:
                    causal_mask = tile_q.index[:, None] >= tile_k.index[None, :]
                    grad_sigmoid = torch.where(causal_mask, grad_sigmoid, 0)

                grad_q[tile_b, tile_q, :] = torch.baddbmm(grad_q[tile_b, tile_q, :], grad_sigmoid, ks)
                grad_k[tile_b, tile_k, :] = torch.baddbmm(grad_k[tile_b, tile_q, :], grad_sigmoid.transpose(-1, -2), qs)

    grad_q = grad_q.view(B, nh, T, hs)
    grad_k = grad_k.view(B, nh, T, hs)
    return grad_q, grad_k


@helion.kernel(autotune_effort="none")
def _sparse_attn_bwd(
    grad_out: torch.Tensor,
    grad_mask: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    max_logits: torch.Tensor,
    lse: torch.Tensor,
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
    grad_q = torch.zeros_like(q)
    grad_k = torch.zeros_like(k)
    grad_v = torch.zeros_like(v)

    if causal:
        count = float(T * (T + 1) // 2)
    else:
        count = float(T * T)

    for tile_b in hl.tile(B * nh):
        for tile_q in hl.tile(T):
            _max_logits = max_logits[tile_b, tile_q]
            _lse = lse[tile_b, tile_q]
            qs = q[tile_b, tile_q, :]

            for tile_k in hl.tile(T):
                ks = k[tile_b, tile_k, :]
                logits = (qs @ ks.transpose(-1, -2) * scale)

                if sample:
                    rand = torch.logit(torch.rand_like(logits))
                    sparse_mask = logits + rand > 0
                    gate = (logits + rand).sigmoid()
                    gate_prob = logits.sigmoid()
                else:
                    sparse_mask = logits > 0
                    gate = logits.sigmoid()
                    gate_prob = gate
                #logits = torch.where(sparse_mask, logits, float('-inf'))

                if causal:
                    causal_mask = tile_q.index[:, None] >= tile_k.index[None, :]
                    logits = torch.where(causal_mask, logits, float('-inf'))

                attn_weights = torch.exp(logits - _max_logits[:, :, None]) / _lse[:, :, None]
                final_weights = torch.where(sparse_mask, attn_weights, 0)

                grad_v[tile_b, tile_k, :] = torch.baddbmm(
                    grad_v[tile_b, tile_k, :],
                    final_weights.transpose(-1, -2),
                    grad_out[tile_b, tile_k, :]
                )

                gvt = torch.matmul(grad_out[tile_b, tile_q, :], v[tile_b, tile_k, :].transpose(-1, -2))
                go = (grad_out[tile_b, tile_q, :] * out[tile_b, tile_q, :]).sum(dim=-1)[:, :, None]

                grad_q[tile_b, tile_q, :] = torch.baddbmm(
                    grad_q[tile_b, tile_q, :],
                    attn_weights * (gvt * gate * (2 - gate) - go),
                    scale * k[tile_b, tile_k, :]
                )

                # sparsity bwd
                grad_sigmoid = grad_mask[tile_b, None, None] * gate_prob * (1 - gate_prob) * scale / count
                grad_q[tile_b, tile_q, :] = torch.baddbmm(grad_q[tile_b, tile_q, :], grad_sigmoid, ks)
                grad_k[tile_b, tile_k, :] = torch.baddbmm(grad_k[tile_b, tile_k, :], grad_sigmoid.transpose(-1, -2), qs)

    #grad_k = grad_k.view(B, nh, T, hs) / count
    return grad_q.view(B, nh, T, hs), None, grad_v.view(B, nh, T, hs)


class SparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal: bool = False, sample: bool = False, return_map: bool = False):

        assert q.shape == k.shape and k.shape == v.shape
        ctx.causal = causal
        ctx.sample = sample
        ctx.cuda_rng = torch.cuda.get_rng_state(device=q.device)

        out, p_mask, adj, max_logits, lse = _sparse_attn_fwd(q, k, v, causal, sample, return_map)

        ctx.save_for_backward(q, k, v, out, p_mask, max_logits, lse)
                #if sample:
        #    ctx.mark_non_differentiable(adj)
        #else:
        #    ctx.mark_non_differentiable(out, adj)

        return out, p_mask, adj

    @staticmethod
    def backward(ctx, grad_out, grad_p_mask, dummy):

        q, k, v, out, p_mask, max_logits, lse = ctx.saved_tensors
        causal = ctx.causal
        sample = ctx.sample

        grad_q = grad_k = grad_v = None
        with torch.random.fork_rng(devices=[q.device]):
            torch.cuda.set_rng_state(ctx.cuda_rng, q.device)
            grad_q, _, grad_v = _sparse_attn_bwd(grad_out, grad_p_mask, q, k, v, out, max_logits, lse, causal, sample)

        #grad_q, grad_k = _sparse_attn_mask_bwd(grad_p_mask, q, k, causal)

        return grad_q, grad_k, grad_v, None, None, None


sparse_attention_ = SparseAttention.apply

if __name__ == '__main__':

    eps = 1e-2  # from helion puzzle

    q, k, v = torch.randn(3, 1, 2, 10, 10, device='cuda', dtype=torch.float32).unbind(0)

    # causal attention test
    with torch.no_grad():
        out, p_mask, adj = sparse_attention_(q, k, v, True, False, False)
        gold_out, gold_p_mask, gold_adj = sparse_attention._sparse_attention_torch(q, k, v, True, False, False)
    print('causal forward test: ')
    assert torch.allclose(out, gold_out, atol=eps, rtol=eps)
    print('out passed with abs diff:', (gold_out - out).abs().max())
    assert torch.allclose(p_mask, gold_p_mask, atol=eps, rtol=eps)
    print('mask passed with abs diff:', (gold_p_mask - p_mask).abs().max())

    # noncausal attention test
    with torch.no_grad():
        out, p_mask, adj = sparse_attention_(q, k, v, False, False, False)
        gold_out, gold_p_mask, gold_adj = sparse_attention._sparse_attention_torch(q, k, v, False, False, False)
    print('noncausal forward test: ')
    assert torch.allclose(out, gold_out, atol=eps, rtol=eps)
    print('out passed with abs diff:', (gold_out - out).abs().max())
    assert torch.allclose(p_mask, gold_p_mask, atol=eps, rtol=eps)
    print('mask passed with abs diff:', (gold_p_mask - p_mask).abs().max())


    # mask backward test
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    q2 = q.detach().clone()
    k2 = k.detach().clone()
    v2 = v.detach().clone()
    q2.requires_grad = True
    k2.requires_grad = True
    v2.requires_grad = True
    out, p_mask, adj = sparse_attention_(q, k, v, False, False, False)
    gold_out, gold_p_mask, gold_adj = sparse_attention._sparse_attention_torch(q2, k2, v2, False, False, False)
    #(out.sum() + p_mask.sum()).backward()
    #(gold_out.sum() + gold_p_mask.sum()).backward()
    out.sum().backward()
    gold_out.sum().backward()
    print('noncausal backward test: ')
    grad_diff = (q.grad - q2.grad).abs().max()
    assert torch.allclose(q.grad, q2.grad, atol=eps, rtol=eps), f'q.grad failed abs max: {grad_diff:.4f}'
    print('q.grad passed with abs diff:', grad_diff)
    #grad_diff = (k.grad - k2.grad).abs().max()
    #assert torch.allclose(k.grad, k2.grad, atol=eps, rtol=eps), f'k.grad failed abs max: {grad_diff:.4f}'
    #print('k.grad passed with abs diff:', grad_diff)
    #print(v.grad, v2.grad)
    grad_diff = (v.grad - v2.grad).abs().max()
    assert torch.allclose(v.grad, v2.grad, atol=eps, rtol=eps), f'v.grad failed abs max: {grad_diff:.4f}'
    print('v.grad passed with abs diff:', grad_diff)
