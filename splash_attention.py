import torch
import torch.nn.functional as F

import helion
import helion.language as hl

import math

import sparse_attention


@helion.kernel(autotune_effort="none")
def _sparse_attn_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: hl.constexpr = False, sample: hl.constexpr = False, return_map: hl.constexpr = False):

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

    # count number of non-masked elements for averaging sparsity
    count = torch.zeros((B * nh, ), device=q.device, dtype=torch.float32)

    p_mask_avg = torch.zeros((B * nh, ), device=q.device, dtype=torch.float32)

    for tile_b in hl.tile(B * nh):
        #p_mask = p_mask_avg[tile_b]
        for tile_q in hl.tile(T):
            _max_logits = max_logits[tile_b, tile_q]
            _lse = lse[tile_b, tile_q]
            qs = q[tile_b, tile_q, :]
            for tile_k in hl.tile(T):
                ks = k[tile_b, tile_k, :]
                logits = (qs @ ks.transpose(-1, -2) * scale)

                if causal:
                    causal_mask = tile_q.index[:, None] >= tile_k.index[None, :]
                    count[tile_b] += causal_mask.sum(dim=-1).sum(dim=-1)
                    logits = torch.where(causal_mask, logits, float('-inf'))
                else:
                    count[tile_b] += tile_k.count * tile_q.count
                #p_mask = p_mask + logits.sigmoid().sum(dim=-1).sum(dim=-1)

                new_max_logits = torch.maximum(_max_logits, logits.amax(dim=-1))
                ratio = torch.exp(_max_logits - new_max_logits)
                exp_weights = torch.exp(logits - new_max_logits[:, :, None])
                curr_lse = exp_weights.sum(dim=-1)
                new_lse = _lse * ratio + curr_lse

                if sample:
                    rand = torch.logit(torch.rand_like(logits))
                    mask = torch.where(logits + rand > 0, 1., 0.)
                else:
                    mask = torch.where(logits > 0, 1., 0.)

                if return_map:
                    adj[tile_b, tile_q, tile_k] = mask
                weights = mask * exp_weights
                curr_out = torch.matmul(weights, v[tile_b, tile_q, :])
                out_old = _lse[:, :, None] * out[tile_b, tile_q, :] * ratio[:, :, None]
                out[tile_b, tile_q, :] = (out_old + curr_out) / new_lse[:, :, None]

                _max_logits = new_max_logits
                _lse = new_lse
            lse[tile_b, tile_q] = _lse
            max_logits[tile_b, tile_q] = _max_logits
        #p_mask_avg[tile_b] = p_mask
    #p_mask_avg = p_mask_avg / count
    return out.view(B, nh, T, hs), p_mask_avg.view(B, nh), adj, max_logits, lse


@helion.kernel(autotune_effort="none")
def _sparse_attn_mask_bwd(q: torch.Tensor, k: torch.Tensor, causal: hl.constexpr = False):

    B, nh, T, hs = q.shape

    q = q.view(-1, T, hs)
    k = k.view(-1, T, hs)

    scale = 1 / math.sqrt(q.size(-1))
    grad_q = torch.empty_like(q)
    grad_k = torch.empty_like(k)

    if causal:
        count = float((1 + T) * T // 2)
    else:
        count = float(T * T)

    for tile_b in hl.tile(B * nh):
        for tile_q in hl.tile(T):
            qs = q[tile_b, tile_q, :]
            for tile_k in hl.tile(T):
                ks = k[tile_b, tile_k, :]
                logits = (qs @ ks.transpose(-1, -2) * scale)

                prob = logits.sigmoid()
                grad_sigmoid = prob * (1 - prob) * scale

                grad_q[tile_b, tile_q, :] = torch.baddbmm(grad_q[tile_b, tile_q, :], grad_sigmoid, ks)
                grad_k[tile_b, tile_k, :] = torch.baddbmm(grad_k[tile_b, tile_q, :], grad_sigmoid.transpose(-1, -2), qs)

    grad_q = grad_q.view(B, nh, T, hs) / count
    grad_k = grad_k.view(B, nh, T, hs) / count
    return grad_q, grad_k


@helion.kernel(autotune_effort="none")
def _sparse_attn_bwd(q: torch.Tensor, k: torch.Tensor, causal: hl.constexpr = False):

    B, nh, T, hs = q.shape

    q = q.view(-1, T, hs)
    k = k.view(-1, T, hs)

    scale = 1 / math.sqrt(q.size(-1))
    grad_q = torch.empty_like(q)
    grad_k = torch.empty_like(k)

    if causal:
        count = float((1 + T) * T // 2)
    else:
        count = float(T * T)

    for tile_b in hl.tile(B * nh):
        for tile_q in hl.tile(T):
            qs = q[tile_b, tile_q, :]
            for tile_k in hl.tile(T):
                ks = k[tile_b, tile_k, :]
                logits = (qs @ ks.transpose(-1, -2) * scale)

                prob = logits.sigmoid()
                grad_sigmoid = prob * (1 - prob) * scale

                grad_q[tile_b, tile_q, :] = torch.baddbmm(grad_q[tile_b, tile_q, :], grad_sigmoid, ks)
                grad_k[tile_b, tile_k, :] = torch.baddbmm(grad_k[tile_b, tile_q, :], grad_sigmoid.transpose(-1, -2), qs)

    grad_q = grad_q.view(B, nh, T, hs) / count
    grad_k = grad_k.view(B, nh, T, hs) / count
    return grad_q, grad_k


class SparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal: bool = False, sample: bool = False, return_map: bool = False):

        assert q.shape == k.shape and k.shape == v.shape

        out, p_mask, adj, max_logits, lse = _sparse_attn_fwd(q, k, v, causal, sample, return_map)

        ctx.save_for_backward(q, k, v, out, p_mask, max_logits, lse)
        ctx.causal = causal
        ctx.sample = sample
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
        grad_q, grad_k = _sparse_attn_mask_bwd(q, k, causal)
        grad_v = torch.zeros_like(v)

        return grad_q, grad_k, grad_v, None, None, None


sparse_attention_ = SparseAttention.apply

if __name__ == '__main__':

    q = torch.randn(1, 2, 10, 10, device='cuda', dtype=torch.float32)
    q.requires_grad = True
    q2 = q.detach().clone()
    q2.requires_grad = True
    
    h, p_mask, adj = sparse_attention_(q, q, q, True, False, False)
    gold, gold_p_mask, gold_adj = sparse_attention._sparse_attention_torch(q2, q2, q2, True, False, False)
    print('causal: ', torch.allclose(h, gold, atol=1e-2, rtol=1e-2), (gold - h).abs().max(), p_mask, gold_p_mask)
    h, p_mask, adj = sparse_attention_(q, q, q, False, False, False)
    gold, gold_p_mask, gold_adj = sparse_attention._sparse_attention_torch(q2, q2, q2, False, False, False)

    print('noncausal: ', torch.allclose(h, gold, atol=1e-2, rtol=1e-2), (gold - h).abs().max(), p_mask, gold_p_mask)
