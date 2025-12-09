import torch
import torch.nn.functional as F

import helion
import helion.language as hl

import math

import random


@helion.kernel(
    autotune_effort="none",
    static_shapes=False,
    ignore_warnings=[helion.exc.TensorOperationInWrapper]
)
def _sparse_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias_gate: hl.constexpr = 0,
    causal: hl.constexpr = False,
    sample: hl.constexpr = False,
    return_map: hl.constexpr = False,
    seed: int = 0,
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

    lse = torch.zeros((B * nh, T), device=q.device, dtype=torch.float32)

    # count number of non-masked elements for averaging sparsity (trivial for now, but can be useful for future)
    if causal:
        count = float(T * (T + 1) // 2)
    else:
        count = float(T * T)

    p_mask_avg = torch.zeros((B * nh, ), device=q.device, dtype=torch.float32)

    for tile_b in hl.tile(B * nh):
        for tile_q in hl.tile(T):
            max_logits = hl.full([tile_b, tile_q], float('-inf'), device=q.device, dtype=torch.float32)
            sumexp = hl.zeros([tile_b, tile_q], device=q.device, dtype=torch.float32)
            qs = q[tile_b, tile_q, :].float()
            for tile_k in hl.tile(T):
                ks = k[tile_b, tile_k, :].float()
                logits = (qs @ ks.transpose(-1, -2) * scale)

                if causal:
                    causal_mask = tile_q.index[:, None] >= tile_k.index[None, :]
                    logits = torch.where(causal_mask, logits, float('-inf'))
                p_mask_avg[tile_b] += (logits + bias_gate).sigmoid().sum(dim=-1).sum(dim=-1)

                new_max_logits = torch.maximum(max_logits, logits.amax(dim=-1))
                old_sumexp = sumexp * torch.exp(max_logits - new_max_logits)
                exp_weights = torch.exp(logits - new_max_logits[:, :, None])
                new_sumexp = old_sumexp + exp_weights.sum(dim=-1)

                if sample:
                    rand = torch.logit(hl.rand(logits.shape, seed=seed, device=q.device))
                    sparse_mask = logits + rand + bias_gate > 0
                else:
                    sparse_mask = logits + bias_gate > 0

                if return_map:
                    if causal:
                        adj[tile_b, tile_q, tile_k] = sparse_mask & causal_mask
                    else:
                        adj[tile_b, tile_q, tile_k] = sparse_mask
                weights = torch.where(sparse_mask, exp_weights, 0) / new_sumexp[:, :, None]
                out_old = (old_sumexp / new_sumexp)[:, :, None] * out[tile_b, tile_q, :]
                out[tile_b, tile_q, :] = torch.baddbmm(out_old, weights, v[tile_b, tile_k, :].float())
                max_logits = new_max_logits
                sumexp = new_sumexp
            lse[tile_b, tile_q] = max_logits + torch.log(sumexp)
    p_mask_avg.div_(count)

    if return_map:
        adj = adj.view(B, nh, T, T)

    return (
        out.view(B, nh, T, hs).to(dtype=q.dtype),
        p_mask_avg.view(B, nh).to(dtype=q.dtype),
        adj,
        lse
    )


@helion.kernel(
    autotune_effort="none",
    static_shapes=False,
    ignore_warnings=[helion.exc.TensorOperationInWrapper]
)
def _sparse_attn_bwd(
    grad_out: torch.Tensor,
    grad_mask: torch.Tensor,
    go: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    bias_gate: float = 0.,
    causal: hl.constexpr = False,
    sample: hl.constexpr = False,
    seed: int = 0,
):

    B, nh, T, hs = q.shape

    out = out.view(-1, T, hs)

    q = q.view(-1, T, hs)
    k = k.view(-1, T, hs)
    v = v.view(-1, T, hs)

    grad_out = grad_out.view(-1, T, hs)
    grad_mask = grad_mask.view(-1)
    go = go.view(-1, T, 1)
    lse = lse.view(-1, T, 1)

    scale = 1 / math.sqrt(q.size(-1))
    grad_q = torch.zeros_like(q, dtype=torch.float32)
    grad_k = torch.zeros_like(k, dtype=torch.float32)
    grad_v = torch.zeros_like(v, dtype=torch.float32)

    if causal:
        count = float(T * (T + 1) // 2)
    else:
        count = float(T * T)

    for tile_b in hl.tile(B * nh):
        _grad_mask = grad_mask[tile_b].float()
        for tile_q in hl.tile(T):
            _lse = lse[tile_b, tile_q, :]
            qs = q[tile_b, tile_q, :].float()
            _grad_out = grad_out[tile_b, tile_q, :].float()
            _go = go[tile_b, tile_q, :].float()

            for tile_k in hl.tile(T):
                ks = k[tile_b, tile_k, :].float()
                logits = (qs @ ks.transpose(-1, -2) * scale)

                if causal:
                    causal_mask = tile_q.index[:, None] >= tile_k.index[None, :]
                    logits = torch.where(causal_mask, logits, float('-inf'))

                if sample:
                    rand = torch.logit(hl.rand(logits.shape, seed=seed, device=q.device))
                    sparse_mask = logits + rand + bias_gate > 0
                    gate = (logits + rand + bias_gate).sigmoid()
                    gate_prob = (logits + bias_gate).sigmoid()
                else:
                    sparse_mask = logits + bias_gate > 0
                    gate = (logits + bias_gate).sigmoid()
                    gate_prob = gate

                attn_weights = torch.exp(logits - _lse)
                final_weights = torch.where(sparse_mask, attn_weights, 0)

                grad_v[tile_b, tile_k, :] = torch.baddbmm(
                    grad_v[tile_b, tile_k, :],
                    final_weights.transpose(-1, -2),
                    _grad_out,
                )

                gvt = torch.matmul(
                    _grad_out,
                    v[tile_b, tile_k, :].transpose(-1, -2).float()
                )

                grad_score = attn_weights * (((gate * (1 - gate)) + sparse_mask.to(q.dtype)) * gvt - _go)

                # sparsity bwd
                grad_sigmoid = _grad_mask[:, None, None] * gate_prob * (1 - gate_prob) / count

                grad_both = scale * (grad_score + grad_sigmoid)

                grad_q[tile_b, tile_q, :] = torch.baddbmm(grad_q[tile_b, tile_q, :], grad_both, ks)
                grad_k[tile_b, tile_k, :] = torch.baddbmm(grad_k[tile_b, tile_k, :], grad_both.transpose(-1, -2), qs)

    return (
        grad_q.view(B, nh, T, hs).to(dtype=q.dtype),
        grad_k.view(B, nh, T, hs).to(dtype=q.dtype),
        grad_v.view(B, nh, T, hs).to(dtype=q.dtype)
    )


class SplashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias_gate: float = 0, causal: bool = False, sample: bool = False, return_map: bool = False):

        assert q.shape == k.shape and k.shape == v.shape

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        ctx.bias_gate = bias_gate
        ctx.causal = causal
        ctx.sample = sample
        seed = random.randint(0, 2 ** 31)
        ctx.seed = seed

        out, p_mask, adj, lse = _sparse_attn_fwd(q, k, v, bias_gate, causal, sample, return_map, seed)

        ctx.save_for_backward(q, k, v, out, p_mask, lse)

        if isinstance(adj, torch.Tensor):
            ctx.mark_non_differentiable(adj)

        return out, p_mask, adj

    @staticmethod
    def backward(ctx, grad_out, grad_p_mask, dummy):

        q, k, v, out, p_mask, lse = ctx.saved_tensors
        bias_gate = ctx.bias_gate
        causal = ctx.causal
        sample = ctx.sample
        seed = ctx.seed

        go = (grad_out[..., None, :] @ out[..., None]).squeeze([-1, -2])

        grad_q, grad_k, grad_v = _sparse_attn_bwd(
            grad_out.contiguous(),
            grad_p_mask.contiguous(),
            go, q, k, v, out, lse, bias_gate,
            causal, sample, seed
        )

        return grad_q, grad_k, grad_v, None, None, None, None


splash_attention = SplashAttention.apply
