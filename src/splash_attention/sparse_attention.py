import math
import torch
import torch.nn.functional as F


def sparse_attention_naive(q, k, v, bias_gate, causal, sample, return_att):
    # manual implementation of attention
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

    # causal mask
    if causal:
        T = q.size(-2)
        mask = torch.tril(torch.ones(T, T, device=q.device, dtype=q.dtype)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float("-inf"))
        count = mask.sum()[None, None]
    else:
        count = q.size(-2) ** 2

    att_gate = att + bias_gate
    edge_samples = gumbel_sample(att_gate, sample=sample)

    sm_att = F.softmax(att, dim=-1)
    masked_att_weights = sm_att * edge_samples

    # masked_att_weights = self.attn_dropout(masked_att_weights)
    y = masked_att_weights @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

    adj = att_gate > 0

    return y, att_gate.sigmoid().sum(dim=[-1, -2]) / count, adj


def gumbel_sample(x: torch.Tensor, sample: bool = False) -> torch.Tensor:
    """
    Gumbel-softmax sampling. Takes logits and returns samples in [0, 1].
    """
    if sample:
        logistics = torch.logit(torch.rand_like(x))
    else:
        logistics = 0
    samples = (logistics + x) > 0
    return (
        samples.to(x.dtype)
        + torch.sigmoid(x + logistics)
        - torch.sigmoid(x + logistics).detach()
    )
