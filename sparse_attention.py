import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseSelfAttention(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 bias,
                 dropout,
                 ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.n_head = num_heads
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        # output projection
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout

    def forward(self, x, is_causal=False, sample=True, return_att=False):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = torch.split(
            qkv, [self.embed_dim, self.embed_dim, self.embed_dim], dim=2
        )
        q = q.view(B, T, self.n_head, self.embed_dim // self.n_head)
        k = k.view(B, T, self.n_head, self.embed_dim // self.n_head)
        v = v.view(B, T, self.n_head, self.embed_dim // self.n_head)
        # (B, nh, T, hs) for q, k, v

        q = q.transpose(1, 2)  # (B, nh, T, hs)
        k = k.transpose(1, 2)  # (B, nh, T, hs)
        v = v.transpose(1, 2)  # (B, nh, T, hs)

        y, l1, att = _sparse_attention_torch(q, k, v, is_causal, sample, return_att)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        # returns the attention pattern for regulaisation
        return y, att


def _sparse_attention_torch(q, k, v, bias_gate, causal, sample, return_att):
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


if __name__ == "__main__":
    # super simple shape test
    B, T, C = 4, 8, 32  # batch size, sequence length, embedding dim
    x = torch.randn(B, T, C)

    attn = SparseSelfAttention(
        embed_dim=C,
        num_heads=4,
        bias=True,
        dropout=0.1,
    )
    y, attn_weights = attn(x, is_causal=True, sample=True)
    print(y.shape)
    print("should be ({}, {}, {})".format(B, T, C))
    print(attn_weights.shape)  # should be (B,nh,T,T)
    print("should be ({}, {}, {}, {})".format(B, attn.n_head, T, T))

    y, attn_weights = attn(x, is_causal=True, sample=False)
    print(y.shape)
    print("should be ({}, {}, {})".format(B, T, C))
    print(attn_weights.shape)  # should be (B,nh,T,T)
    print("should be ({}, {}, {}, {})".format(B, attn.n_head, T, T))

    y, attn_weights = attn(x, is_causal=False, sample=True)
    print(y.shape)
    print("should be ({}, {}, {})".format(B, T, C))
    print(attn_weights.shape)  # should be (B,nh,T,T)
    print("should be ({}, {}, {}, {})".format(B, attn.n_head, T, T))
    y, attn_weights = attn(x, is_causal=False, sample=False)

    print(y.shape)
    print("should be ({}, {}, {})".format(B, T, C))
    print(attn_weights.shape)  # should be (B,nh,T,T)
    print("should be ({}, {}, {}, {})".format(B, attn.n_head, T, T))


