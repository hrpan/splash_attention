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

    def forward(self, x, is_causal=False, sample=True):
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

        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # causal mask
        if is_causal:
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            att = att.masked_fill(mask == 0, float("-inf"))

        if sample:
            edge_samples = self.gumbel_sample(att)
        else:
            edge_samples = (att > 0).float()
        sm_att = F.softmax(att, dim=-1)
        masked_att_weights = sm_att * edge_samples

        masked_att_weights = self.attn_dropout(masked_att_weights)
        y = masked_att_weights @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        # returns the attention pattern for regulaisation
        return y, att

    def gumbel_sample(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gumbel-softmax sampling. Takes logits and returns samples in [0, 1].
        """
        logistics = torch.logit(torch.rand_like(x))
        samples = (logistics + x) > 0
        return (
            samples.float()
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


