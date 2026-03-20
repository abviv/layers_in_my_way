import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadBatched(nn.Module):
    """
    A simple batched multi-head attention implementation which supports
    both the selfAttention case (where query, key, value all have the same sequence length)
    and the crossAttention case (where query has a sequence length != key/value).
    """
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads

        assert (self.emb_dim%self.num_heads)==0, "Need to be divisible"
        self.head_dim = emb_dim//num_heads

        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, q, k, v, mask=None):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        # split B from Q since cross attention might come with different seq lens
        B, S_q, _ = q.shape # [B, seq_len_q, emb_dim]
        S_kv = k.shape[1] # [B, seq_len_kv, emb_dim]
        
        # Reshape helps for upcoming matmul op
        q = q.view(B, S_q, self.num_heads, self.head_dim).transpose(1,2) # [B, num_heads, seq_len_q, head_dim]
        k = k.view(B, S_kv, self.num_heads, self.head_dim).transpose(1,2) # [B, num_heads, seq_len_kv, head_dim]
        v = v.view(B, S_kv, self.num_heads, self.head_dim).transpose(1,2) # [B, num_heads, seq_len_kv, head_dim]

        # doing the matmul in the head dimension is more efficient since it can be done in parallel across heads
        # also K.t() since the inner dim needs to be the same for matmul
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale # [B, num_heads, seq_len_q, seq_len_kv]
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1) 
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v) # [B, num_heads, seq_len_q, head_dim]
        # need conitgous since transpose can make the memory layout non contiguous and view only works on contiguous tensors
        out = out.transpose(1,2).contiguous().view(B, S_q, self.emb_dim) # [B, seq_len_q, emb_dim]
        out = self.out_proj(out)

        return out


class MultiHeadSDPA(MultiHeadBatched):
    """
    Multi-head attention using F.scaled_dot_product_attention.
    Same interface as MultiHeadBatched but uses fused SDPA kernels
    (FlashAttention / memory-efficient attention) when available.
    """

    def forward(self, q, k, v, mask=None):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        B, S_q, _ = q.shape
        S_kv = k.shape[1]

        q = q.view(B, S_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Convert mask: parent convention [B, S_kv] where 0=invalid
        # SDPA bool mask: True=attend, False=ignore
        attn_mask = None
        if mask is not None:
            attn_mask = (mask != 0).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S_kv]

        # the newer version also supports gqa which is useful in the future.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        out = out.transpose(1, 2).contiguous().view(B, S_q, self.emb_dim)
        out = self.out_proj(out)

        return out
