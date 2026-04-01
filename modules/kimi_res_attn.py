"""
Attention Residuals — PyTorch reference implementation.
Paper: "Attention Residuals" (Kimi Team, 2026)
https://github.com/MoonshotAI/Attention-Residuals

Implements standard residuals, Full AttnRes, and Block AttnRes.
B=batch, T=sequence length, d=hidden dim, L=total layers, N=blocks, S=layers/block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class DummyAttn(nn.Module):
    """Placeholder self-attention (linear proj for shape demo)."""

    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DummyMLP(nn.Module):
    """Placeholder MLP (linear proj for shape demo)."""

    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# Standard Residuals

class StandardTransformerBlock(nn.Module):
    """Single Transformer block: h = h + f(norm(h))."""

    def __init__(self, d: int):
        super().__init__()
        self.attn_norm = RMSNorm(d)
        self.attn = DummyAttn(d)
        self.mlp_norm = RMSNorm(d)
        self.mlp = DummyMLP(d)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = h + self.attn(self.attn_norm(h))
        h = h + self.mlp(self.mlp_norm(h))
        return h


class StandardResidualModel(nn.Module):
    """
    h_l = h_{l-1} + f_{l-1}(h_{l-1})
    Hidden state magnitude grows as O(L).
    """

    def __init__(self, d: int, num_transformer_blocks: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [StandardTransformerBlock(d) for _ in range(num_transformer_blocks)]
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            h = block(h)
        return h


# Full Attention Residuals

def full_attn_res(
    w_l: torch.Tensor,
    sources: list[torch.Tensor],
    norm: RMSNorm,
) -> torch.Tensor:
    """
    Softmax attention over all previous layer outputs (Eq. 2-4).

    phi(q, k) = exp(q^T RMSNorm(k))
    alpha_{i->l} = phi(w_l, k_i) / sum_j phi(w_l, k_j)
    h_l = sum_i alpha_{i->l} * v_i
    """
    V = torch.stack(sources, dim=0)       # [num_sources, B, T, d]
    K = norm(V)
    logits = torch.einsum("d, n b t d -> n b t", w_l, K)
    alpha = logits.softmax(dim=0)
    return torch.einsum("n b t, n b t d -> b t d", alpha, V)


class FullAttnResTransformerBlock(nn.Module):
    """
    Transformer block with Full Attention Residuals.

    Before each sub-layer, replaces the standard residual with softmax
    attention over all preceding layer outputs via a learned pseudo-query w_l.
    """

    def __init__(self, d: int):
        super().__init__()
        self.attn_norm = RMSNorm(d)
        self.attn = DummyAttn(d)
        self.mlp_norm = RMSNorm(d)
        self.mlp = DummyMLP(d)

        self.attn_res_query = nn.Parameter(torch.zeros(d))
        self.attn_res_norm = RMSNorm(d)
        self.mlp_res_query = nn.Parameter(torch.zeros(d))
        self.mlp_res_norm = RMSNorm(d)

    def forward(self, sources: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            sources: all previous layer outputs; sources[0] = embedding

        Returns:
            sources with two new entries appended (attn_out, mlp_out)
        """
        h = full_attn_res(self.attn_res_query, sources, self.attn_res_norm)
        sources.append(self.attn(self.attn_norm(h)))

        h = full_attn_res(self.mlp_res_query, sources, self.mlp_res_norm)
        sources.append(self.mlp(self.mlp_norm(h)))

        return sources


class FullAttnResModel(nn.Module):
    """
    Full Attention Residuals (Section 3.1).
    Memory: O(L·d). Compute: O(L²·d).
    """

    def __init__(self, d: int, num_transformer_blocks: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [FullAttnResTransformerBlock(d) for _ in range(num_transformer_blocks)]
        )
        self.out_res_query = nn.Parameter(torch.zeros(d))
        self.out_res_norm = RMSNorm(d)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        sources: list[torch.Tensor] = [h]
        for block in self.blocks:
            sources = block(sources)
        return full_attn_res(self.out_res_query, sources, self.out_res_norm)


# Block Attention Residuals

def block_attn_res(
    completed_blocks: list[torch.Tensor],
    partial_block: torch.Tensor,
    w_l: torch.Tensor,
    norm: RMSNorm,
) -> torch.Tensor:
    """
    Inter-block attention over completed block reps + current partial sum (Eq. 6).

    RMSNorm on keys prevents magnitude bias between full blocks (S outputs summed)
    and partial sums (fewer outputs).
    """
    V = torch.stack(completed_blocks + [partial_block], dim=0)
    K = norm(V)
    logits = torch.einsum("d, n b t d -> n b t", w_l, K)
    alpha = logits.softmax(dim=0)
    return torch.einsum("n b t, n b t d -> b t d", alpha, V)


class BlockAttnResTransformerBlock(nn.Module):
    """
    Transformer block for Block AttnRes.

    Tracks completed_blocks (finalized block reps b_0..b_{n-1}) and
    partial_block (running intra-block sum). At block boundaries, partial_block
    is committed to completed_blocks.
    """

    def __init__(self, d: int, block_size: int, block_layer_offset: int):
        super().__init__()
        self.block_size = block_size
        self.block_layer_offset = block_layer_offset

        self.attn_norm = RMSNorm(d)
        self.attn = DummyAttn(d)
        self.mlp_norm = RMSNorm(d)
        self.mlp = DummyMLP(d)

        self.attn_res_query = nn.Parameter(torch.zeros(d))
        self.attn_res_norm = RMSNorm(d)
        self.mlp_res_query = nn.Parameter(torch.zeros(d))
        self.mlp_res_norm = RMSNorm(d)

    def _inter_block_attn(
        self,
        completed_blocks: list[torch.Tensor],
        partial_block: torch.Tensor | None,
        w_l: nn.Parameter,
        norm: RMSNorm,
    ) -> torch.Tensor:
        if partial_block is None:
            V = torch.stack(completed_blocks, dim=0)
            K = norm(V)
            logits = torch.einsum("d, n b t d -> n b t", w_l, K)
            alpha = logits.softmax(dim=0)
            return torch.einsum("n b t, n b t d -> b t d", alpha, V)
        return block_attn_res(completed_blocks, partial_block, w_l, norm)

    def forward(
        self,
        completed_blocks: list[torch.Tensor],
        partial_block: torch.Tensor | None,
        layer_in_block: int,
    ) -> tuple[list[torch.Tensor], torch.Tensor | None, int]:

        # Attn sub-layer
        if layer_in_block == self.block_size:
            completed_blocks.append(partial_block)
            partial_block = None
            layer_in_block = 0

        h = self._inter_block_attn(completed_blocks, partial_block, self.attn_res_query, self.attn_res_norm)
        attn_out = self.attn(self.attn_norm(h))
        partial_block = attn_out if partial_block is None else partial_block + attn_out
        layer_in_block += 1

        # MLP sub-layer
        if layer_in_block == self.block_size:
            completed_blocks.append(partial_block)
            partial_block = None
            layer_in_block = 0

        h = self._inter_block_attn(completed_blocks, partial_block, self.mlp_res_query, self.mlp_res_norm)
        mlp_out = self.mlp(self.mlp_norm(h))
        partial_block = mlp_out if partial_block is None else partial_block + mlp_out
        layer_in_block += 1

        return completed_blocks, partial_block, layer_in_block


class BlockAttnResModel(nn.Module):
    """
    Block Attention Residuals (Section 3.2).

    L layers grouped into N = L/S blocks.
    Intra-block: standard residual → b_n = Σ_{j ∈ B_n} f_j(h_j)
    Inter-block: softmax attention over [b_0, ..., b_{n-1}, b_n^i]

    Paper's 48B model: L=54, S=6, N=9 blocks + embedding = 10 sources.
    Memory: O(N·d) vs O(L·d) for Full AttnRes.
    """

    def __init__(self, d: int, num_transformer_blocks: int, block_size: int = 6):
        super().__init__()
        self.block_size = block_size
        self.blocks = nn.ModuleList(
            [BlockAttnResTransformerBlock(d, block_size, i * 2)
             for i in range(num_transformer_blocks)]
        )
        self.out_res_query = nn.Parameter(torch.zeros(d))
        self.out_res_norm = RMSNorm(d)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        completed_blocks: list[torch.Tensor] = [h]
        partial_block: torch.Tensor | None = None
        layer_in_block = 0

        for block in self.blocks:
            completed_blocks, partial_block, layer_in_block = block(
                completed_blocks, partial_block, layer_in_block
            )

        if partial_block is not None:
            completed_blocks.append(partial_block)

        return full_attn_res(self.out_res_query, completed_blocks, self.out_res_norm)


def main():
    torch.manual_seed(42)

    B, T, d = 2, 128, 256
    num_transformer_blocks = 9
    block_size = 6
    L = num_transformer_blocks * 2
    N = L // block_size
    S = block_size

    print("=" * 70)
    print("Attention Residuals — Shape & Forward Pass Demo")
    print("=" * 70)
    print(f"  B={B}, T={T}, d={d}, blocks={num_transformer_blocks}, L={L}, S={S}, N={N}")

    x = torch.randn(B, T, d)

    print("\n--- Standard Residuals ---")
    model_std = StandardResidualModel(d, num_transformer_blocks)
    out_std = model_std(x.clone())
    print(f"  Output: {list(out_std.shape)}, ||h_L|| mean: {out_std.norm(dim=-1).mean():.2f}")
    print(f"  Params: {sum(p.numel() for p in model_std.parameters()):,}")

    print("\n--- Full Attention Residuals ---")
    model_full = FullAttnResModel(d, num_transformer_blocks)
    print("\n  Forward pass shape trace:")
    print(f"    sources = [embedding]  →  len=1, each [{B}, {T}, {d}]")
    for i in range(num_transformer_blocks):
        num_src_pre_attn = 1 + i * 2
        num_src_pre_mlp = num_src_pre_attn + 1
        print(f"\n    Transformer block {i+1}:")
        print(f"      Pre-Attn AttnRes:")
        print(f"        V = stack(sources)       : [{num_src_pre_attn}, {B}, {T}, {d}]")
        print(f"        K = RMSNorm(V)           : [{num_src_pre_attn}, {B}, {T}, {d}]")
        print(f"        logits = einsum(w_l, K)  : [{num_src_pre_attn}, {B}, {T}]")
        print(f"        alpha  = softmax(logits) : [{num_src_pre_attn}, {B}, {T}]")
        print(f"        h_l    = einsum(alpha, V)    : [{B}, {T}, {d}]")
        print(f"      Attn(h_l)                  : [{B}, {T}, {d}]  → sources len={num_src_pre_attn+1}")
        print(f"      Pre-MLP AttnRes:")
        print(f"        V = stack(sources)       : [{num_src_pre_mlp}, {B}, {T}, {d}]")
        print(f"        h_l                      : [{B}, {T}, {d}]")
        print(f"      MLP(h_l)                   : [{B}, {T}, {d}]  → sources len={num_src_pre_mlp+1}")
    print(f"\n    Final aggregation:")
    print(f"      V_final = stack(sources)   : [{1 + num_transformer_blocks * 2}, {B}, {T}, {d}]")
    print(f"      h_out                      : [{B}, {T}, {d}]")

    out_full = model_full(x.clone())
    print(f"\n  Output: {list(out_full.shape)}, ||h_L|| mean: {out_full.norm(dim=-1).mean():.2f}")
    n_std = sum(p.numel() for p in model_std.parameters())
    n_full = sum(p.numel() for p in model_full.parameters())
    print(f"  Params: {n_full:,}  (extra over baseline: {n_full - n_std:,})")

    print("\n--- Block Attention Residuals ---")
    model_block = BlockAttnResModel(d, num_transformer_blocks, block_size=block_size)
    print(f"\n  Forward pass shape trace:")
    print(f"    completed_blocks = [b_0=emb]  →  len=1, each [{B}, {T}, {d}]")
    print(f"    partial_block = None")

    layer_in_block = 0
    n_completed = 1
    for i in range(num_transformer_blocks):
        print(f"\n    Transformer block {i+1}:")
        for sub, name in [(0, "Attn"), (1, "MLP")]:
            if layer_in_block == block_size:
                n_completed += 1
                layer_in_block = 0
                print(f"      ── Block boundary! partial → completed (len={n_completed}), partial=None ──")

            if layer_in_block == 0:
                ns = n_completed
                print(f"      Pre-{name} AttnRes (1st in block, no partial):")
                print(f"        V = stack(completed)        : [{ns}, {B}, {T}, {d}]")
            else:
                ns = n_completed + 1
                print(f"      Pre-{name} AttnRes (with partial):")
                print(f"        V = stack(completed+partial) : [{ns}, {B}, {T}, {d}]")

            print(f"        K = RMSNorm(V)               : [{ns}, {B}, {T}, {d}]")
            print(f"        logits                       : [{ns}, {B}, {T}]")
            print(f"        alpha                        : [{ns}, {B}, {T}]")
            print(f"        h_l                          : [{B}, {T}, {d}]")
            print(f"      {name}(h_l) → partial += out     : [{B}, {T}, {d}]")
            layer_in_block += 1

    if layer_in_block > 0:
        n_completed += 1
    print(f"\n    Final aggregation:")
    print(f"      V_final = stack(all blocks)    : [{n_completed}, {B}, {T}, {d}]")
    print(f"      h_out                          : [{B}, {T}, {d}]")

    out_block = model_block(x.clone())
    print(f"\n  Output: {list(out_block.shape)}, ||h_L|| mean: {out_block.norm(dim=-1).mean():.2f}")
    print(f"  Params: {sum(p.numel() for p in model_block.parameters()):,}")

    print("\n--- Memory Comparison ---")
    print(f"  Standard     : 1 × d = {d} floats")
    print(f"  Full AttnRes : L × d = {L} × {d} = {L*d:,} floats")
    print(f"  Block AttnRes: N × d = {N} × {d} = {N*d:,} floats  ({L*d/(N*d):.1f}× reduction)")
    print(f"\n  I/O per layer (Table 1):")
    print(f"    Standard     : 3d = {3*d:,}")
    print(f"    Full AttnRes : (S+N)d = ({S}+{N})×{d} = {(S+N)*d:,}")
    print(f"    Block AttnRes: (N/S+5)d = ({N/S:.1f}+5)×{d} = {int((N/S+5)*d):,}")
    print(f"    mHC (m=4)    : ~34d = {34*d:,}")


if __name__ == "__main__":
    main()
