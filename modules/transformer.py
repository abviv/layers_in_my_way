import torch
import torch.nn as nn
from .attention import MultiHeadSDPA
from .mlp import MlpS


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block.

        x = x + dropout(attn(norm1(x)))
        x = x + dropout(mlp(norm2(x)))

    Uses F.scaled_dot_product_attention via MultiHeadSDPA.
    """

    def __init__(
        self,
        emb_dim,
        num_heads,
        mlp_hidden_dim=None,
        dropout_p=0.1,
        activation_fn="gelu",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = MultiHeadSDPA(emb_dim, num_heads)
        self.drop1 = nn.Dropout(dropout_p)

        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = MlpS(
            input_dim=emb_dim,
            hidden_dim=mlp_hidden_dim,
            output_dim=emb_dim,
            dropout_p=dropout_p,
            activation_fn=activation_fn,
        )
        self.drop2 = nn.Dropout(dropout_p)

    def forward(self, x, mask=None):
        # Self-attention sublayer with pre-norm
        normed = self.norm1(x)
        x = x + self.drop1(self.attn(normed, normed, normed, mask=mask))

        # MLP sublayer with pre-norm
        x = x + self.drop2(self.mlp(self.norm2(x)))

        return x


if __name__ == "__main__":

    B, S, D = 2, 16, 128
    num_heads = 8

    block = TransformerBlock(emb_dim=D, num_heads=num_heads)
    x = torch.randn(B, S, D)
    out = block(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")

    # With mask
    mask = torch.ones(B, S, dtype=torch.bool)
    mask[0, 8:] = False
    out_masked = block(x, mask=mask)
    print(f"Masked: {out_masked.shape}")

    # Stacked
    blocks = nn.ModuleList([TransformerBlock(emb_dim=D, num_heads=num_heads) for _ in range(4)])
    h = x
    for blk in blocks:
        h = blk(h)
    print(f"Stacked (4 layers): {h.shape}")
