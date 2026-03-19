import pytest
import torch
import torch.nn as nn
from transformer import TransformerBlock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, S, D, H = 2, 16, 64, 8


@pytest.fixture
def block():
    torch.manual_seed(0)
    return TransformerBlock(emb_dim=D, num_heads=H, dropout_p=0.0)


@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(B, S, D)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_output_shape(block, x):
    out = block(x)
    assert out.shape == (B, S, D)


def test_output_shape_different_seq_len(block):
    x = torch.randn(B, 32, D)
    out = block(x)
    assert out.shape == (B, 32, D)


# ---------------------------------------------------------------------------
# Finiteness
# ---------------------------------------------------------------------------

def test_output_is_finite(block, x):
    out = block(x)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Pre-norm structure
# ---------------------------------------------------------------------------

def test_uses_layernorm():
    block = TransformerBlock(emb_dim=D, num_heads=H)
    assert isinstance(block.norm1, nn.LayerNorm)
    assert isinstance(block.norm2, nn.LayerNorm)


def test_residual_connection(x):
    """With zeroed sublayer weights, output should approximately equal input
    (residual path dominates)."""
    torch.manual_seed(0)
    block = TransformerBlock(emb_dim=D, num_heads=H, dropout_p=0.0)

    with torch.no_grad():
        # Zero out attention output projection
        block.attn.out_proj.weight.zero_()
        block.attn.out_proj.bias.zero_()
        # Zero out MLP output layer
        block.mlp.fc2.weight.zero_()
        block.mlp.fc2.bias.zero_()

    out = block(x)
    assert torch.allclose(out, x, atol=1e-5)


# ---------------------------------------------------------------------------
# Mask tests
# ---------------------------------------------------------------------------

def test_with_mask(block, x):
    mask = torch.ones(B, S, dtype=torch.bool)
    mask[0, 8:] = False
    out = block(x, mask=mask)
    assert out.shape == (B, S, D)


def test_all_valid_mask_equals_no_mask(x):
    torch.manual_seed(0)
    block = TransformerBlock(emb_dim=D, num_heads=H, dropout_p=0.0)
    block.eval()

    mask = torch.ones(B, S, dtype=torch.bool)
    out_masked = block(x, mask=mask)
    out_unmasked = block(x)
    assert torch.allclose(out_masked, out_unmasked, atol=1e-5)


# ---------------------------------------------------------------------------
# Stacking
# ---------------------------------------------------------------------------

def test_stacked_blocks(x):
    torch.manual_seed(0)
    blocks = nn.ModuleList([
        TransformerBlock(emb_dim=D, num_heads=H, dropout_p=0.0) for _ in range(4)
    ])

    h = x
    for blk in blocks:
        h = blk(h)
    assert h.shape == (B, S, D)
    assert torch.isfinite(h).all()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_invalid_num_heads_raises():
    with pytest.raises(AssertionError):
        TransformerBlock(emb_dim=D, num_heads=5)  # 64 % 5 != 0


def test_custom_mlp_hidden_dim(x):
    block = TransformerBlock(emb_dim=D, num_heads=H, mlp_hidden_dim=D * 4, dropout_p=0.0)
    out = block(x)
    assert out.shape == (B, S, D)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flow(block, x):
    out = block(x)
    out.sum().backward()
    for name, p in block.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
