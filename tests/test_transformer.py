import warnings

import pytest
import torch
import torch.nn as nn
from modules.transformer import TransformerBlock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, S, D, H = 2, 16, 64, 8


def make_cross_attention_block(dropout_p=0.0):
    block = TransformerBlock(emb_dim=D, num_heads=H, dropout_p=dropout_p)
    block.is_crossattention = True
    block.norm_kv = nn.LayerNorm(D)
    return block


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
    """A standard forward pass should preserve batch, sequence, and embedding dimensions."""
    out = block(x)
    assert out.shape == (B, S, D)


def test_output_shape_different_seq_len(block):
    """The block should accept different sequence lengths without changing feature size."""
    x = torch.randn(B, 32, D)
    out = block(x)
    assert out.shape == (B, 32, D)


def test_cross_attention_accepts_distinct_q_and_kv_lengths():
    """With cross-attention enabled, the block should accept different query and key/value lengths."""
    torch.manual_seed(2)
    block = make_cross_attention_block(dropout_p=0.0)

    q = torch.randn(B, 5, D)
    k = torch.randn(B, 9, D)
    v = torch.randn(B, 9, D)
    alt_k = torch.randn(B, 9, D)
    alt_v = torch.randn(B, 9, D)
    mask = torch.ones(B, 9, dtype=torch.bool)
    mask[0, 7:] = False

    out = block(q, k, v, mask=mask)
    out_with_alt_kv = block(q, alt_k, alt_v, mask=mask)

    assert out.shape == q.shape
    assert torch.isfinite(out).all()
    assert not torch.allclose(out, out_with_alt_kv)


# ---------------------------------------------------------------------------
# Finiteness
# ---------------------------------------------------------------------------

def test_output_is_finite(block, x):
    """Well-formed inputs should not produce NaN or Inf values."""
    out = block(x)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Pre-norm structure
# ---------------------------------------------------------------------------

def test_uses_layernorm():
    """The default block should expose the self-attention and MLP pre-norm layers."""
    block = TransformerBlock(emb_dim=D, num_heads=H)
    assert isinstance(block.norm1, nn.LayerNorm)
    assert isinstance(block.norm2, nn.LayerNorm)
    assert not hasattr(block, "norm_kv")


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
    """Applying a mask should still return one output vector per input token."""
    mask = torch.ones(B, S, dtype=torch.bool)
    mask[0, 8:] = False
    out = block(x, mask=mask)
    assert out.shape == (B, S, D)


def test_cross_attention_fully_masked_row_keeps_backward_finite():
    """Cross-attention should stay backward-safe when one batch item's KV mask is fully invalid."""
    torch.manual_seed(3)
    block = make_cross_attention_block(dropout_p=0.0)

    q = torch.randn(B, 5, D, requires_grad=True)
    k = torch.randn(B, 9, D, requires_grad=True)
    v = torch.randn(B, 9, D, requires_grad=True)
    mask = torch.tensor(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0]],
        dtype=torch.bool,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Anomaly Detection has been enabled.*",
            category=UserWarning,
        )
        with torch.autograd.detect_anomaly(check_nan=True):
            out = block(q, k, v, mask=mask)
            loss = out.square().mean()
            loss.backward()

    for tensor in (q, k, v):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()

    for _, parameter in block.named_parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_all_valid_mask_equals_no_mask(x):
    """A fully valid mask should behave the same as running the block without a mask."""
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
    """Multiple transformer blocks should compose cleanly while preserving shape and finiteness."""
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
    """Embedding size must be divisible by the head count for attention to split evenly."""
    with pytest.raises(AssertionError):
        TransformerBlock(emb_dim=D, num_heads=5)  # 64 % 5 != 0


def test_custom_mlp_hidden_dim(x):
    """Providing an explicit MLP hidden size should not change the outer tensor shape."""
    block = TransformerBlock(emb_dim=D, num_heads=H, mlp_hidden_dim=D * 4, dropout_p=0.0)
    out = block(x)
    assert out.shape == (B, S, D)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flow(block, x):
    """Backward pass should populate gradients for every learnable parameter used in self-attention mode."""
    out = block(x)
    out.sum().backward()
    for name, p in block.named_parameters():
        if name.startswith("norm_kv"):
            continue
        assert p.grad is not None, f"No gradient for {name}"


def test_cross_attention_gradient_flow():
    """Cross-attention mode should backpropagate through the dedicated KV normalization as well."""
    torch.manual_seed(4)
    block = make_cross_attention_block(dropout_p=0.0)

    q = torch.randn(B, 5, D)
    k = torch.randn(B, 9, D)
    v = torch.randn(B, 9, D)
    mask = torch.ones(B, 9, dtype=torch.bool)

    out = block(q, k, v, mask=mask)
    out.sum().backward()

    for name, p in block.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
