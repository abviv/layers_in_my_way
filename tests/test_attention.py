import pytest
import torch
from modules.attention import MultiHeadBatched


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    torch.manual_seed(0)
    return MultiHeadBatched(emb_dim=128, num_heads=8)


@pytest.fixture
def self_attn_inputs():
    """Query, key, value all share the same sequence length (self-attention)."""
    B, S, D = 2, 10, 128
    torch.manual_seed(1)
    q = torch.randn(B, S, D)
    k = torch.randn(B, S, D)
    v = torch.randn(B, S, D)
    return q, k, v


@pytest.fixture
def cross_attn_inputs():
    """Query has a shorter sequence than key/value (cross-attention)."""
    B, S_q, S_kv, D = 2, 5, 10, 128
    torch.manual_seed(1)
    q = torch.randn(B, S_q, D)
    k = torch.randn(B, S_kv, D)
    v = torch.randn(B, S_kv, D)
    return q, k, v


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_self_attention_output_shape(model, self_attn_inputs):
    """Output must have the same shape as the query: [B, S_q, emb_dim]."""
    q, k, v = self_attn_inputs
    out = model(q, k, v)
    assert out.shape == q.shape


def test_cross_attention_output_shape(model, cross_attn_inputs):
    """In cross-attention S_q != S_kv; output shape must still follow query."""
    q, k, v = cross_attn_inputs
    out = model(q, k, v)
    assert out.shape == q.shape


def test_all_valid_mask_equals_no_mask(model, cross_attn_inputs):
    """A mask of all True (all positions valid) should give the same result as no mask at all."""
    q, k, v = cross_attn_inputs
    B, S_kv = q.shape[0], k.shape[1]
    mask = torch.ones(B, S_kv, dtype=torch.bool)  # every KV position is valid

    out_masked   = model(q, k, v, mask=mask)
    out_unmasked = model(q, k, v)

    assert torch.allclose(out_masked, out_unmasked)


def test_partial_mask_output_shape(model, cross_attn_inputs):
    """Output shape must be correct even when some KV positions are masked out."""
    q, k, v = cross_attn_inputs
    B, S_kv = q.shape[0], k.shape[1]
    mask = torch.ones(B, S_kv, dtype=torch.bool)
    mask[0, 5:] = False  # mask the last 5 KV positions for batch item 0

    out = model(q, k, v, mask=mask)
    assert out.shape == q.shape


def test_invalid_num_heads_raises():
    """emb_dim must be divisible by num_heads; otherwise an AssertionError is expected."""
    with pytest.raises(AssertionError):
        MultiHeadBatched(emb_dim=128, num_heads=6)  # 128 % 6 != 0


def test_all_invalid_mask_produces_finite_output(model, cross_attn_inputs):
    """When every KV position is masked, nan_to_num(0.0) after softmax zeros the
    attention weights, so the output must be finite (no NaN / Inf)."""
    q, k, v = cross_attn_inputs
    B, S_kv = q.shape[0], k.shape[1]
    mask = torch.zeros(B, S_kv, dtype=torch.bool)  # every KV position is invalid

    out = model(q, k, v, mask=mask)
    assert torch.isfinite(out).all(), (
        "Expected finite output when every KV position is masked: "
        "nan_to_num(0.0) should replace NaN attention weights with zero."
    )


def test_output_is_finite(model, cross_attn_inputs):
    """All output values must be finite (no NaN or Inf) for a well-formed input."""
    q, k, v = cross_attn_inputs
    out = model(q, k, v)
    assert torch.isfinite(out).all()
