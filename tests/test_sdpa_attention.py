import pytest
import torch
from attention import MultiHeadBatched, MultiHeadSDPA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, S, S_KV, D, H = 2, 10, 14, 128, 8


@pytest.fixture
def model():
    torch.manual_seed(0)
    return MultiHeadSDPA(emb_dim=D, num_heads=H)


@pytest.fixture
def self_attn_inputs():
    torch.manual_seed(1)
    q = torch.randn(B, S, D)
    k = torch.randn(B, S, D)
    v = torch.randn(B, S, D)
    return q, k, v


@pytest.fixture
def cross_attn_inputs():
    torch.manual_seed(1)
    q = torch.randn(B, S, D)
    k = torch.randn(B, S_KV, D)
    v = torch.randn(B, S_KV, D)
    return q, k, v


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_self_attention_output_shape(model, self_attn_inputs):
    q, k, v = self_attn_inputs
    out = model(q, k, v)
    assert out.shape == q.shape


def test_cross_attention_output_shape(model, cross_attn_inputs):
    q, k, v = cross_attn_inputs
    out = model(q, k, v)
    assert out.shape == q.shape


# ---------------------------------------------------------------------------
# Mask tests
# ---------------------------------------------------------------------------

def test_all_valid_mask_equals_no_mask(model, cross_attn_inputs):
    q, k, v = cross_attn_inputs
    mask = torch.ones(B, S_KV, dtype=torch.bool)

    out_masked = model(q, k, v, mask=mask)
    out_unmasked = model(q, k, v)
    assert torch.allclose(out_masked, out_unmasked, atol=1e-5)


def test_partial_mask_output_shape(model, cross_attn_inputs):
    q, k, v = cross_attn_inputs
    mask = torch.ones(B, S_KV, dtype=torch.bool)
    mask[0, 7:] = False

    out = model(q, k, v, mask=mask)
    assert out.shape == q.shape


def test_all_invalid_mask_produces_finite_output(model, cross_attn_inputs):
    q, k, v = cross_attn_inputs
    mask = torch.zeros(B, S_KV, dtype=torch.bool)

    out = model(q, k, v, mask=mask)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Finiteness
# ---------------------------------------------------------------------------

def test_output_is_finite(model, self_attn_inputs):
    q, k, v = self_attn_inputs
    out = model(q, k, v)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Inheritance & init
# ---------------------------------------------------------------------------

def test_inherits_from_multi_head_batched():
    model = MultiHeadSDPA(emb_dim=D, num_heads=H)
    assert isinstance(model, MultiHeadBatched)


def test_invalid_num_heads_raises():
    with pytest.raises(AssertionError):
        MultiHeadSDPA(emb_dim=D, num_heads=6)


# ---------------------------------------------------------------------------
# Parity with MultiHeadBatched
# ---------------------------------------------------------------------------

def test_matches_manual_attention(self_attn_inputs):
    """MultiHeadSDPA and MultiHeadBatched should produce the same output
    when sharing the same weights (no masking)."""
    torch.manual_seed(42)
    manual = MultiHeadBatched(emb_dim=D, num_heads=H)
    sdpa = MultiHeadSDPA(emb_dim=D, num_heads=H)
    sdpa.load_state_dict(manual.state_dict())

    q, k, v = self_attn_inputs
    manual.eval()
    sdpa.eval()

    out_manual = manual(q, k, v)
    out_sdpa = sdpa(q, k, v)
    assert torch.allclose(out_manual, out_sdpa, atol=1e-5)


def test_matches_manual_attention_cross(cross_attn_inputs):
    """Parity check for cross-attention."""
    torch.manual_seed(42)
    manual = MultiHeadBatched(emb_dim=D, num_heads=H)
    sdpa = MultiHeadSDPA(emb_dim=D, num_heads=H)
    sdpa.load_state_dict(manual.state_dict())

    q, k, v = cross_attn_inputs
    manual.eval()
    sdpa.eval()

    out_manual = manual(q, k, v)
    out_sdpa = sdpa(q, k, v)
    assert torch.allclose(out_manual, out_sdpa, atol=1e-5)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flow(model, self_attn_inputs):
    q, k, v = self_attn_inputs
    out = model(q, k, v)
    out.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
