import pytest
import torch
from modules.attention import MultiHeadBatched, MultiHeadSDPA


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
    """SDPA self-attention should preserve the query tensor shape."""
    q, k, v = self_attn_inputs
    out = model(q, k, v)
    assert out.shape == q.shape


def test_cross_attention_output_shape(model, cross_attn_inputs):
    """SDPA cross-attention should still emit one output vector per query token."""
    q, k, v = cross_attn_inputs
    out = model(q, k, v)
    assert out.shape == q.shape


# ---------------------------------------------------------------------------
# Mask tests
# ---------------------------------------------------------------------------

def test_all_valid_mask_equals_no_mask(model, cross_attn_inputs):
    """A fully valid mask should be equivalent to leaving the SDPA mask unset."""
    q, k, v = cross_attn_inputs
    mask = torch.ones(B, S_KV, dtype=torch.bool)

    out_masked = model(q, k, v, mask=mask)
    out_unmasked = model(q, k, v)
    assert torch.allclose(out_masked, out_unmasked, atol=1e-5)


def test_partial_mask_output_shape(model, cross_attn_inputs):
    """Masking some keys/values must not change the shape of the SDPA output."""
    q, k, v = cross_attn_inputs
    mask = torch.ones(B, S_KV, dtype=torch.bool)
    mask[0, 7:] = False

    out = model(q, k, v, mask=mask)
    assert out.shape == q.shape


def test_all_invalid_mask_produces_finite_output(model, cross_attn_inputs):
    """Fully masked SDPA rows should be sanitized so the final output stays finite."""
    q, k, v = cross_attn_inputs
    mask = torch.zeros(B, S_KV, dtype=torch.bool)

    out = model(q, k, v, mask=mask)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Finiteness
# ---------------------------------------------------------------------------

def test_output_is_finite(model, self_attn_inputs):
    """A normal SDPA forward pass should not produce NaN or Inf values."""
    q, k, v = self_attn_inputs
    out = model(q, k, v)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Inheritance & init
# ---------------------------------------------------------------------------

def test_inherits_from_multi_head_batched():
    """The SDPA module should preserve the manual attention module interface via inheritance."""
    model = MultiHeadSDPA(emb_dim=D, num_heads=H)
    assert isinstance(model, MultiHeadBatched)


def test_invalid_num_heads_raises():
    """SDPA attention should reject head counts that do not evenly divide the embedding size."""
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


def test_matches_manual_attention_with_mask(self_attn_inputs):
    """Manual attention and SDPA should match for self-attention with a partial validity mask."""
    torch.manual_seed(42)
    manual = MultiHeadBatched(emb_dim=D, num_heads=H)
    sdpa = MultiHeadSDPA(emb_dim=D, num_heads=H)
    sdpa.load_state_dict(manual.state_dict())

    q, k, v = self_attn_inputs
    mask = torch.ones(B, S, dtype=torch.bool)
    mask[0, 6:] = False
    mask[1, 2] = False

    manual.eval()
    sdpa.eval()

    out_manual = manual(q, k, v, mask=mask)
    out_sdpa = sdpa(q, k, v, mask=mask)
    assert torch.allclose(out_manual, out_sdpa, atol=1e-5)


def test_matches_manual_attention_cross_with_mask(cross_attn_inputs):
    """Manual attention and SDPA should match for cross-attention with a partial validity mask."""
    torch.manual_seed(42)
    manual = MultiHeadBatched(emb_dim=D, num_heads=H)
    sdpa = MultiHeadSDPA(emb_dim=D, num_heads=H)
    sdpa.load_state_dict(manual.state_dict())

    q, k, v = cross_attn_inputs
    mask = torch.ones(B, S_KV, dtype=torch.bool)
    mask[0, 7:] = False
    mask[1, 1::2] = False

    manual.eval()
    sdpa.eval()

    out_manual = manual(q, k, v, mask=mask)
    out_sdpa = sdpa(q, k, v, mask=mask)
    assert torch.allclose(out_manual, out_sdpa, atol=1e-5)


def test_matches_manual_attention_when_everything_is_masked(cross_attn_inputs):
    """Manual attention and SDPA should agree even when every key/value position is masked out."""
    torch.manual_seed(42)
    manual = MultiHeadBatched(emb_dim=D, num_heads=H)
    sdpa = MultiHeadSDPA(emb_dim=D, num_heads=H)
    sdpa.load_state_dict(manual.state_dict())

    q, k, v = cross_attn_inputs
    mask = torch.zeros(B, S_KV, dtype=torch.bool)

    manual.eval()
    sdpa.eval()

    out_manual = manual(q, k, v, mask=mask)
    out_sdpa = sdpa(q, k, v, mask=mask)
    assert torch.allclose(out_manual, out_sdpa, atol=1e-5)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flow(model, self_attn_inputs):
    """Backward pass should populate gradients for all SDPA attention parameters."""
    q, k, v = self_attn_inputs
    out = model(q, k, v)
    out.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
