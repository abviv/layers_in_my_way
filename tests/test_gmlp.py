import pytest
import torch
from modules.gmlp import SpatialGatingUnit, gMLPBlock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sgu():
    torch.manual_seed(0)
    return SpatialGatingUnit(d_model=256, seq_len=50)


@pytest.fixture
def block():
    torch.manual_seed(0)
    return gMLPBlock(d_model=128, d_ffn=256, seq_len=50)


@pytest.fixture
def sample_input():
    """Standard input tensor: [B, seq_len, d_model]."""
    torch.manual_seed(1)
    return torch.randn(4, 50, 128)


# ---------------------------------------------------------------------------
# SpatialGatingUnit tests
# ---------------------------------------------------------------------------

def test_sgu_output_shape(sgu):
    """SGU output dimension should be half the input channel dimension."""
    x = torch.randn(4, 50, 256)
    out = sgu(x)
    assert out.shape == (4, 50, 128)


def test_sgu_halves_channels(sgu):
    """Output channel dim must be exactly d_model // 2."""
    x = torch.randn(2, 50, 256)
    out = sgu(x)
    assert out.shape[-1] == 256 // 2


def test_sgu_init_weights_near_identity(sgu):
    """At init, spatial_proj weight is zero and bias is one, so the spatial
    projection acts close to identity (passes v through unchanged)."""
    assert torch.allclose(sgu.spatial_proj.weight, torch.zeros_like(sgu.spatial_proj.weight))
    assert torch.allclose(sgu.spatial_proj.bias, torch.ones_like(sgu.spatial_proj.bias))


def test_sgu_output_is_finite(sgu):
    """All output values must be finite for well-formed input."""
    x = torch.randn(4, 50, 256)
    out = sgu(x)
    assert torch.isfinite(out).all()


def test_sgu_batch_independence(sgu):
    """Each batch element should be processed independently."""
    x = torch.randn(4, 50, 256)
    out_full = sgu(x)
    out_single = sgu(x[0:1])
    assert torch.allclose(out_full[0:1], out_single, atol=1e-6)


# ---------------------------------------------------------------------------
# gMLPBlock tests
# ---------------------------------------------------------------------------

def test_block_output_shape(block, sample_input):
    """Output shape must match input shape due to residual connection."""
    out = block(sample_input)
    assert out.shape == sample_input.shape


def test_block_residual_connection(block, sample_input):
    """Output should differ from input (the block does something) but have
    the same shape (residual adds back to the shortcut)."""
    out = block(sample_input)
    assert out.shape == sample_input.shape
    assert not torch.allclose(out, sample_input)


def test_block_output_is_finite(block, sample_input):
    """All output values must be finite for well-formed input."""
    out = block(sample_input)
    assert torch.isfinite(out).all()


def test_block_batch_independence(block, sample_input):
    """Each batch element should be processed independently."""
    out_full = block(sample_input)
    out_single = block(sample_input[0:1])
    assert torch.allclose(out_full[0:1], out_single, atol=1e-6)


def test_block_deterministic(block, sample_input):
    """Two forward passes with the same input should yield identical output."""
    block.eval()
    out1 = block(sample_input)
    out2 = block(sample_input)
    assert torch.allclose(out1, out2)


def test_block_different_batch_sizes(block):
    """Block should handle various batch sizes correctly."""
    for B in [1, 2, 8, 16]:
        x = torch.randn(B, 50, 128)
        out = block(x)
        assert out.shape == (B, 50, 128)


def test_block_gradient_flow(block, sample_input):
    """Gradients must flow back through the block to the input."""
    sample_input.requires_grad_(True)
    out = block(sample_input)
    loss = out.sum()
    loss.backward()
    assert sample_input.grad is not None
    assert sample_input.grad.shape == sample_input.shape
    assert torch.isfinite(sample_input.grad).all()


def test_block_eval_vs_train(block, sample_input):
    """Block should produce the same output in train and eval mode
    (LayerNorm has no mode-dependent behaviour like dropout)."""
    block.train()
    out_train = block(sample_input)
    block.eval()
    out_eval = block(sample_input)
    assert torch.allclose(out_train, out_eval)


def test_odd_d_ffn_raises():
    """d_ffn must be even since SGU splits channels in half;
    an odd d_ffn should cause a shape mismatch during forward."""
    block = gMLPBlock(d_model=128, d_ffn=255, seq_len=50)
    x = torch.randn(2, 50, 128)
    with pytest.raises(Exception):
        block(x)
