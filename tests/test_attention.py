import warnings

import pytest
import torch
from torch.testing import assert_close

from modules.attention import MultiHeadBatched, safe_mask


B, S_SELF, S_Q, S_KV, D, H = 2, 10, 5, 10, 128, 8


@pytest.fixture
def model():
    torch.manual_seed(0)
    return MultiHeadBatched(emb_dim=D, num_heads=H)


@pytest.fixture
def identity_model():
    model = MultiHeadBatched(emb_dim=4, num_heads=2)
    with torch.no_grad():
        eye = torch.eye(4)
        for layer in (model.q_proj, model.k_proj, model.v_proj, model.out_proj):
            layer.weight.copy_(eye)
            layer.bias.zero_()
    return model


@pytest.fixture
def self_attn_inputs():
    torch.manual_seed(1)
    q = torch.randn(B, S_SELF, D)
    k = torch.randn(B, S_SELF, D)
    v = torch.randn(B, S_SELF, D)
    return q, k, v


@pytest.fixture
def cross_attn_inputs():
    torch.manual_seed(1)
    q = torch.randn(B, S_Q, D)
    k = torch.randn(B, S_KV, D)
    v = torch.randn(B, S_KV, D)
    return q, k, v


def test_self_attention_output_shape(model, self_attn_inputs):
    """Self-attention should preserve the query tensor shape."""
    q, k, v = self_attn_inputs
    out = model(q, k, v)
    assert out.shape == q.shape


def test_cross_attention_output_shape(model, cross_attn_inputs):
    """Cross-attention should still return one output vector per query token."""
    q, k, v = cross_attn_inputs
    out = model(q, k, v)
    assert out.shape == q.shape


def test_safe_mask_reopens_fully_invalid_rows():
    """Fully invalid rows should be reopened so attention never sees an all-masked row."""
    mask = torch.tensor([[0, 0, 0], [1, 0, 1]], dtype=torch.bool)

    adjusted = safe_mask(mask)
    expected = torch.tensor([[1, 1, 1], [1, 0, 1]], dtype=torch.bool)

    assert_close(adjusted, expected)


def test_safe_mask_leaves_partially_valid_rows_unchanged():
    """Rows that already contain a valid key should not be modified."""
    mask = torch.tensor([[1, 1, 0, 1], [0, 1, 0, 0]], dtype=torch.bool)

    adjusted = safe_mask(mask)

    assert_close(adjusted, mask)


def test_safe_mask_treats_integer_and_boolean_masks_equally():
    """safe_mask should normalize integer and boolean masks to the same boolean result."""
    bool_mask = torch.tensor([[0, 0, 0], [1, 0, 1]], dtype=torch.bool)
    int_mask = bool_mask.to(torch.int64)

    adjusted_bool = safe_mask(bool_mask)
    adjusted_int = safe_mask(int_mask)

    assert adjusted_bool.dtype is torch.bool
    assert adjusted_int.dtype is torch.bool
    assert_close(adjusted_bool, adjusted_int)


def test_all_valid_mask_equals_no_mask(model, cross_attn_inputs):
    """A fully valid mask should be equivalent to omitting the mask entirely."""
    q, k, v = cross_attn_inputs
    mask = torch.ones(B, S_KV, dtype=torch.bool)

    out_masked = model(q, k, v, mask=mask)
    out_unmasked = model(q, k, v)

    assert_close(out_masked, out_unmasked)


def test_partial_mask_output_shape(model, cross_attn_inputs):
    """Masking some key/value positions must not change the output tensor shape."""
    q, k, v = cross_attn_inputs
    mask = torch.ones(B, S_KV, dtype=torch.bool)
    mask[0, 5:] = False

    out = model(q, k, v, mask=mask)
    assert out.shape == q.shape


def test_invalid_num_heads_raises():
    """The module should reject head counts that do not evenly divide the embedding size."""
    with pytest.raises(AssertionError):
        MultiHeadBatched(emb_dim=D, num_heads=6)


def test_all_invalid_mask_returns_output_bias(model, cross_attn_inputs):
    """When every key is masked, attention contributes zeros and only the output bias remains."""
    q, k, v = cross_attn_inputs
    mask = torch.zeros(B, S_KV, dtype=torch.bool)

    out = model(q, k, v, mask=mask)
    expected = model.out_proj.bias.view(1, 1, -1).expand_as(out)

    assert_close(out, expected)


def test_output_is_finite(model, cross_attn_inputs):
    """A normal forward pass should not produce NaN or Inf values."""
    q, k, v = cross_attn_inputs
    out = model(q, k, v)
    assert torch.isfinite(out).all()


def test_masked_prefix_matches_trimmed_kv(model):
    """Masking a suffix of keys/values should match running attention on the unmasked prefix only."""
    torch.manual_seed(2)
    q = torch.randn(1, 4, D)
    k = torch.randn(1, 6, D)
    v = torch.randn(1, 6, D)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.bool)

    out_masked = model(q, k, v, mask=mask)
    out_trimmed = model(q, k[:, :3], v[:, :3])

    assert_close(out_masked, out_trimmed)


def test_single_valid_token_is_broadcast_to_every_query(identity_model):
    """If exactly one value token is valid, every query should receive that same value vector."""
    q = torch.tensor(
        [[[1.0, 0.0, 2.0, -1.0], [0.5, 1.5, -0.5, 2.0], [-1.0, 2.0, 0.0, 1.0]]]
    )
    k = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [-4.0, -3.0, -2.0, -1.0]]])
    v = torch.tensor([[[10.0, 20.0, 30.0, 40.0], [5.0, 6.0, 7.0, 8.0]]])
    mask = torch.tensor([[0, 1]], dtype=torch.bool)

    out = identity_model(q, k, v, mask=mask)
    expected = v[:, 1:2, :].expand(-1, q.shape[1], -1)

    assert_close(out, expected)


def test_mask_only_changes_masked_batch_item(model, cross_attn_inputs):
    """A per-batch mask should only affect the batch element whose mask actually changed."""
    q, k, v = cross_attn_inputs
    mask = torch.ones(B, S_KV, dtype=torch.bool)
    mask[0, 7:] = False

    out_masked = model(q, k, v, mask=mask)
    out_unmasked = model(q, k, v)

    assert_close(out_masked[1], out_unmasked[1])


def test_integer_and_boolean_masks_are_equivalent(model, cross_attn_inputs):
    """The masking logic should treat integer and boolean validity masks the same way."""
    q, k, v = cross_attn_inputs
    bool_mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]],
        dtype=torch.bool,
    )
    int_mask = bool_mask.to(torch.int64)

    out_bool = model(q, k, v, mask=bool_mask)
    out_int = model(q, k, v, mask=int_mask)

    assert_close(out_bool, out_int)


def test_masked_value_positions_have_zero_gradient(model):
    """Masked value positions should not receive gradient because they never contribute to the output."""
    torch.manual_seed(3)
    q = torch.randn(1, 4, D, requires_grad=True)
    k = torch.randn(1, 6, D, requires_grad=True)
    v = torch.randn(1, 6, D, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.bool)

    out = model(q, k, v, mask=mask)
    out.sum().backward()

    assert v.grad is not None
    assert v.grad[:, :3].abs().sum() > 0
    assert torch.count_nonzero(v.grad[:, 3:]) == 0


def test_fully_masked_batch_item_keeps_backward_finite(model):
    """Backward should stay finite even if one batch item has every key/value position masked out."""
    torch.manual_seed(4)
    q = torch.randn(2, 4, D, requires_grad=True)
    k = torch.randn(2, 6, D, requires_grad=True)
    v = torch.randn(2, 6, D, requires_grad=True)
    mask = torch.tensor(
        [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]],
        dtype=torch.bool,
    )

    # A fully masked batch row is reopened before softmax and then zeroed after
    # attention so anomaly detection stays quiet during backward.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Anomaly Detection has been enabled.*",
            category=UserWarning,
        )
        with torch.autograd.detect_anomaly(check_nan=True):
            out = model(q, k, v, mask=mask)
            loss = out.square().mean()
            loss.backward()

    for tensor in (q, k, v):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()

    for _, parameter in model.named_parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_backward_populates_input_and_parameter_gradients(model, self_attn_inputs):
    """Backward pass should populate finite gradients for both inputs and learned projections."""
    q, k, v = (tensor.clone().requires_grad_() for tensor in self_attn_inputs)

    out = model(q, k, v)
    loss = out.square().mean()
    loss.backward()

    for tensor in (q, k, v):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()

    for _, parameter in model.named_parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
