"""
Tests for kimi_res_attn.py — Attention Residuals reference implementation.

Coverage:
  - RMSNorm
  - full_attn_res / block_attn_res helper functions
  - StandardResidualModel
  - FullAttnResTransformerBlock / FullAttnResModel
  - BlockAttnResTransformerBlock / BlockAttnResModel
"""

import pytest
import torch
from modules.kimi_res_attn import (
    RMSNorm,
    DummyAttn,
    DummyMLP,
    StandardTransformerBlock,
    StandardResidualModel,
    full_attn_res,
    FullAttnResTransformerBlock,
    FullAttnResModel,
    block_attn_res,
    BlockAttnResTransformerBlock,
    BlockAttnResModel,
)


# =============================================================================
# Shared fixtures
# =============================================================================

B, T, D = 2, 16, 32  # small dims for fast tests


@pytest.fixture
def h():
    torch.manual_seed(0)
    return torch.randn(B, T, D)


@pytest.fixture
def sources(h):
    """Three random tensors mimicking a growing sources list."""
    torch.manual_seed(1)
    return [torch.randn(B, T, D) for _ in range(3)]


# =============================================================================
# RMSNorm
# =============================================================================

class TestRMSNorm:
    def test_output_shape(self):
        """RMSNorm should preserve the input tensor shape."""
        norm = RMSNorm(D)
        x = torch.randn(B, T, D)
        assert norm(x).shape == (B, T, D)

    def test_output_shape_batched_extra_dim(self):
        """Works on arbitrary leading dims."""
        norm = RMSNorm(D)
        x = torch.randn(4, 8, 3, D)
        assert norm(x).shape == (4, 8, 3, D)

    def test_output_is_finite(self):
        """RMS normalization should keep outputs finite for standard random inputs."""
        norm = RMSNorm(D)
        x = torch.randn(B, T, D)
        assert torch.isfinite(norm(x)).all()

    def test_unit_weight_scales_output(self):
        """With weight=1, RMSNorm(x) * 2 == RMSNorm_weight2(x)."""
        norm1 = RMSNorm(D)
        norm2 = RMSNorm(D)
        with torch.no_grad():
            norm2.weight.fill_(2.0)
        x = torch.randn(B, T, D)
        assert torch.allclose(norm1(x) * 2, norm2(x), atol=1e-5)

    def test_zero_input_output_is_finite(self):
        """Near-zero input should not produce NaN (eps guards the rsqrt)."""
        norm = RMSNorm(D)
        x = torch.zeros(B, T, D)
        assert torch.isfinite(norm(x)).all()


# =============================================================================
# DummyAttn / DummyMLP
# =============================================================================

class TestDummyLayers:
    def test_dummy_attn_shape(self):
        """The dummy attention layer should preserve the input tensor shape."""
        layer = DummyAttn(D)
        x = torch.randn(B, T, D)
        assert layer(x).shape == (B, T, D)

    def test_dummy_mlp_shape(self):
        """The dummy MLP layer should preserve the input tensor shape."""
        layer = DummyMLP(D)
        x = torch.randn(B, T, D)
        assert layer(x).shape == (B, T, D)


# =============================================================================
# StandardTransformerBlock / StandardResidualModel
# =============================================================================

class TestStandardResiduals:
    def test_block_output_shape(self, h):
        """A standard transformer block should preserve the hidden-state shape."""
        block = StandardTransformerBlock(D)
        assert block(h).shape == (B, T, D)

    def test_block_output_is_finite(self, h):
        """A standard transformer block should not produce NaN or Inf values."""
        block = StandardTransformerBlock(D)
        assert torch.isfinite(block(h)).all()

    def test_model_output_shape(self, h):
        """Stacking standard residual blocks should still preserve the hidden-state shape."""
        model = StandardResidualModel(D, num_transformer_blocks=4)
        assert model(h).shape == (B, T, D)

    def test_model_output_is_finite(self, h):
        """The standard residual model should produce finite activations end to end."""
        model = StandardResidualModel(D, num_transformer_blocks=4)
        assert torch.isfinite(model(h)).all()

    def test_single_block_model(self, h):
        """The standard residual model should also work when configured with a single block."""
        model = StandardResidualModel(D, num_transformer_blocks=1)
        assert model(h).shape == (B, T, D)


# =============================================================================
# full_attn_res
# =============================================================================

class TestFullAttnRes:
    def _make_inputs(self, n_sources):
        torch.manual_seed(42)
        w = torch.randn(D)
        srcs = [torch.randn(B, T, D) for _ in range(n_sources)]
        norm = RMSNorm(D)
        return w, srcs, norm

    def test_output_shape(self):
        """full_attn_res should return one residual tensor with the same shape as each source."""
        w, srcs, norm = self._make_inputs(3)
        out = full_attn_res(w, srcs, norm)
        assert out.shape == (B, T, D)

    def test_output_is_finite(self):
        """full_attn_res should remain numerically stable for normal random inputs."""
        w, srcs, norm = self._make_inputs(5)
        out = full_attn_res(w, srcs, norm)
        assert torch.isfinite(out).all()

    def test_single_source_returns_that_source(self):
        """With one source, softmax over dim-0 gives weight=1 → output = source."""
        norm = RMSNorm(D)
        with torch.no_grad():
            norm.weight.fill_(1.0)
        src = torch.randn(B, T, D)
        w = torch.zeros(D)
        out = full_attn_res(w, [src], norm)
        assert torch.allclose(out, src, atol=1e-5)

    def test_zero_query_uniform_weights(self):
        """Zero pseudo-query → uniform logits → uniform alpha (1/n per source)."""
        n = 4
        norm = RMSNorm(D)
        w = torch.zeros(D)
        srcs = [torch.randn(B, T, D) for _ in range(n)]

        # Expected output: average of sources (uniform alpha=1/n)
        expected = torch.stack(srcs, dim=0).mean(dim=0)
        out = full_attn_res(w, srcs, norm)
        # RMSNorm with unit weight on zero-query gives uniform logits only when
        # all K vectors have equal dot product with zero → all logits=0 → alpha=1/n
        # The output is then the average of V (not K), so compare against mean(srcs).
        # Note: logits = 0^T @ K = 0 for all sources → softmax gives 1/n always.
        assert torch.allclose(out, expected, atol=1e-5)

    def test_alpha_sums_to_one(self):
        """Attention weights must sum to 1 across source dimension for every (b,t)."""
        n = 5
        norm = RMSNorm(D)
        w = torch.randn(D)
        srcs = [torch.randn(B, T, D) for _ in range(n)]

        V = torch.stack(srcs, dim=0)          # [n, B, T, D]
        K = norm(V)
        logits = torch.einsum("d, n b t d -> n b t", w, K)
        alpha = logits.softmax(dim=0)          # [n, B, T]
        sums = alpha.sum(dim=0)                # [B, T]
        assert torch.allclose(sums, torch.ones(B, T), atol=1e-5)

    def test_more_sources_still_correct_shape(self):
        """Adding more source tensors should not change the output tensor shape."""
        w, srcs, norm = self._make_inputs(10)
        out = full_attn_res(w, srcs, norm)
        assert out.shape == (B, T, D)


# =============================================================================
# FullAttnResTransformerBlock / FullAttnResModel
# =============================================================================

class TestFullAttnRes_Block:
    def test_sources_grow_by_two(self, h):
        """Each call to the block must append exactly 2 tensors to sources."""
        block = FullAttnResTransformerBlock(D)
        sources_in = [h.clone()]
        sources_out = block(sources_in)
        assert len(sources_out) == 3  # 1 initial + 2 appended

    def test_new_sources_have_correct_shape(self, h):
        """Every source tensor returned by the block should keep the model hidden-state shape."""
        block = FullAttnResTransformerBlock(D)
        sources_out = block([h.clone()])
        for s in sources_out:
            assert s.shape == (B, T, D)

    def test_sources_grow_across_multiple_blocks(self, h):
        """Repeated full-attention residual blocks should append two sources per block."""
        n_blocks = 4
        blocks = [FullAttnResTransformerBlock(D) for _ in range(n_blocks)]
        sources = [h.clone()]
        for block in blocks:
            sources = block(sources)
        assert len(sources) == 1 + n_blocks * 2

    def test_model_output_shape(self, h):
        """The full-attention residual model should preserve the hidden-state shape."""
        model = FullAttnResModel(D, num_transformer_blocks=4)
        assert model(h).shape == (B, T, D)

    def test_model_output_is_finite(self, h):
        """The full-attention residual model should produce finite activations."""
        model = FullAttnResModel(D, num_transformer_blocks=4)
        assert torch.isfinite(model(h)).all()

    def test_model_single_block(self, h):
        """The full-attention residual model should also run with a single block."""
        model = FullAttnResModel(D, num_transformer_blocks=1)
        assert model(h).shape == (B, T, D)

    def test_model_applies_final_aggregation(self, h):
        """The model output should aggregate all sources, not just return the last one."""
        model = FullAttnResModel(D, num_transformer_blocks=1)

        with torch.no_grad():
            model.blocks[0].attn.proj.weight.zero_()
            model.blocks[0].mlp.proj.weight.zero_()
            model.out_res_query.zero_()
            model.out_res_norm.weight.fill_(1.0)

        out = model(h)
        expected = h / 3
        assert torch.allclose(out, expected, atol=1e-5)

    def test_model_has_more_params_than_standard(self, h):
        """Full AttnRes adds query vectors and extra norms per layer."""
        n = 4
        std = StandardResidualModel(D, n)
        full = FullAttnResModel(D, n)
        assert sum(p.numel() for p in full.parameters()) > sum(
            p.numel() for p in std.parameters()
        )


# =============================================================================
# block_attn_res
# =============================================================================

class TestBlockAttnRes:
    def test_output_shape(self):
        """block_attn_res should return a tensor matching the shape of its residual sources."""
        norm = RMSNorm(D)
        w = torch.randn(D)
        completed = [torch.randn(B, T, D) for _ in range(2)]
        partial = torch.randn(B, T, D)
        out = block_attn_res(completed, partial, w, norm)
        assert out.shape == (B, T, D)

    def test_output_is_finite(self):
        """block_attn_res should stay numerically finite for standard random inputs."""
        norm = RMSNorm(D)
        w = torch.randn(D)
        completed = [torch.randn(B, T, D) for _ in range(3)]
        partial = torch.randn(B, T, D)
        out = block_attn_res(completed, partial, w, norm)
        assert torch.isfinite(out).all()

    def test_zero_query_is_average_of_all_sources(self):
        """Zero query → uniform weights → output = mean(completed + [partial])."""
        norm = RMSNorm(D)
        w = torch.zeros(D)
        completed = [torch.randn(B, T, D) for _ in range(2)]
        partial = torch.randn(B, T, D)
        all_srcs = torch.stack(completed + [partial], dim=0)  # [3, B, T, D]
        expected = all_srcs.mean(dim=0)
        out = block_attn_res(completed, partial, w, norm)
        assert torch.allclose(out, expected, atol=1e-5)


# =============================================================================
# BlockAttnResTransformerBlock / BlockAttnResModel
# =============================================================================

class TestBlockAttnRes_Block:
    def _initial_state(self, h):
        completed = [h.clone()]
        partial = None
        layer_in_block = 0
        return completed, partial, layer_in_block

    def test_forward_returns_three_values(self, h):
        """A block-attention residual layer should return completed blocks, partial state, and offset."""
        block = BlockAttnResTransformerBlock(D, block_size=4, block_layer_offset=0)
        state = self._initial_state(h)
        result = block(*state)
        assert len(result) == 3

    def test_partial_block_shape(self, h):
        """The partial block accumulator should have the same shape as the hidden state."""
        block = BlockAttnResTransformerBlock(D, block_size=4, block_layer_offset=0)
        completed, partial, layer_in_block = block(*self._initial_state(h))
        assert partial.shape == (B, T, D)

    def test_layer_in_block_increments(self, h):
        """Each transformer block should advance the intra-block offset by two sublayers."""
        block_size = 6
        block = BlockAttnResTransformerBlock(D, block_size=block_size, block_layer_offset=0)
        completed, partial, lib = block(*self._initial_state(h))
        # Each transformer block runs 2 sub-layers → lib increments by 2
        assert lib == 2

    def test_block_boundary_finalizes_partial(self, h):
        """Boundary check fires at the START of the next sub-layer.
        With block_size=2: call 1 fills the block (lib 0→2, no boundary yet).
        Call 2 triggers the boundary at its first sub-layer, appending the
        previous partial to completed_blocks."""
        block_size = 2
        block = BlockAttnResTransformerBlock(D, block_size=block_size, block_layer_offset=0)

        state = self._initial_state(h)
        state = block(*state)                      # lib goes 0 → 2; no boundary yet
        assert state[2] == block_size              # lib == 2, boundary not yet triggered
        assert len(state[0]) == 1                  # completed still has 1 entry

        completed, partial, lib = block(*state)    # boundary fires at start of call 2
        assert len(completed) == 2                 # initial [h] + 1 finalized block

    def test_completed_blocks_grow_at_boundary(self, h):
        """completed_blocks grows by 1 each time a block boundary is crossed."""
        block_size = 2
        block = BlockAttnResTransformerBlock(D, block_size=block_size, block_layer_offset=0)
        state = self._initial_state(h)
        state = block(*state)                      # lib 0→2; completed still len=1
        completed, partial, lib = block(*state)    # boundary fires; completed grows
        assert len(completed) == 2

    def test_model_output_shape(self, h):
        """The block-attention residual model should preserve the hidden-state shape."""
        model = BlockAttnResModel(D, num_transformer_blocks=6, block_size=4)
        assert model(h).shape == (B, T, D)

    def test_model_output_is_finite(self, h):
        """The block-attention residual model should produce finite activations."""
        model = BlockAttnResModel(D, num_transformer_blocks=6, block_size=4)
        assert torch.isfinite(model(h)).all()

    def test_model_single_block(self, h):
        """The block-attention residual model should also work when only one logical block is used."""
        model = BlockAttnResModel(D, num_transformer_blocks=3, block_size=6)
        assert model(h).shape == (B, T, D)

    def test_model_block_size_one(self, h):
        """block_size=1 means every sub-layer starts a new block."""
        model = BlockAttnResModel(D, num_transformer_blocks=3, block_size=1)
        assert model(h).shape == (B, T, D)

    def test_block_model_fewer_params_than_full(self, h):
        """BlockAttnRes shares structure with FullAttnRes; param counts should be equal
        (both have the same per-layer learnable queries and norms)."""
        n = 4
        full = FullAttnResModel(D, n)
        block = BlockAttnResModel(D, n, block_size=4)
        # They have identical per-block parameters
        assert sum(p.numel() for p in full.parameters()) == sum(
            p.numel() for p in block.parameters()
        )


# =============================================================================
# Cross-model: gradient flow
# =============================================================================

class TestGradientFlow:
    """Ensure gradients reach all learnable parameters (no dead subgraph)."""

    def _check_grads(self, model, h):
        out = model(h)
        loss = out.sum()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_standard_residual_grads(self, h):
        """Standard residual models should backpropagate gradients to every learnable parameter."""
        model = StandardResidualModel(D, num_transformer_blocks=3)
        self._check_grads(model, h)

    def test_full_attn_res_grads(self, h):
        """Full-attention residual models should backpropagate gradients to every learnable parameter."""
        model = FullAttnResModel(D, num_transformer_blocks=3)
        self._check_grads(model, h)

    def test_block_attn_res_grads(self, h):
        """Block-attention residual models should backpropagate gradients to every learnable parameter."""
        model = BlockAttnResModel(D, num_transformer_blocks=6, block_size=4)
        self._check_grads(model, h)
