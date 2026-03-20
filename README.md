# Layers In My Way

Personal reference repo for PyTorch building blocks around attention, transformer variants, residual-routing ideas, and a small gMLP implementation.

## Currently Supported

### `modules/attention.py`

Multi-head attention with two interchangeable implementations:

- `MultiHeadBatched`: manual scaled dot-product attention
- `MultiHeadSDPA`: `torch.nn.functional.scaled_dot_product_attention` backend
- `safe_mask(mask)`: public helper for converting a `[B, S_kv]` validity mask into a backward-safe mask

Supported behavior:

- self-attention and true cross-attention
- different query and key/value sequence lengths
- boolean or integer validity masks with shape `[B, S_kv]`
- fully masked key/value rows without NaNs in backward
- fully masked rows contributing zero attention signal, so the final output falls back to the output projection bias
- parity tests between the manual and SDPA implementations

### `modules/transformer.py`

`TransformerBlock` is a pre-norm transformer block built from:

- `MultiHeadSDPA` attention
- residual connections
- dropout
- `MlpS` feed-forward block

Supported behavior:

- standard self-attention mode
- cross-attention mode via `block.is_crossattention = True`
- different `seq_len_q` and `seq_len_kv` in cross-attention
- `[B, S_kv]` masks in both self-attention and cross-attention
- backward-safe handling of fully masked key/value rows through the shared `safe_mask` logic

### `modules/gmlp.py`

Small gMLP reference implementation with:

- `SpatialGatingUnit`
- `gMLPBlock`

Supported behavior:

- residual gMLP block over `[B, seq_len, d_model]`
- learnable spatial projection over the sequence dimension
- deterministic near-identity SGU initialization

### `modules/kimi_res_attn.py`

PyTorch reference implementation of residual routing variants inspired by the Kimi Team's [Attention Residuals](https://github.com/MoonshotAI/Attention-Residuals) paper.

Implemented variants:

| Variant | Residual mechanism | Memory |
|---|---|---|
| Standard | `h = h + f(norm(h))` | O(d) |
| Full AttnRes | Softmax over all prior layer outputs | O(L·d) |
| Block AttnRes | Softmax over completed block summaries + current partial block | O(N·d) |

Included components:

- `RMSNorm`
- `StandardTransformerBlock` and `StandardResidualModel`
- `FullAttnResTransformerBlock` and `FullAttnResModel`
- `BlockAttnResTransformerBlock` and `BlockAttnResModel`

Run the shape/demo script:

```bash
python modules/kimi_res_attn.py
```

## Setup

```bash
conda env create -f environment.yml
conda activate torch_gpu
```

## Tests

The repo is organized test-first: each module has direct coverage for shape, masking, finiteness, and gradient behavior.

```bash
pytest
```

Current test coverage includes:

- `tests/test_attention.py` for manual attention, `safe_mask`, masking semantics, and backward stability
- `tests/test_sdpa_attention.py` for SDPA attention parity and masked-row safety
- `tests/test_transformer.py` for pre-norm transformer blocks, self/cross-attention, masks, and gradient flow
- `tests/test_gmlp.py` for `SpatialGatingUnit` and `gMLPBlock`
- `tests/test_kimi_res_attn.py` for all Attention Residuals variants, normalization, shape checks, and gradient flow
