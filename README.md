# benchmarking_layers

Personal reference repo — snippets and scripts I reach for most often when working on attention and GPU kernels.

## Files

**`attention.py`** — Clean `MultiHeadBatched` PyTorch module supporting both self-attention and cross-attention with optional key/value masking.

**`attn_marking.py`** — Benchmarking scaffold comparing naive scaled dot-product attention, `torch.compile`, and `F.scaled_dot_product_attention` (FlashAttention). Supports single-run timing, `torch.profiler` trace export, and a sequence-length sweep.

## Tests

Tests live in `tests/` and are run with pytest:

```bash
pytest
```
