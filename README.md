# Layers In My Way 

Personal reference repo — snippets and scripts I reach for when working on attention mechanisms and GPU kernels.

---

## Contents

### `attention.py` — MultiHead Attention

A clean `MultiHeadBatched` PyTorch module supporting both self-attention and cross-attention with optional masking.

- Handles asymmetric sequence lengths (S_q =/= S_kv) for cross-attention
- Applies `nan_to_num` after softmax to guard against fully-masked rows
- Standard scaled dot-product with learned Q/K/V/out projections

### `kimi_res_attn.py` — Attention Residuals

PyTorch reference implementation of the three residual connection schemes from the Kimi Team's [Attention Residuals](https://github.com/MoonshotAI/Attention-Residuals) paper (2026).

| Variant | Residual mechanism | Memory |
|---|---|---|
| Standard | `h = h + f(norm(h))` | O(d) |
| Full AttnRes | Softmax over all L prior layer outputs | O(L·d) |
| Block AttnRes | Softmax over N block summaries + partial | O(N·d) |

**Full Attention Residuals** — before each sub-layer, a learned pseudo-query `w_l ∈ ℝᵈ` attends over every prior layer output via softmax. Replaces the uniform residual sum with a dynamically weighted mix.

**Block Attention Residuals** — groups layers into blocks of size S. Within a block, outputs accumulate via plain addition into a `partial_block`. Across blocks, the same inter-layer attention runs before every sub-layer but attends only over finalized block summaries and the current partial — O(N) sources instead of O(L).

Run the shape trace demo:

```bash
python kimi_res_attn.py
```


## Tests
Repo is structured from the test first principle. Write tests first and then write the blocks. In this way the chances of bugs are less and also keep the llms away from hallucination territory.

```bash
pytest
```

Tests live in `tests/` and cover:

- `test_attention.py` — `MultiHeadBatched` shape, masking, self/cross-attention correctness
- `test_kimi_res_attn.py` — all three residual models: output shapes, finite values, gradient flow, block boundary transitions, attention weight normalization
