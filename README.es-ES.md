

# Capas A Mi Manera

Repositorio de referencia personal con bloques de construcción de PyTorch centrados en atención, variantes de transformadores, ideas de enrutamiento residual y una pequeña implementación de gMLP.

## Soporte Actual

### `modules/attention.py`

Atención multiencabezado con dos implementaciones intercambiables:

- `MultiHeadBatched`: atención producto punto escalado manual
- `MultiHeadSDPA`: backend `torch.nn.functional.scaled_dot_product_attention`
- `safe_mask(mask)`: función auxiliar pública para convertir una máscara de validez `[B, S_kv]` en una máscara segura para la propagación inversa

Comportamiento soportado:

- autoatención y atención cruzada real
- diferentes longitudes de secuencia para consulta y clave/valor
- máscaras de validez booleanas o enteras con forma `[B, S_kv]`
- filas de clave/valor totalmente enmascaradas sin NaNs en la propagación inversa
- filas totalmente enmascaradas que contribuyen con una señal de atención cero, por lo que la salida final recurre al sesgo (bias) de la proyección de salida
- pruebas de paridad entre las implementaciones manual y SDPA

### `modules/transformer.py`

El `TransformerBlock` es un bloque transformador pre-norma construido a partir de:

- atención `MultiHeadSDPA`
- conexiones residuales
- dropout
- bloque feed-forward `MlpS`

Comportamiento soportado:

- modo estándar de autoatención
- modo de atención cruzada mediante `block.is_crossattention = True`
- `seq_len_q` y `seq_len_kv` diferentes en atención cruzada
- máscaras `[B, S_kv]` tanto en autoatención como en atención cruzada
- manejo seguro para la propagación inversa de filas de clave/valor totalmente enmascaradas mediante la lógica compartida de `safe_mask`

### `modules/gmlp.py`

Pequeña implementación de referencia de gMLP con:

- `SpatialGatingUnit`
- `gMLPBlock`

Comportamiento soportado:

- bloque residual de gMLP sobre `[B, seq_len, d_model]`
- proyección espacial aprendible sobre la dimensión de secuencia
- inicialización determinista de SGU cercana a la identidad

### `modules/kimi_res_attn.py`

Implementación de referencia en PyTorch de variantes de enrutamiento residual inspiradas en el paper [Attention Residuals](https://github.com/MoonshotAI/Attention-Residuals) del equipo Kimi.

Variantes implementadas:

| Variante | Mecanismo residual | Memoria |
|---|---|---|
| Estándar | `h = h + f(norm(h))` | O(d) |
| AttnRes Completo | Softmax sobre todas las salidas de capas anteriores | O(L·d) |
| AttnRes por Bloque | Softmax sobre resúmenes de bloques completados + bloque parcial actual | O(N·d) |

Componentes incluidos:

- `RMSNorm`
- `StandardTransformerBlock` y `StandardResidualModel`
- `FullAttnResTransformerBlock` y `FullAttnResModel`
- `BlockAttnResTransformerBlock` y `BlockAttnResModel`

Ejecuta el script de forma/demostración:

```bash
python modules/kimi_res_attn.py
```

## Configuración

```bash
conda env create -f environment.yml
conda activate torch_gpu
```

## Pruebas

El repositorio está organizado con enfoque test-first: cada módulo tiene cobertura directa para forma, enmascaramiento, finitud y comportamiento de gradientes.

```bash
pytest
```

La cobertura de pruebas actual incluye:

- `tests/test_attention.py` para atención manual, `safe_mask`, semánticas de enmascaramiento y estabilidad de propagación inversa
- `tests/test_sdpa_attention.py` para paridad de atención SDPA y seguridad de filas enmascaradas
- `tests/test_transformer.py` para bloques transformadores pre-norma, auto/atención cruzada, máscaras y flujo de gradientes
- `tests/test_gmlp.py` para `SpatialGatingUnit` y `gMLPBlock`
- `tests/test_kimi_res_attn.py` para todas las variantes de Attention Residuals, normalización, verificaciones de forma y flujo de gradientes
