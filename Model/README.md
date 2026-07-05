# SAKIGo Model

This folder contains the first PyTorch model implementation for SAKIGo. It is intentionally model-only: game rules, search, training losses, data generation, and action masking are separate phases.

The package lives in `Model/sakigo_model/`, and model sizes are selected from `Design/ModelSpecs/ModelSpecs.md`.

## Goals

- Use the same forward API for all model specs: `forward(board, rules)`.
- Keep the main model D4-equivariant across board rotations/reflections.
- Condition on rules through register initialization by default.
- Keep policy and pass as a single action distribution shape, `N*N + 1`.
- Support CUDA-friendly inference, including bf16 and CUDA graph replay for fixed shapes.

## Public API

```python
from sakigo_model import SakiGoModel, model_from_spec

model = model_from_spec("model1")
outputs = model(board, rules)
```

Inputs:

- `board`: tensor `[B, 6, N, N]`
- `rules`: tensor `[B, rule_dim]`, already encoded as one-hot/scalar rule features

Outputs:

- `wdl_logits`: `[B, 4]`
- `score`: `[B, 1]`
- `ownership_logits`: `[B, N*N]`
- `policy_logits`: `[B, N*N + 1]`, with pass at the final index
- `budget_logits`: `[B, N*N + 1]`, with pass at the final index

The active board size comes from the input tensor. The spec `max_board_size` is only a validation and cache cap.

## Main Model Rationale

`model1` uses D4 regular representations. Each feature channel carries 8 group components, one for each board symmetry. The model keeps these components through the stem, trunk, register interactions, and spatial heads, then collapses the D4 axis only at output time.

This is meant to make rotation/reflection behavior a property of the architecture instead of something the model must learn from augmentation. For spatial outputs, applying a D4 transform to the input should transform board logits the same way. For global outputs and pass logits, the result should be invariant.

The current register implementation is equivariant, not merely invariant. Registers have shape `[B, R, C, 8]`; under a board transform, the register group axis is permuted by the regular representation. Rule-conditioned register initialization starts group-uniform, then register gather/broadcast attention preserves equivariance.

## Forward Mechanism

The main `SakiGoModel` path is:

1. Validate `board` and `rules`.
2. Lift scalar board planes from `[B, 6, N, N]` to regular features `[B, 6, 8, N, N]`.
3. Apply a regular pointwise stem to reach trunk width.
4. Initialize registers from a learned seed plus `rule_mlp(rules)`.
5. Expand initialized registers uniformly over the D4 axis.
6. Run trunk blocks:
   - regular RMS norms
   - register gather from board to registers on configured blocks
   - two spatial GQA attention passes with canonical-frame RoPE
   - pointwise bottleneck projection back to trunk width
   - register broadcast back to board on configured blocks
7. Collapse the D4 axis with mean reduction in heads.
8. Produce global, spatial, policy, and budget outputs.

Rule conditioning uses register initialization only in v1. FiLM is reserved in the design/spec language for a future add-on, but there is no inactive FiLM branch in the current code.

## Heads

Global heads run on merged registers:

- WDL
- score
- policy pass logit
- budget pass logit

Spatial heads run on board features:

- ownership logits
- policy board logits
- budget board logits

Policy and budget concatenate board logits with their pass logit, producing one `[B, N*N + 1]` tensor each. Training can apply one softmax across that full action vector.

## Important Files

- `config.py`: dataclass configuration shared by model variants.
- `specs.py`: loader and factory for `Design/ModelSpecs/ModelSpecs.md`.
- `d4.py`: D4 group tables and tensor transforms.
- `layers.py`: regular-representation layers and trunk blocks.
- `model.py`: D4-equivariant `SakiGoModel`.
- `inference.py`: frozen inference helper with dtype/device movement and CUDA graph replay.
- `test_sakigo_model.py`: equivariance, spec, shape, and CUDA tests.

## Verification

Run from the repository root:

```powershell
$env:UV_CACHE_DIR='D:\stuff\Documents\SAKIGo\.uv-cache'
uv run --frozen pytest Model
```

The CUDA smoke tests require CUDA-visible PyTorch. The project is pinned to `torch==2.11.0+cu128`.
