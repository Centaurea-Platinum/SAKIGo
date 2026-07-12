# Non-Board Input

The model receives ten non-board rule features. Mutually exclusive rules use
one-hot groups; komi and capture difference use normalized scalars. This avoids
turning global rules into board-sized planes or sampling a meaningless full
hypercube of rule combinations.

The ten features are:

| Group | Encoding |
|---|---|
| Scoring | one-hot: Area, Area + Ancient Chinese group tax, Territory, Territory with seki scoring |
| Ko | one-hot: Simple Ko, Positional Superko |
| Suicide | one-hot: allowed, forbidden |
| Komi | scalar normalized by board area |
| Captured stones | scalar normalized by board area |

Captured stones is measured from the side-to-move perspective:

```text
opponent stones I captured - my stones the opponent captured
```

Handicap-related rule effects are represented through komi. The two scalars use
the model contract's normalized `[-1, 1]` range.

## Register Initialization

The feature vector directly initializes the register tokens:

```python
registers = mlp_10_32_128(rule_features)
registers = reshape(registers, [B, 2, 64, 1])
registers = expand(registers, group=8)
```

The final MLP bias provides the rule-independent component, so there is no
separate learned register seed. The registers broadcast into the board once at
the beginning of the trunk. After all board blocks, one gather updates the
registers from the final board representation for global and pass heads.

There is no FiLM path or repeated rule injection inside the board blocks.
