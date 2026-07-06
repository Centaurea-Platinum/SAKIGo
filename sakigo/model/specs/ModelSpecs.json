{
  "format": "JSON-compatible YAML 1.2",
  "schema_version": 3,
  "default_model": "model1",
  "includes": {
    "stem_shapes": "StemShapes.md",
    "head_shapes": "HeadShapes.md"
  },
  "notes": [
    "These files are intentionally pure data so Python tools can ingest them without a YAML dependency.",
    "max_board_size is only a validation/caching cap, not an architectural parameter; smaller square boards reuse the same weights.",
    "architecture is a descriptive tag; specs.py maps it to the implementation class.",
    "stem_shape and head_shape name reusable shapes from sibling files in this directory.",
    "expanded_channel is m from ModelArchitecture.md: the persistent per-token trunk feature channel.",
    "register_channel is the per-register feature width; it may be narrower than expanded_channel.",
    "bottleneck_channel is n from ModelArchitecture.md: the inner trunk-block attention/bottleneck width.",
    "register_bottleneck_channel is the register gather/broadcast attention width; it may be narrower than bottleneck_channel.",
    "Merged register/global-head input width is derived as register_count * register_channel.",
    "The per-head q/kv channel is derived as bottleneck_channel / q_heads; register cross-attention uses register_bottleneck_channel / q_heads.",
    "register_gather_blocks and register_broadcast_blocks are 1-based trunk block numbers.",
    "register_gather_blocks is 'all' or a 1-based block list; at least one gather block is required so global heads see the board.",
    "Rules are already encoded as one-hot groups plus normalized scalar values before entering the model.",
    "Register seed conditioning adds an MLP-produced delta to the learned register seed and expands it uniformly across the D4 axis.",
    "Pass is the final logit in policy and budget outputs, so each has board_area + 1 active logits.",
    "RoPE frequencies fill the {} slots below; each frequency rotates 4 derived per-head dimensions (row+col pairs), and any remaining per-head dimensions stay unrotated.",
    "Global frequency: theta = {} * index / (boardsize - 1)",
    "Local frequency: theta = {} * index"
  ],
  "models": {
    "model1": {
      "name": "Model 1",
      "architecture": "d4_equivariant",
      "activation": "SiLU",
      "max_board_size": 32,
      "stem_shape": "regular_v1",
      "head_shape": "standard_v1",
      "trunk": {
        "block_count": 8,
        "register_count": 2,
        "expanded_channel": 32,
        "register_channel": 32,
        "bottleneck_channel": 16,
        "register_bottleneck_channel": 16,
        "register_gather_blocks": [1, 8],
        "register_broadcast_blocks": [1, 8],
        "q_heads": 2,
        "kv_heads": 1,
        "global_rope_frequencies": ["pi"],
        "local_rope_frequencies": ["pi/2"]
      },
      "norm_eps": 0.000001
    },
    "model2": {
      "name": "Model 2",
      "architecture": "d4_equivariant",
      "activation": "SiLU",
      "max_board_size": 32,
      "stem_shape": "regular_v1",
      "head_shape": "standard_v1",
      "trunk": {
        "block_count": 16,
        "register_count": 2,
        "expanded_channel": 128,
        "register_channel": 64,
        "bottleneck_channel": 64,
        "register_bottleneck_channel": 32,
        "register_gather_blocks": [1, 6, 13, 16],
        "register_broadcast_blocks": [1, 6, 13, 16],
        "q_heads": 2,
        "kv_heads": 1,
        "global_rope_frequencies": ["pi"],
        "local_rope_frequencies": ["pi/2"]
      },
      "norm_eps": 0.000001
    },
    "model3": {
      "name": "Model 3",
      "architecture": "d4_equivariant",
      "activation": "SiLU",
      "max_board_size": 32,
      "stem_shape": "regular_v1",
      "head_shape": "standard_v1",
      "trunk": {
        "block_count": 16,
        "register_count": 2,
        "expanded_channel": 128,
        "register_channel": 64,
        "bottleneck_channel": 64,
        "register_bottleneck_channel": 32,
        "register_gather_blocks": [1, 6, 13, 16],
        "register_broadcast_blocks": [1, 6, 13, 16],
        "q_heads": 2,
        "kv_heads": 1,
        "global_rope_frequencies": ["pi", "pi/2"],
        "local_rope_frequencies": ["pi/2", "pi/4"]
      },
      "norm_eps": 0.000001
    }
  }
}
