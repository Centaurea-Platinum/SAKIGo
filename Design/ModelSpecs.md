{
  "format": "JSON-compatible YAML 1.2",
  "schema_version": 1,
  "default_model": "model1",
  "notes": [
    "This file is intentionally pure data so Python tools can ingest it without a YAML dependency.",
    "max_board_size is only a validation/caching cap, not an architectural parameter; smaller square boards reuse the same weights.",
    "register_gather_blocks and register_broadcast_blocks are 1-based trunk block numbers.",
    "register_gather_blocks is 'all' or a 1-based block list; at least one gather block is required so global heads see the board.",
    "Rules are already encoded as one-hot groups plus normalized scalar values before entering the model.",
    "rule_dim is scoring_one_hot(4) + ko_one_hot(2) + suicide_one_hot(2) + komi + capture_diff = 10.",
    "Register seed conditioning adds an MLP-produced delta to the learned register seed and expands it uniformly across the D4 axis.",
    "Pass is the final logit in policy and budget outputs, so each has n*n+1 active logits.",
    "RoPE frequencies fill the {} slots below; each frequency rotates 4 head dims (row+col pairs), and any head_dim beyond 4 * total frequencies stays unrotated.",
    "Global frequency: theta = {} * index / (boardsize - 1)",
    "Local frequency: theta = {} * index",
    "Scalar control models keep the same forward API and register schedule but remove D4 weight sharing.",
    "A one-scalar-channel collapse per regular channel is an intentionally smaller control, not a trainable-parameter match: regular linear maps use 8 relative-group kernel weights per input/output regular-channel pair.",
    "model1_control_params therefore uses roughly sqrt(8) scalar width per regular channel to approximate model1 trainable parameters.",
    "model1_control_compute uses 8 scalar channels per regular channel to match model1 active scalar feature width and the dense compute shape of the current regular implementation."
  ],
  "models": {
    "model1": {
      "name": "Model 1",
      "architecture": "SakiGoModel",
      "activation": "SiLU",
      "max_board_size": 32,
      "input_planes": 6,
      "rule_dim": 10,
      "stem_channels": [6, 16, 32],
      "rule_mlp_channels": [10, 32, 64],
      "trunk": {
        "block_count": 5,
        "register_count": 2,
        "trunk_channels": 32,
        "bottleneck_channels": 16,
        "register_gather_blocks": "all",
        "register_broadcast_blocks": [5],
        "q_heads": 2,
        "kv_heads": 1,
        "head_dim": 8,
        "global_rope_frequencies": ["pi"],
        "local_rope_frequencies": ["pi/2"]
      },
      "heads": {
        "wdl": {
          "channels": [64, 8, 3],
          "collapse": "mean_d4_axis"
        },
        "score": {
          "channels": [64, 8, 1],
          "collapse": "mean_d4_axis"
        },
        "ownership": {
          "channels": [32, 8, 1],
          "collapse": "mean_d4_axis"
        },
        "policy": {
          "channels": [32, 8, 1],
          "pass_channels": [64, 8, 1],
          "collapse": "mean_d4_axis"
        },
        "budget": {
          "channels": [32, 8, 1],
          "pass_channels": [64, 8, 1],
          "collapse": "mean_d4_axis"
        }
      },
      "norm_eps": 0.000001
    },
    "model1_control_params": {
      "name": "Model 1 Scalar Control, Parameter Matched",
      "architecture": "ScalarSakiGoModel",
      "control_of": "model1",
      "control_match": "approximate_trainable_parameters",
      "activation": "SiLU",
      "max_board_size": 32,
      "input_planes": 6,
      "rule_dim": 10,
      "stem_channels": [6, 45, 91],
      "rule_mlp_channels": [10, 182],
      "trunk": {
        "block_count": 5,
        "register_count": 2,
        "trunk_channels": 91,
        "bottleneck_channels": 48,
        "register_gather_blocks": "all",
        "register_broadcast_blocks": [5],
        "q_heads": 6,
        "kv_heads": 3,
        "head_dim": 8,
        "global_rope_frequencies": ["pi"],
        "local_rope_frequencies": ["pi/2"]
      },
      "heads": {
        "wdl": {
          "channels": [182, 23, 3],
          "collapse": "none"
        },
        "score": {
          "channels": [182, 23, 1],
          "collapse": "none"
        },
        "ownership": {
          "channels": [91, 23, 1],
          "collapse": "none"
        },
        "policy": {
          "channels": [91, 23, 1],
          "pass_channels": [182, 23, 1],
          "collapse": "none"
        },
        "budget": {
          "channels": [91, 23, 1],
          "pass_channels": [182, 23, 1],
          "collapse": "none"
        }
      },
      "norm_eps": 0.000001
    },
    "model1_control_compute": {
      "name": "Model 1 Scalar Control, Compute Width Matched",
      "architecture": "ScalarSakiGoModel",
      "control_of": "model1",
      "control_match": "active_scalar_feature_width",
      "activation": "SiLU",
      "max_board_size": 32,
      "input_planes": 6,
      "rule_dim": 10,
      "stem_channels": [6, 128, 256],
      "rule_mlp_channels": [10, 256, 512],
      "trunk": {
        "block_count": 5,
        "register_count": 2,
        "trunk_channels": 256,
        "bottleneck_channels": 128,
        "register_gather_blocks": "all",
        "register_broadcast_blocks": [5],
        "q_heads": 16,
        "kv_heads": 8,
        "head_dim": 8,
        "global_rope_frequencies": ["pi"],
        "local_rope_frequencies": ["pi/2"]
      },
      "heads": {
        "wdl": {
          "channels": [512, 64, 3],
          "collapse": "none"
        },
        "score": {
          "channels": [512, 64, 1],
          "collapse": "none"
        },
        "ownership": {
          "channels": [256, 64, 1],
          "collapse": "none"
        },
        "policy": {
          "channels": [256, 64, 1],
          "pass_channels": [512, 64, 1],
          "collapse": "none"
        },
        "budget": {
          "channels": [256, 64, 1],
          "pass_channels": [512, 64, 1],
          "collapse": "none"
        }
      },
      "norm_eps": 0.000001
    }
  }
}
