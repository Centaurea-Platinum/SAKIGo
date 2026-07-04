{
  "format": "JSON-compatible YAML 1.2",
  "schema_version": 1,
  "notes": [
    "Channel entries can be integers or model-context names resolved by specs.py.",
    "The first stem channel is the board-input plane count; the first rule MLP channel is the encoded rule-vector width.",
    "expanded_channel is m from ModelArchitecture.md; bottleneck_channel is n."
  ],
  "stem_shapes": {
    "regular_v1": {
      "stem_channels": [6, 16, "expanded_channel"],
      "rule_mlp_channels": [10, 32, "expanded_channel * register_count", "expanded_channel * register_count"]
    }
  }
}
