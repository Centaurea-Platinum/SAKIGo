> **Current scope: distillation only.** The history-aware engine remains relevant
> for producing the legality input plane; search-related behavior is not currently
> considered.

The neural-network input is not bijective with game state because long-repeat and
superko rules depend on history. For current distillation, the history-aware engine
retains that state and projects its effect into the NonTrivialIllegal input plane.
The full history is intentionally not a neural-network input.
