> **Status: Not currently considered.** Search-based training and self-play
> refinements are outside the current KataGo-teacher distillation scope.

Instead of using a visit count based cutoff like katago with techniques like playout caps, I propose a cutoff based on best move visits, which naturally gives more visits to moves with high entropy and thus should receive more compute to reduce noise and provide harvestable subtrees.
