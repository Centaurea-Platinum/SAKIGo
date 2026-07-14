> **Current scope: book distillation only.** Budget learns normalized concrete
> book raw visits (`v`), while policy is uniform across concrete moves tied at the
> mover-optimal rounded book score. Search-time use and alternative future
> semantics are not currently considered.

Pages containing an aggregate `other` row remain eligible, but that row is
discarded before either target is constructed. Policy and budget are separate
heads even though they are derived from the same book page.

When one exported row lists several symmetry-equivalent coordinates, its `v`
count is assigned in full to every listed concrete action before normalization.
This expands the representative move into its symmetry orbit; it does not split
one visit count across that orbit. Adjusted visits (`av`) are not used.
Pass is represented as one extra logit appended to the n^2 board-move logits. Board moves and pass enter the same softmax.
In training, illegal moves are not masked but treated as part of the loss. In inference, do a mask on illegal moves. This is not a test train mismatch, the model should learn to produce legal moves, the mask in inference should be regarded as a precaution.
