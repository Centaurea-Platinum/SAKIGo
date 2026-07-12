> **Current scope: distillation only.** Budget learns KataGo's raw policy and
> policy learns the teacher's top-1 move. Search-time use, alternative policy
> variants, and other future semantics are not currently considered.

Policy and budget are separate distillation targets. Budget matches KataGo's raw
policy distribution, while policy matches the teacher's top-1 move.
Pass is represented as one extra logit appended to the n^2 board-move logits. Board moves and pass enter the same softmax.
In training, illegal moves are not masked but treated as part of the loss. In inference, do a mask on illegal moves. This is not a test train mismatch, the model should learn to produce legal moves, the mask in inference should be regarded as a precaution.
