# Output heads

The active model has four heads:

- spatial policy logits and budget logits over the board, each with a separate pass logit;
- global WDL logits with four classes, and one normalized score scalar.

Spatial heads use 1x1 convolutions over board features. Global heads use MLPs over register tokens. Ownership and auxiliary heads are not part of the current model.
