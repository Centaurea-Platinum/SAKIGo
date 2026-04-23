This document defines how equivariant attention should be applied in v1.

Scope:
	Attention in v1 uses regular representations only. Irreducible-representation attention and explicit cross-group-fiber attention are out of scope for the baseline.

Terminology:
	D4 is the symmetry group of the square, with 8 elements.
	Grouped-query attention refers to the standard efficiency trick where multiple query heads share fewer key/value heads. It does not mean attention across D4 group fibers.

Tokenization:
	There is one attention token per board position.
	If the latent state has C regular representations, then one token is a tensor x_i[c, g] with shape (C, 8), where c indexes feature channels and g indexes the D4 group element.
	An implementation may flatten this to a vector of length 8 * C, but it must still be treated conceptually as C channels over an 8-dimensional regular-representation fiber.

Actual D4 vector form:
	For one regular representation, a token can be written as
	[x_e, x_r, x_r2, x_r3, x_m, x_mr, x_mr2, x_mr3]
	where the subscripts correspond to the D4 elements.
	Under a global board transform h in D4, the vector is permuted by the regular action:
	x[g] -> x[h^-1 g]
	The important point is that these 8 entries are not arbitrary coordinates. They are the same feature observed in 8 orientation/reflection slots.

Q, K, V construction:
	Q, K, and V are computed independently for each board position. They are not built from token pairs.
	For each head, Q_i, K_i, and V_i are obtained by equivariant regular-to-regular linear maps applied to x_i.
	The equivariant tying rule is:
	Q_i[a, g_out] = sum_{b, g_in} Theta_Q[a, b, g_out^-1 g_in] * x_i[b, g_in]
	and similarly for K_i and V_i.
	This means the weights depend on the relative group element g_out^-1 g_in, not on the absolute labels of g_out and g_in.
	Therefore every coordinate in the regular fiber does not share a single scalar weight. The map is constrained, but it is much richer than total sharing.

Toy example with an explicit small vector:
	To make the tying concrete, consider a simplified 4-slot cyclic toy fiber:
	x_i = [a, b, c, d]^T
	and one-step rotation acts by permutation:
	rho(r) x_i = [d, a, b, c]^T
	An equivariant linear map W must satisfy W rho(r) = rho(r) W, which forces W to be circulant:
	[u0, u1, u2, u3]
	[u3, u0, u1, u2]
	[u2, u3, u0, u1]
	[u1, u2, u3, u0]
	This shows the structure of the constraint:
	The map is not a fully dense matrix.
	The map is also not a single shared weight.
	Instead, each output slot sees a shifted copy of one learned template over relative offsets.

Pairwise attention score:
	For a pair of positions i and j, a head produces one scalar logit:
	l_ij = (1 / sqrt(8 * d_head)) * sum_{a, g} Q_i[a, g] * K_j[a, g] + b(orbit(dx, dy))
	where d_head is the number of regular-representation channels assigned to that head,
	dx and dy are the spatial offsets from i to j,
	and orbit(dx, dy) is the D4-invariant displacement key:
	orbit(dx, dy) = (max(abs(dx), abs(dy)), min(abs(dx), abs(dy)))
	This bias treats all D4-equivalent displacements as the same parameter.

Explicit expansion of the toy score:
	Let x_i = [a, b, c, d]^T and x_j = [p, q, r, s]^T.
	Because Q and K are equivariant, the effective bilinear score has the form
	l_ij = x_i^T M x_j
	where M is circulant:
	[m0, m1, m2, m3]
	[m3, m0, m1, m2]
	[m2, m3, m0, m1]
	[m1, m2, m3, m0]
	Expanding this gives
	l_ij =
		m0 * (a * p + b * q + c * r + d * s)
	  + m1 * (a * q + b * r + c * s + d * p)
	  + m2 * (a * r + b * s + c * p + d * q)
	  + m3 * (a * s + b * p + c * q + d * r)
	So the score can detect different relative orientation offsets. It is not limited to exact slot-wise matching.

Softmax and value aggregation:
	The softmax is applied over spatial positions j only:
	alpha_ij = softmax_j(l_ij)
	The output is then
	y_i[a, g] = sum_j alpha_ij * V_j[a, g]
	The attention weight alpha_ij is a scalar invariant, while V_j remains an equivariant value tensor.
	This preserves equivariance because a global D4 action permutes the value fiber but does not change the scalar routing weights.

Expressivity:
	This design can compare tokens using relative orientation information, not just raw invariant content.
	It can learn different pairwise interactions for different relative group offsets through the equivariant Q and K projections.
	With multiple regular-representation channels, it can compare channel c in one token to channel c' in another token while still respecting D4 symmetry.
	Standard grouped-query attention can still be used: fewer K/V heads than Q heads only changes sharing and memory cost, not the symmetry rule.

Restriction limits:
	The score cannot depend on absolute orientation labels such as north-facing versus east-facing. Only relative orientation is allowed.
	The simple v1 design gives one scalar attention weight per token pair. It cannot assign different pair-specific weights to different coordinates inside the value fiber.
	Therefore this design routes whole equivariant value tensors between positions, rather than performing pair-conditioned mixing across group components.
	This is the main expressivity limit of the baseline attention design.

Out-of-scope richer extension:
	A more expressive future design could output coefficients A_ij[delta] over relative group elements delta and aggregate shifted value fibers using those coefficients.
	That would allow pair-conditioned mixing across orientations, but it is intentionally excluded from v1 because it adds substantial complexity.

Decision for v1:
	Use one token per board position.
	Use regular-representation fibers inside each token.
	Use equivariant regular-to-regular linear maps for Q, K, and V.
	Use a scalar invariant attention score with D4-orbit relative bias.
	Use grouped-query attention only as an implementation optimization.
	Do not use explicit cross-group-fiber attention in v1.
