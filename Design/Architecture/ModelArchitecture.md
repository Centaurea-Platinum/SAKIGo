The model's trunk will use equivariant GQA. Reference [./EquivariantAttention.md]
G = D4

The stem first lifts scalar input planes into D4-regular features, then uses D4-regular equivariant fiber mixing to adjust width to trunk width. The lift repeats scalars across the group axis and regular 1x1 mixing keeps group-constant fibers group-constant, so all 8 stem output components are identical; components first diverge at the first RoPE'd attention in block 1.

The trunk will be a bottlenecked nested residual setup with register tokens. Each block will consist of:
    in                                      #m regular reps
    x_1 = Nonlinear(f^1(norm(in)))          #spatial mlp/1x1 conv, n regular reps
    x_2 = f^2(norm(x_1)) * alpha_1 + x_1    #attention, n regular reps
    x_3 = f^3(norm(x_2)) * alpha_2 + x_2    #attention, n regular reps
    out = f^4(norm(x_3)) * beta + in        #spatial mlp/1x1 conv, m regular reps

    Only a subset of trunk block gather/broadcasts register information back to board features:
    R_i = R_i + gamma_1 * g^1(norm(R_i), norm(in))      #Q from register, KV from in
    out_{i,j} = out_{i,j} + gamma_2 * g^2(norm(out), norm(R_i))      #Q from out, KV from register

    Residual constant {alpha_1, alpha_2, beta, gamma_1} are independent in each block, initialize to 1/sqrt(2*Block_count). gamma_2 exists only in the final broadcast block and uses the same initialization.

Global heads (win/draw/loss/no-result) will be a D4-regular MLP applied to the merged register tokens, then collapsed to invariant logits by averaging over the D4 axis.

Spatial heads (policy) will be D4-regular equivariant fiber MLPs, then collapsed to invariant policy logits by averaging over the D4 axis.

Baseline designs:
    norm will use RMS norm computed over all regular rep feature channels
    Nonlinear is the activation selected by the model spec, baseline SiLU. Only f^1 carries it; f^4 stays linear so the residual update stays sign-unconstrained. Stem and head MLPs place a fixed SiLU between layers, never after the final layer.
    positional embedding will use 2D RoPE with spec-listed global and local frequency lists, baseline one global {pi} and one local {pi/2}. Each frequency rotates 4 head dims (row+col pairs); head_dim beyond 4 * total stays unrotated. RoPE rotates board-side Q/K only; register tokens are unrotated.
