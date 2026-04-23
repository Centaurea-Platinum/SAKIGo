These are hypothetical stuff not ready fo rimplementation.
Sparse attention(more useful for larger boards, cross attention between 361 tokens for standard 19 Go is fine): 
    Query pairs: The first query token is evaluated with every board position, asking info like are you in a ladder threat. Position with weight above a certain threshold forms the set A with elements a_1,...,a_n. The second query token is evaluated with every board position, asking related questions like are you on the outside and can effect ladder. Positions with weight above certain threshold forms a second set B with element b_1,...,b_m. Cross attention is then applied n*m times on each ab pair. The second query token could be modified by the query token, so you have n sets of b. 
    Light-heavy attention: A light attention is ran first to mask position pairs with low importance out before the full heavy attention is ran.
Incrementalization(highly theoretical as of now):
    The Go game is mostly updated locally at the position of move, thus it seeems inefficient for the whole neural net to be ran again. An update on the latent space could perhaps be learned instead.
MOE:
    Small large separation: smaller models are used for simpler position, larger ones for more complex ones.
    Expertise separation: Different experts have respective fields, such as one model might be used for opening, one for life and death, and so on.
LoopNet:
    The same weights are reused k times in a residual manner, saving parameter count.