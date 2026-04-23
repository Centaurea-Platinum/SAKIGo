The Go rules will be encoded using a stack of one hot encoding.

Scoring: Area/AncientChinese/Territory/TerritorySekiScored

Seki scoring(In Japan rules, stone in seki does not have territory points. In some rules, controlled empty crosspoint from stones in seki counts toward territory)

Ancient chinese connectivity(In ancient chinese rules, each group of alive stones is penalized by two area points to accomodate for how they need two illegal points to remain alive. Lock to 0 for territorial scoring)

Ko rules: SimpleKoDraw/SimpleKoSeki/PositionalSuperKo

I acknowledge how there are more than one possibility for SuperKo rules, but I'm keeping the most used one only. 

Long repeat seki(corresponds to long repeat forces a draw or stones in long repeat are treated as in seki)

Suicide: No/Yes
Simple boolean, encoded as 1,0 and 0,1 to match one-hot format. Useful in edge cases like ko threats

The format will be: 0,0,0,0|0,0,0|0,0
Chinese rule will correspond to 1,0,0,0|0,0,1|1,0