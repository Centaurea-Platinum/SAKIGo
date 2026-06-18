Go's ruleset are very complicated. Furthermore, it does not map neatly to a hypercube, as some rules naturally excludes others. Thus, only a subset of rules are used to avoid overhead and rare sampling, and a one-hot encoding for correlated rules are used.
The actual structure of a single one-hot will be something like [1,0,0], [0,1,0], or [0,0,1], but for simplicity, I will be recording them as an enumeration(1,2,3), and the vector length wil be implied.
In the engine, different one-hot will be concated into a single vector, and feeded into two mlp for each film(bias+scale) injection site.
Scoring:
    (1) Area
    (2) Area + AncientChinese   #Penalized 2 points for each unconnected piece of alive group
    (3) Territory
    (4) TerritoryWithSekiScore  #Stones in seki do not have territory points, but they can

Ko:
    (1) SimpleKo    #long repeat = draw
    (2) PositionalSuperKo   #Board move cannot revert to any previous board position

Suicide:
    (1) Yes
    (2) No

Komi + CapturedStones: 
    #This is two scalars in [-1,1], normalized via division board area. CapturedStones(My-Opponent) can exceed board area, but I can leave it out.
    #Handicap related rules are handled by komi value implicitly