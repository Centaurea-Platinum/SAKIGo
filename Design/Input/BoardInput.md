Minimal setup: leave it to the model to figure it out
6 planes:
    MyStones
    OpponentStones
    EmptyPositions
    Boundary    #This also allows for stuff like non-rectangular boards
        Corner
        Edge
    NonTrivialIllegal(Suicide, Ko, SuperKo,...)