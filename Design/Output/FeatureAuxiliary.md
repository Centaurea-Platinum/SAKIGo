> **Status: Not currently considered.** This note is retained only as a possible
> future direction; current work is limited to KataGo-teacher distillation.

6 planes:
    MyStones
        1 liberty
            Move on adjacent empty pos can increase liberty
            Move on adjacent empty pos cannot increase liberty
        2 liberty
        3+ liberty
    OpponentStones
        1 liberty
            Move on adjacent empty pos can increase liberty
            Move on adjacent empty pos cannot increase liberty
        2 liberty
        3+ liberty
    EmptyPositions
        My
            Capture oppourtunity
        Opponent
            Capture oppourtunity
    Boundary
        Corner
        Edge
    NonTrivialIllegal
        Suicide
        Ko/SuperKo
