Time auxiliary heads are essentially heads that try to predict the value of main heads after some \delta t.
Mathematically, this will be:
    f(x_t): Output of a head at board state x_t
    g(x_t) = f(x_{t+n}), where g is auxiliary head

Some potentially useful auxilary head are for ownership and score.