This file denotes the config of the model for v1 baseline.
Stem:
    3x3 conv, (4, 8, 16) //8, 16 represent number of regular representation, it corresponds 64, 128 channels

Trunk:
    8 residual blocks. Each block has the structure(4 head attention, 16 to 64 through 1x1 conv, depthwise separate 3x3 conv, 64 to 16 through 1x1 conv.)

Heads:
    Tactic head and spatial heads uses a 4 head attention and then average over a regular representation, and then 1x1 convs(16,4,1) into a scalar map
    Global heads uses attention pooling with 4 heads, average across regular representation to form a vector of dimension 16, and then a mlp of (16,4,1)
    Policy runs an attention pooling after cross attention to form a vector of dimension 16 and then a mlp of (16,4,1) to determine passing probability

Film injection. The one hot encoded vector of dimension 9 will be passed through two mlps of (9,16,16,16) to form the bias and weight at each injection site. The 31 dimension vector of percentile score will be feed into two mlp of (31,16,16,16) to modify policy.

