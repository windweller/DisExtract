# Week 8 Report

We still have FairSeq ([link](https://fairseq.readthedocs.io/en/latest/getting_started.html#)) vs. Tensor2Tensor ([link](https://github.com/tensorflow/tensor2tensor/blob/master/docs/overview.md))

But both have Language Modeling, and both support Transformer

Both support beam decode

Unique advantages:

FairSeq supports large batch size, but not flexible loss function (we don't care about this...).

Tensor2Tensor might have better API wrapper, also BPE.

Disadvantages:

FairSeq needs seperate BPE runs.

Tensor2Tensor: TensorFlow might be hard to debug and understand. 

We re-organize and think about what is the ultimate goal/set of experiments that we want!

Let's stay with FairSeq right now. `SequenceScorer` is very useful.

**Plans for L2E/L2EC**

Novelty:

- Might potentially still beat Winograd (by training LM, and evaluate ensemble)
- Not many have looked into why-question
- Can generate sequence to question

Bad parts:

- No new model/algorithm

**TODO**

1. Gigaword
2. S -> Q

