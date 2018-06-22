## DisSent

DisSent is a sentence embeddings method that provides semantic sentence representations. 
It is trained on BookCorpus data to predict the explicit discourse marker given
the context. The trained embeddings generalize well to many different tasks, but especially excel at any
discourse-related tasks such as PDTB (compare to SkipThought and current SOTA InferSent).

## Dependencies

This code is written in python 2. The dependencies are:

- Python 2.7 (with recent versions of NumPy/SciPy)
- Pytorch (recent version)
- NLTK >= 3

With minimum modification it should run on Python 3.

Our model is largely based on the InferSent repository.

## Download Model

We release three trained models for download.

We find that models trained on different set of discourse markers have slightly different performance. 
We in general find Books 8 and Books ALL have better generalization performance on discourse-related tasks,
but Books 5 have better performance on all other tasks.

We also took snapshot (by epoch) on these models, and in our experiment, we 
searched over all snapshots to find the best epoch.
The last epoch is not always the best epoch for generalization performance.

If you want to have access to all snapshots, please contact us directly. 

## Reference

Please cite our paper if you found this code useful and/or if you decide to use our dataset.

[1] Nie, A., Bennett, E. D., & Goodman, N. D. (2017). 
Dissent: Sentence representation learning from explicit discourse relations. 
arXiv preprint arXiv:1710.04334.

```
@article{nie2017dissent,
  title={Dissent: Sentence representation learning from explicit discourse relations},
  author={Nie, Allen and Bennett, Erin D and Goodman, Noah D},
  journal={arXiv preprint arXiv:1710.04334},
  year={2017}
}
```

## Dependency Pattern Instructions

```
depdency_patterns = [
    "still": [
        {"POS": "RB", "S2": "advmod", "S1": "parataxis", "acceptable_order": "S1 S2"},
        {"POS": "RB", "S2": "advmod", "S1": "dep", "acceptable_order": "S1 S2"},
    ],
    "for example": [
        {"POS": "NN", "S2": "nmod", "S1": "parataxis", "head": "example"}
    ],
    "but": [
        {"POS": "CC", "S2": "cc", "S1": "conj", "flip": True}
    ],
]
  
```

`Acceptable_order: "S1 S2""`: a flag to reject weird discourse term like "then" referring to previous sentence, 
but in itself a S2 S1 situation. In general it's hard to generate a valid (S1, S2) pair.

`flip: True`: a flag that means rather than S2 connecting to marker, S1 to S2; it's S1 connecting to marker, 
then S1 connects to S2.

`head: "example""`: for two words, which one is the one we use to match dependency patterns. 

## Parsing performance

Out of 176 Wikitext-103 examples:
* accuracy = 0.81 (how many correct pairs or rejections overall?)
* precision = 0.89 (given that the parser returns a pair, was that pair correct?)

This excludes all "still" examples, because they were terrible.
