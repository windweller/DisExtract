## Note

This is the official repo that we used to train for DisSent paper. It is not yet cleaned up and not ready for official release. But it has come to our notice that the community still lacks efficient sentence representation models (public ones), so we are providing access to people our trained model. 

You can download the trained models from the following AWS S3 link:
https://s3-us-west-2.amazonaws.com/dissent/ (Books 5, Books 8, and Books ALL)

You can also load the models by using the following script:
https://github.com/windweller/SentEval/blob/master/examples/dissent.py
https://github.com/windweller/SentEval/blob/master/examples/dissent_eval.py

Please contact anie@stanford.edu if you have problem using these scripts! Thank you!

We wrote the majority of the code in 2017 when PyTorch was still at version 0.1 and Python 2 was still popular. You might need to adjust your library versions in order to load in the model.

## DisSent Corpus

The links to data is available under the same link: 

https://s3-us-west-2.amazonaws.com/dissent/ (Books 5, Books 8, Books ALL, Books ALL -- perfectly balanced)

If you scroll down, beneath the trained model pickle files, you can find download links for all our data.

We include all the training files (with the original train/valid/test split). We do not provide access to the original raw BookCorpus data at all.

```
aws s3 cp https://s3-us-west-2.amazonaws.com/dissent/discourse_EN_ALL_and_then_because_though_still_after_when_while_but_also_as_so_although_before_if_2017dec21_train.tsv .
```

## Dependency Pattern Instructions

```python
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

## TODO

[x] switch to better (more automatic) dependency parsing with corenlp
[x] finish making test cases and editing parser for all discourse markers in english
[x] rewrite producer
[ ] run model on new dataset for English
[ ] get Chinese and Spanish corpora
[ ] check parser on a few examples of English BookCorpus data
[ ] record dependency patterns for Chinese and Spanish
[ ] what if this doesn't work?

## Parsing performance

Out of 176 Wikitext-103 examples:
* accuracy = 0.81 (how many correct pairs or rejections overall?)
* precision = 0.89 (given that the parser returns a pair, was that pair correct?)

This excludes all "still" examples, because they were terrible.
