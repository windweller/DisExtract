

## TODO

[x] switch to better (more automatic) dependency parsing with corenlp
[ ] refactor extraction code to be prettier
[x] finish making test cases and editing parser for all discourse markers in english
[ ] get Chinese and Spanish Wikipedia corpora
[ ] run filter on English BookCorpus
[ ] check parser on English BookCorpus data
[ ] record dependency patterns for Chinese and Spanish

## Parsing performance

Out of 176 Wikitext-103 examples:
* accuracy = 0.81 (how many correct pairs or rejections overall?)
* precision = 0.89 (given that the parser returns a pair, was that pair correct?)

This excludes all "still" examples, because they were terrible.
