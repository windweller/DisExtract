
## Dependency Pattern Instructions

```python
depdency_patterns = [
    "still": {
    "POS": ["RB"],
    "S2": ["advmod"],
    "S1": ["parataxis", "dep"],
    "acceptable_order": "S1 S2"},
    
    "for example": {
    "POS": ["NN"],
    "S2": ["nmod"],
    "S1": ["parataxis"],
    "head": "example"},
    
    "but": {
    "POS": ["CC"],
    "S2": ["cc"],
    "S1": ["conj"],
    "flip": True}
]
  
```

`Acceptable_order: "S1 S2""`: a flag to reject weird discourse term like "then" referring to previous sentence, 
but in itself a S2 S1 situation. In general it's hard to generate a valid (S1, S2) pair.

`flip: True`: a flag that means rather than S2 connecting to marker, S1 to S2; it's S1 connecting to marker, 
then S1 connects to S2.

`head: "example""`: for two words, which one is the one we use to match dependency patterns. 

## TODO

☐ switch to better (more automatic) dependency parsing with corenlp
☐ refactor testing to be prettier
☐ finish making test cases and editing parser for all discourse markers

