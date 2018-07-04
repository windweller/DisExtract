## Preprocessing Pipeline

So far the English preprocessing is relatively stable.

First step to do is to create a corpus-specific filtering/parsing file similar to `ptb.py` or `bookcorpus.py`.

Run two stage process layed out by these files like:

```bash
python ptb.py --filter
python ptb.py --parse
```

Then this would create a specific folder structure and parse everything down.

This data file will contain all discourse markers stored in `tsv` style.

Then we split this large data file by discourse markers, and run `producer.py` on each of the set.

## BookCorpus Preprocessing

`producer.py` usage:

python producer.py --data_file discourse_EN_FIVE_and_but_because_if_when_2017dec12.tsv --out_prefix discourse_EN_FIVE_and_but_because_if_when_2017dec12


## Gigaword Preprocessing

