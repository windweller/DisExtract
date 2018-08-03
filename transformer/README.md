# Result

Overall, Transformer with auxiliary language modeling loss is better than vanilla LSTM (without language modeling loss).

I only trained on Books 5 because I tuned some hyperparameters and it would be too slow to do that for Books 8 and Books ALL.

| Books 5 Task -- Model        | Accuracy (overall) |
| ---------------------------- | ------------------ |
| Bidir LSTM                   | 77.3               |
| Transformer + Language Model | 80.0               |

However, this improvement did not carry over to the SentEval training tasks. This is a noticeable and unfortunate trend: for SentEval style tasks (that are relatively short, especially MR, CR), Transformer cannot perform better. Similar observations are also made in [Universal Sentence Encoder](https://arxiv.org/abs/1803.11175) and [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf). 

| Model       | MR    | CR    | SUBJ  | MPQA  | SST2  | TREC | SICKRelatedness | SICKEntailment | MRPC        | STS14         | ACC_AVG |
| ----------- | ----- | ----- | ----- | ----- | ----- | ---- | --------------- | -------------- | ----------- | ------------- | ------- |
| Transformer | 71.9  | 78.01 | 86.76 | 87.51 | 75.07 | 76.2 | 0.6671          | 73.94          | 72.75/72.75 | 0.2389/0.2389 | 77.768  |
| Bidir LSTM  | 80.18 | 85.38 | 93.15 | 90.22 | 82.81 | 91.2 | 0.8449          | 83.54          | 76.12/76.12 | 0.4742/0.4742 | 85.325  |

Both models are pre-trained on Books 5 corpus. Evaluation is done in the same way as before: extract a fixed sentence vector and run logistic regression on it.