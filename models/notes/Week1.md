# Week 1 Report

## Outline

1. We investigated old experiment data, we can't conclude that `because` is learned badly because algorithms fail to understand causal relations (or acquire real-world knowledge) (confounding factor: number of training examples in corpus)
2. We then follow up with balanced experiments.
3. Analyze the error examples with visualization

## Balanced experiments

We follow up on a balanced corpus, where we threshold every marker to be the same as the least frequent marker. We run two experiments: balanced books 5 and balanced books all. Since we no longer consider generalization, we vary the hidden states size from 1024, 2048, to 4096. 

Books ALL balanced: **181,184** training examples, each marker has **13,421** examples.

Books 5 balanced: **836,790** training examples, each marker has **167,358** examples.

**Books ALL Model Performance**

| Model | Epoch | Accuracy | Precision | Recall | F1   |
| ----- | ----- | -------- | --------- | ------ | ---- |
| 1024  | 9     | 46.0     | 46.8      | 46.8   | 46.8 |
| 2048  | 7     | 46.0     | 47.6      | 46.9   | 47.0 |
| 4096  | 7     | 47.0     | 47.4      | 47.1   | 47.1 |

(Report from the test set, because we use valid set to early stop) (conclusion: 4096 performs slightly better but 2048 is just as close, and takes half the time to train)

**Model 4096 Per-class Performance**

| Books ALL | Precision | Recall   |
| ------- | --------- | -------- |
|after|*0.58*|0.52|
|also|0.42|0.47|
|**although**|**0.35**|0.42|
|and|0.54|0.48|
|as|0.49|0.51|
|**because**|**0.38**|**0.37**|
|before|0.52|0.53|
|**but**|**0.36**|**0.38**|
|if|0.57|*0.61*|
|so|0.52|0.54|
|still|0.55|0.57|
|then|0.49|0.48|
|though|0.55|0.42|
|**when**|**0.38**|**0.34**|
|while|0.42|0.43|

Conclusion: when threshold for the same amount of data

1. `because` ranked the third lowest in both precision and recall. 
2. Other three markers with similar performance are `but`, `when`, and `although`. If we only look at `when` and `although`, both have much worse extraction quality than `because`.

![ExtractionError](./imgs/ExtractionErrorSmall.jpg)

**For Books 5 Model Performance**



## Investigating old data

We investigated old record, trying to figure out why `because` has lower precision and recall. We conclude that `number` of training example is the deciding factor. Discourse markers with similar frequency have similar precision and recall such as `while`. However, despite  `because`'s low frequency, humans are able to capture it very well. Also, markers like `after` and `still`, despite having much lower frequency, still have relatively high precision. This shows the hypothesis that **some markers are harder to learn** can still be proven to be correct.

| Books 5 | Number   | Precision | Recall   |
| ------- | -------- | --------- | -------- |
| and     | 50958    | 0.72      | 0.78     |
| because | **8468** | **0.45**  | **0.36** |
| but     | 51574    | 0.71      | 0.73     |
| if      | 23642    | 0.79      | 0.75     |
| when    | 26185    | 0.61      | 0.62     |
| while   | **8055** | **0.46**  | **0.36** |
| after   | **4886** | **0.55**  | **0.42** |
| also    | **811**  | **0.36**  | **0.14** |
| still   | **698**  | **0.42**  | **0.21** |

| Books 5 Val Pred \ Label | True | False  |
| ------------------------ | ---- | ------ |
| Positive                 | 3105 | 2689   |
| Negative                 | 5362 | 152359 |

