# Week 1 Report

## Outline



## Balanced experiments

We follow up on a balanced corpus, where we threshold every marker to be the same as the least frequent marker. We run two experiments: balanced books 5 and balanced books all. Since we no longer consider generalization, we vary the hidden states size from 1024, 2048, to 4096. 



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

