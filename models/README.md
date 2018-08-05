Our original model is written in Tensorflow (first release on ArXiv).

In this second and release version, we adapted code from [InferSent](https://github.com/facebookresearch/InferSent) instead, 
including model, data loading and training code. 

Preprocessing is done with StanfordNLP tokenization. 

Our model code is different from Infersent code in:
1. We included addition to get order-invariant information, and use 
subtraction to get order-specific information.
2. In the original model, we did not use a fully-connected layer after
LSTM.


If you want to take a look at the old Tensorflow code, here's the repo: [https://github.com/windweller/discourse](https://github.com/windweller/discourse)

We also extended InferSent code to include logging and automatically saves log files and hyperparameters.
Snapshots are also taken due to long training time.

## Evaluate

`evaluate.py` is designed to actually process files that are split into "train", "dev", "test". It will simply load them all and combine
those files into a test file. No need to worry about combining them.