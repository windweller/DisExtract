Our original model is written in Tensorflow (first release on ArXiv).

In this official release version, we heavily adapted code from [InferSent](https://github.com/facebookresearch/InferSent), 
including model, data loading and training code. 

Preprocessing is done with StanfordNLP tokenization. 

Our model code is different from NLI code:
1. We added addition to get order-invariant information, and use 
subtraction to get order-specific information.
2. In the original model, we did not use a fully-connected layer after
LSTM.


If you want to take a look at the old Tensorflow code, here's the repo: [https://github.com/windweller/discourse](https://github.com/windweller/discourse)

The data distribution for Discourse 5 is:

{'but': 508648, 
'when': 552540, 
'because': 116444, 
'if': 491394, 
'and': 818634}

We extended InferSent code to include logging and automatically saves log files, and save hyperparameters.
We also save models for more than one training epoch.