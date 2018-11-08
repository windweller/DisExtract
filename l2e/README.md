Contains source code for training a Transformer LM (language model) on data that just contain `because`, without dependency parsing. 
The goal is to investigate the difference between learning on a chunk of text vs. dependency parsed well-formed sentence fragments.

We should be able to show that dependnecy parsing allows us to capture
and generate complete sentence fragments and LM 1). does not learn sentence boundary well; 
2). cannot generate good explanation even on a concentrated dataset. 