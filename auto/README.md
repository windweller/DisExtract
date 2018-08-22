If we apply an auto-encoder penalty, we will not be able to 
have BiLSTM that's 4096 x 2 size. It simply can't fit into 12 GB memory.

So instead, we try BiLSTM encoder with 2048 size, and decoder of 512 size
(there needs to be a compression layer before decoder).
