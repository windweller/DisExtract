
# coding: utf-8

# # Formatting BookCorpus for BERT MLM
# 
# 
# ## Preparing input to this script
# 
# Filter sentences using `bookcorpus.py`. Creates files:

# In[1]:


input_files = [
    "/home/anie/DisExtract/preprocessing/corpus/bookcorpus/markers_ALL/sentences/ALL.json",
    "/home/anie/DisExtract/preprocessing/corpus/bookcorpus/markers_EIGHT/sentences/EIGHT.json",
    "/home/anie/DisExtract/preprocessing/corpus/bookcorpus/markers_FIVE/sentences/FIVE.json"
]


# Each of which is formatted:
# 
# ```
# {
#     "discourse marker 0": {
#         "previous": ["S1_0", "S1_1", "S1_2", ...],
#         "sentence": ["S2_0", "S2_1", "S2_2", ...]
#     }
#     "discourse marker 1": {
#         "previous": ["S1_0", "S1_1", "S1_2", ...],
#         "sentence": ["S2_0", "S2_1", "S2_2", ...]
#     }
#     ...
# }
# ```

# In[2]:


import json

def read_sentence_pairs(input_files):
    S1 = []
    S2 = []

    for f in input_files:
        d = json.load(open(f, "r"))
        for k, s_pairs in d.items():
            S1.extend(s_pairs["previous"])
            S2.extend(s_pairs["sentence"])

    return zip(S1, S2)

sentence_pairs = list(set(read_sentence_pairs(input_files)))


# ## Output of this script
# 
# Randomly shuffle S1-S2 pairs across all discourse markers.

# In[3]:


import numpy as np

np.random.seed(12345)
np.random.shuffle(sentence_pairs)


# Convert to a single set of `train.txt` (90%), `valid.txt` (5%) and `test.txt` (5%) files formatted:
# 
# ```
# S1_0
# S2_0
# 
# S1_1
# S2_1
# 
# S1_2
# S2_2
# 
# ...
# ```

# In[4]:


import os
from os.path import join as pjoin

def write_bert_files(sentence_pairs, output_dir, train_proportion=0.9):
    assert(train_proportion < 1 and train_proportion > 0)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    n_total = len(sentence_pairs)
    
    test_proportion = (1-train_proportion)/2
    n_test = round(n_total * test_proportion)
    n_valid = n_test
    n_train = n_total - (n_test + n_valid)

    splits = {}
    splits["test"] = sentence_pairs[:n_test]
    splits["valid"] = sentence_pairs[n_test:(n_test + n_valid)]
    splits["train"] = sentence_pairs[(n_test + n_valid):]

    words = set()
    
    for split, sentence_pairs in splits.items():
        write_path = pjoin(output_dir, "%s.txt" % split)
        with open(write_path, "w") as write_file:
            for S1, S2 in sentence_pairs:
                words.update(S1.split(" "))
                words.update(S2.split(" "))
                write_file.write(S1)
                write_file.write(S2)
                write_file.write("\n")
                
    with open(pjoin(output_dir, "vocab.txt"), "w") as write_file:
        write_file.write("\n".join(words))
    

output_dir = "/data/erindb/bookcorpus/sentence_pairs_with_discourse_markers/"
write_bert_files(sentence_pairs, output_dir)

