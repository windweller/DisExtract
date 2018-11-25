
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

# In[6]:


import json

def read_sentence_pairs(input_files):
    sentence_pairs_data = {}
    sentence_pairs_stats = {}

    # iterate through discourse marker sets
    for f in input_files:
        # read data for discourse marker set
        d = json.load(open(f, "r"))
        # iterate through discourse markers
        for marker, sentence_pairs in d.items():
            # convert to tuples (S1, S2)
            sentence_pairs = zip(sentence_pairs["previous"], sentence_pairs["sentence"])
            # get unique pairs
            sentence_pairs = list(set(sentence_pairs))
            # store data
            sentence_pairs_data[marker] = sentence_pairs
            sentence_pairs_stats[marker] = len(sentence_pairs)
    
    return sentence_pairs_data, sentence_pairs_stats

sentence_pairs_data, sentence_pairs_stats = read_sentence_pairs(input_files)
print(sentence_pairs_stats)


# ## Output of this script
# 
# Randomly shuffle S1-S2 pairs across all discourse markers.
# 
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

# In[18]:


import os
from os.path import join as pjoin
import numpy as np
np.random.seed(12345)

def write_bert_files(sentence_pairs_data, sentence_pairs_stats, output_dir, train_proportion=0.9):
    assert(train_proportion < 1 and train_proportion > 0)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(pjoin(output_dir, "stats.json"), "w") as write_file:
        write_file.write(json.dumps(sentence_pairs_stats))
        
    splits = {split: [] for split in ["train", "valid", "test"]}
    
    words = set()

    for marker, sentence_pairs in sentence_pairs_data.items():
        # how many sentence pairs in each split for this marker?
        n_total = len(sentence_pairs)
        test_proportion = (1-train_proportion)/2
        n_test = round(n_total * test_proportion)
        n_valid = n_test
        n_train = n_total - (n_test + n_valid)

        # distribute sentence pairs between splits for this marker
        np.random.shuffle(sentence_pairs)
        splits["test"].extend(sentence_pairs[:n_test])
        splits["valid"].extend(sentence_pairs[n_test:(n_test + n_valid)])
        splits["train"].extend(sentence_pairs[(n_test + n_valid):])
        
    for split, sentence_pairs in splits.items():
        # shuffle across discourse markers
        np.random.shuffle(sentence_pairs)
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
write_bert_files(sentence_pairs_data, sentence_pairs_stats, output_dir)

