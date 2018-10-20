# Week 4 Report

## Plans

1. ReasonQA
   - How to turn a statement (cause) into a question?
     - Erin provided a pattern, try it!!
2. Think about using A/B testing
   - Ask humans to distinguish between human reason vs. machine generated reason
   - Rank reasons for plausiblity

## TODO List

1. Run PGNet on the cluster, install everything

2. Generate new ReasonQA dataset

## ReasonQA

Title - ReasonQA: Learning to Explain Through Automatic Dataset Curation

We first prep the Gigaword (116M) and NewsCrawl (191M) corpus in the format that matches DrQA Document Retriever's [format](https://github.com/facebookresearch/DrQA/tree/master/scripts/retriever). Since the original 300M dataset is rather large, we decide to collect only a fixed context! We find the `because` sentence pairs (that does not contain `because of`), and then use the 

So the reason we want a few paragraphs is not that the answer is `contained` in these paragraphs. The paragraphs are only providing `context`! Why questions are not as explicit in the paragraphs as other types of questions!

We choose the +/- 10 sentences when we first find `because` as the context. If two `because` appeared in the same context, the range will overlap.

We aim to create the largest distant supervision QA dataset to date for why question answering.

One nice thing we might be able to show is that **ReasonQA is perfect for pre-training PGNet type model, without DrQA** (rationale extraction). DrQA paper showed that PGNet drastically under-performed DrQA and DrQA + PGNet. This is perhaps the training set **100k** is too small.

Sadly, since news_crawl is shuffled, we need to hash everything. We do threshold for length, <50. We get **186,301,978 (186M)** sentences as candidates for context! This is smaller than 116 + 191, but this probably is the result of sentence tokenization error occurred in the first stage. When write the paper, avoid talking about this part.

 Then we build **DrQA document retriever** to pick the the candidate sentences. DrQA has a lot of filtering work. We only want some vague context. We don't care about exact retrieval.

**DrQA document retriever** will index everything and generate hash value for the bigrams. It fast-retrieves over millions of documents. We use it to match for S1 and S2 (it's ok that S1 and S2 are not exactly the same sentence as before. Also it's ok to retrieve more context/background than the original position of the sentence -- common knowledge / background / context can come from any relevant stories).

We use a `inclusion_match` function to determine how much of query showed up in the reference. We manually tune and set the threshold to be **0.35** (meaning 65% of the query showed up in reference, a significant overlap). We remove reference that has significant overlap because we don't want repeating sentences or paraphrased sentences. For example:

```
That banned his most threatening challenger , Rally leader Alassane Ouattara , from running for president because he is only half - Ivorian .
```

and

```
An election law Bedie pushed through in December barred the strongest challenger , Alassane Ouattara of the Rally of the Republicans , from running for president because he is only half - Ivorian . Other opposition candidates boycotted the election in protest and have urged militants to disrupt voting . 
```

Even though the second (reference) is a paraphrasing, like `strongest` instead of `most threatening`,  we still don't want it to serve as context.

We also demand the `score` returned by DrQA's method `doc_names, doc_scores = ranker.closest_docs(query, k)` to be larger than **200**. This is also manually tuned and set. Anything lower than 200 looks extremely bad. 

We will exclude the data point if the query does not get references that match our criteria. Since we grabbed the +/- 10 sentences from the `because` sentence, if the match returns empty, it means the context around the `because` sentence bears very little resemblence to the sentence itself. But this situation should be rare.

The commands used to preprocess:

```bash
cd ~/DisExtract/preprocessing
python drqa_preprocess.py

# build dataset
cd ~/DrQA/scripts/retriever
python3 build_db_single_file.py /home/anie/DisExtract/preprocessing/corpus/because/because_db.txt /home/anie/DisExtract/preprocessing/corpus/because/because_db.db

python3 build_tfidf.py /home/anie/DisExtract/preprocessing/corpus/because/because_db.db  /home/anie/DisExtract/preprocessing/corpus/because/ --num-workers 8

python3 interactive.py --model /home/anie/DisExtract/preprocessing/corpus/because/because_db_buffer10-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz --data_json /home/anie/DisExtract/preprocessing/corpus/because/because_db_buffer10.txt --filter

```

### Fast and Memory Efficient Processing

We hit a small hiccup because the OOM error on a 500GB memory machine.

So we decide to split into smaller chunks (sharding)!

We first use Ubuntu's own tool to split the text file. Original file has `186301978` lines. We want 20 files, so we get 186301978 / 10 = 18630197 (18.6M lines per file)

```bash
split -l 18630197 -d because_db.txt because_db_split

# build dataset
cd ~/DrQA/scripts/retriever
python3 build_db_multi_file.py  /home/anie/DisExtract/preprocessing/corpus/because/shards/  /home/anie/DisExtract/preprocessing/corpus/because/shards_db because_db_split 10

# run parallel
bash exec_tfidf_multi.sh

# interactive over all sharded databases
python3 interactive.py --model /home/anie/DisExtract/preprocessing/corpus/because/shards_db --data_json /home/anie/DisExtract/preprocessing/corpus/because/because_db.txt --filter
```

This can split into 20 shards. Then we use script `build_db_multi_file.py` that searches for shards. One note: do need to clear out the `shards_db` if you need to re-execute the command.

This attempt has failed...

### Processing again from Ordered News Crawl

We go back to 10 before 10 sentences after strategy.

ReasonQA is a bit different from SQuAD and CoQA. It's prohibitively abstractive. The paragraph only provides context, but does not include the actual answer. It also does not try to match and detect `because`, and generate whatever is after `because`. However, the paragraph might still contain `because` sentences. It only will not contain the S1(Q) nor S2.

It is generating the best explanations, asking to reason about what are the plausible scenarios that could happen under that context.

We processed the new corpora, which is 43G with all file structure. This is different from the originally shuffled dataset, which is about 22 GB in total. The final tokenized file has 27GB in total, compared to originally extracted dataset that also has 22 GB storage. The new tokenized file has **233,195,563 (233M)** sentences. The original file only had **191M** sentences.

So:

1. We compare if this new 27GB file can give us more `because` sentence pairs! We investigate by filtering first.
   - We obtained **3,743,177 (3.7M)**  `because` sentences, with just filtering. Compared to **3.1M** sentences filtered through.

## Paper Summary

**A Semi-Supervised Learning Approach to Why-Question Answering**

(https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12208/12056, 2016)

This is only in Japanese and they require annotated corpus for "causal" relations.

**Developing an approach for why-question answering**

(https://pdfs.semanticscholar.org/d431/7fba0a026d6bb957f2a48a981775ca9dc58f.pdf)



**DrQA: Reading Wikipedia to Answer Open-Domain Questions**

(https://arxiv.org/abs/1704.00051)

DrQA shows that through a bag of clever engineering tricks, we can scale up MR to MRS (Machine Reading at Scale).

Also, DrQA uses SQuAD to learn typed questions (what, where, when, who, etc.). Use it as base model, it can further be trained on other datasets using distant supervision (WikiMovie, CuratedTrec, etc.).

Each task can be thought of as giving the module/model/system an ability to do something (SQuAD --> answer different types of questions).

DrQA also creates distant supervision dataset using document retriever (something we can also do).

It's a nice training paradigm that should work on all retrieve -> read type of framework.

Distant supervision -> they find top 5 Wikipedia articles that contain relevant fact for datasets like WebQuestions and WikiMovies. Then since they have actual answer (short span), they find paragraphs that contain the answer, discard the rest.

**CoQA: A Conversational Question Answering Challenge**

(https://arxiv.org/pdf/1808.07042.pdf)

Nice dataset analysis section, including a section on **Linguistic Phenomenon** (lexical match, paraphrasing -- synonymy, actonymy, hyponymy and negation, pragmatics) 

(we can analyze our pairs collecting similar statistics)

**Models** section

First, co-attention or question-aligned paragraph encoding, etc., can be subsumed under the Transformer type of encoding (no need for explicit segmentation between question and answer).

Second, subsume extractive model under **Pointer-Generator network** (PGNet): p \<q\> q_{i-n} \<a\> a\_{i-n} ... (Just like the OpenAI paper.)

DrQA + PGNet: according to Danqi, PGNet only gets fed with spans from DrQA, and in their mind, PGNet is used to squash "Mary, Andrea, and Mike" to "Three" in order to generate `abstractive` type answer.

In CoQA, there are only 32.2% that are abstractive answers. 

Also according to Danqi, if the answer is purely extractive, they rely solely on a SQuAD model. We can use a similar strategy :) use a good SQuAD model to answer extractive questions (pre-trained Transformer?) and use our PGNet to answer abstractive question?

But it still might perform quite badly! Especially considering CoQA has many "three" or "yes/no" type answers. But maybe as a pre-training tool for PGNet, it's not too bad!

