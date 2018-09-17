# Week 3 Report

## Plans

1. Process **Wikipedia dataset** (currently the dump is roughly the same size as News Crawl 2007-2017)
   - Extract `because` from them
   - Then we have two types of `because` relations: one fact-based (explaination type of `because`), one causation based (Wikipedia). We can see transfer learning between the two.
2. Try out **winograd schema** challenge
   - Common-sense LM framework
3. Try to generate context and make it a conditional contextual generation
   - A Question-Answer dataset
4. Probing tests
   * It seems to learn logical structure (like "not A because not B")
   * Can we probe it? This prompts the fact that learning `but` could also be interesting.
5. Parse for 4-5 discourse marker
   - Low priority...treat it as multilingual generation
   - Or PERSONA, each discourse is a different "PERSONA" https://arxiv.org/pdf/1801.07243.pdf (this new work plus Jiwei Li's work)
6. Read into OpenNMT more carefully. If BLEU is not a good objective metric, is negative log-likelihood?

### Problems

1. BLEU can't be used as objective metric. If people are going to use this dataset as their dataset, how can they measure success?
   - Idea: use a LM to measure whether a response is likely to be generated? (Look into that adversarial evaluation paper for dialogue) (But if LM is good enough, it should generate good explanation anyway, but it can't...so a bit paradoxical)
2. What are the experiments to prove the model is useful?
   - Winograd Schema Challenge
   - Logical test for learning the structure (then we only need `but`)
     - Can collect negations in antecedent and consequent (sentA and sentB) (Discover if the patterns captured in Seq2Seq match patterns in real sentence pairs)
3. 

