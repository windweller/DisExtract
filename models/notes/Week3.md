# Week 3 Report

## Plans

1. Process **Wikipedia dataset** (currently the dump is roughly the same size as News Crawl 2007-2017)
   - Extract `because` from them
   - Then we have two types of `because` relations: one fact-based (explaination type of `because`), one causation based (Wikipedia). We can see transfer learning between the two.
2. Try out **winograd schema** challenge
   - Common-sense LM framework (run this)
   - Use Enc-dec and rank as usual 
   - Use dec to actually generate reasons, and see how good it is (if previous task fails, we can still claim we generated good alternative, albeit more bland explanations)
   - If Transformer enc-dec didn't work, you can always try out LSTM
   - Ha, our own dataset might also be regarded as the commonsense dataset!
   - Error analysis 
3. Probing tests
   * It seems to learn logical structure (like "not A because not B")
   * Can we probe it? This prompts the fact that learning `but` could also be interesting.
   * Collect negations (in antecedent and precedent, see if Seq2Seq captures it well) (Should compare with LM, if we are having a paper)
4. Try to generate context and make it a conditional contextual generation
   - A Question-Answer dataset
5. Parse for 4-5 discourse marker
   - Low priority...treat it as multilingual generation
   - Or PERSONA, each discourse is a different "PERSONA" https://arxiv.org/pdf/1801.07243.pdf (this new work plus Jiwei Li's work)
   - This is also called a **typed decoder**. There are "hard typed decoder" and "soft typed decoder". (https://arxiv.org/pdf/1805.04843.pdf)
6. Read into OpenNMT more carefully. If BLEU is not a good objective metric, is negative log-likelihood?

### Problems

1. BLEU can't be used as objective metric. If people are going to use this dataset as their dataset, how can they measure success?
   - Idea: use a LM to measure whether a response is likely to be generated? (Look into that adversarial evaluation paper for dialogue) (But if LM is good enough, it should generate good explanation anyway, but it can't...so a bit paradoxical)
2. What are the experiments to prove the model is useful?
   - Winograd Schema Challenge
   - Logical test for learning the structure (then we only need `but`)
     - Can collect negations in antecedent and consequent (sentA and sentB) (Discover if the patterns captured in Seq2Seq match patterns in real sentence pairs)

### Directions

1. We can show that this "typed" decoder works better than LM at generating explanations? (Or better than SkipThought style generation.)
2. This can be used to answer generic why-questions by collecting large amounts of facts. (Prove by reformulating S1 to a question) (Tie this to question-answering)
3. Or somehow we can tie this to chatbot??? (Dialogue system is pretty cool now) (Use within-sentence discourse marker as pre-train objective for chichat dialogue system)

## Report

### Corpus Curation

**Wikipedia English**

96,330,860 sentences in total (as of Aug 2018)

After filtering, we obtained 649,952 sentences that contain `because`.

After parsing, we obtained 380,774 sentence pairs.

(Data is on Arthur2, OpenNMT is on Arthur1, Parsed result is on Cocoserv2)

Notes:

1. Dependency parsing becomes **significantly worse** when parsing Wikipedia sentences, which are less-formed, and more complex than factual/news story sentences.

Example:

| S1                                                           | S2                                  | marker  |
| ------------------------------------------------------------ | ----------------------------------- | ------- |
| When , i.e. , when the triangle is isosceles with the two sides incident to the angle equal , the law of cosines simplifies significantly . | Namely the law of cosines becomes . | because |

2. The type of `because` we get from Wikipedia still largely resembles what we got from the News Crawl. I need to read up more on this material.

| S1                                                           | S2                                                           | Type         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------ |
| Ants appear to be beneficial models .                        | They possess effective protective traits such as , chemical defences , and aggressiveness . | Explanation  |
| When , i.e. , when the triangle is isosceles with the two sides incident to the angle equal , the law of cosines simplifies significantly . | Namely the law of cosines becomes .                          | Causation(?) |
| The presence of this toxin in tampons may be linked to endometriosis ; they are chemically stable and can be absorbed by fat tissue , where they are then stored in the body . | Dioxins last a long time in the body .                       | Explanation  |
| Moving between Los Angeles and New York .                    | Her father was a journeyman writer and producer for TV .     | Causation(?) |

### PDP & Winograd Schema

PDP-60 is about pronoun resolution.

Winograd Scheme 273 is also similarly formed.

There are only 73 sentences in WSC-273 that contain `because`. We test if our system can generate good candidate sentences for these S1.

WSC has many discourse markers that we have encountered before.

| Marker                    | Number of instances |
| ------------------------- | ------------------- |
| because                   | 69                  |
| but                       | 53                  |
| so                        | 25                  |
| when                      | 18                  |
| if                        | 19                  |
| after                     | 11                  |
| although                  | 5                   |
| (Other - mostly sequence) | 73*                 |

*: number computed by simple subtraction from 273, should be higher.

There are sentences that contain multiple markers that form complex discourse. For example:

```
Sam and Amy are passionately in love , but Amy 's parents are unhappy about it , because Sam and Amy are fifteen .
```

So we combine `because` and `so` sentences to form our evaluation dataset. Maybe our decoder will do better on these! There are 94 tests from only `because` and `so` sentences.

```
"Susan knew that Ann 's son had been in a car accident , so Susan told her about it .",
 "Susan knew that Ann 's son had been in a car accident , because Ann told her about it ."
```

Here `so` is used in a sequence manner. Maybe we need to manually tag these 273 sentence pairs.

**S2 Generation**

Even though it's not super right, we can still generate candidate explanations for these sentences -- a bit murky because we didn't add `pointer-generator`. Maybe we should! Transformer also has `copy_attn` keyword! Pointer-generator can solve `<unk>` issue or `name` problem.

We pick sentences that are from `because` and `so`. After some simple processing to match the format, we get **62** sentences.

**Winograd and PDP Test**



