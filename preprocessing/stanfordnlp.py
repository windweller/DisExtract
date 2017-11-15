"""
Add StanfordNLP code connection
(this is a testing file)
"""

from stanfordcorenlp import StanfordCoreNLP


nlp = StanfordCoreNLP(r'/Users/Aimingnie/Documents/School/Stanford/CS 224N/stanford-corenlp-full-2017-06-09')

sentence = 'Guangdong University of Foreign Studies is located in Guangzhou.'
print 'Tokenize:', nlp.word_tokenize(sentence)
print 'Part of Speech:', nlp.pos_tag(sentence)
print 'Named Entities:', nlp.ner(sentence)
print 'Constituency Parsing:', nlp.parse(sentence)
print 'Dependency Parsing:', nlp.dependency_parse(sentence)

props={'annotators': 'tokenize,ssplit,pos,dep','pipelineLanguage':'en','outputFormat':'json'}
print nlp.annotate(sentence, properties=props)