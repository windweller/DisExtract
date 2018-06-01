

import argparse
import sklearn.linear_model
from collections import Counter
from sys import exit

parser = argparse.ArgumentParser(description='Baselines')

parser.add_argument("--corpus", type=str, default='books_5', help="books_5|books_old_5|books_8|books_all|gw_5|gw_8")
parser.add_argument("--outputdir", type=str, default='sandbox/', help="Output directory")
parser.add_argument("--features", type=str, default="bow", help="bow|avg|mostcommonclass")
parser.add_argument("--ndims", type=int, default=1000)
parser.add_argument("--min_count", type=int, default=3)

params, _ = parser.parse_known_args()
print(params)

def read_corpus(corpus_label):
	corpus_dir = "/home/anie/DisExtract/data/books/"
	labels = {
		"books_all": "discourse_EN_ALL_and_then_because_though_still_after_when_while_but_also_as_so_although_before_if_2017dec21_",
		"books_8": "discourse_EN_EIGHT_and_but_because_if_when_before_so_though_2017dec18_",
		"books_5": "discourse_EN_FIVE_and_but_because_if_when_2017dec12_"
	}
	corpus = {}
	for split in ["train", "valid", "test"]:
		filename = corpus_dir + labels[corpus_label] + split + ".tsv"
		print("reading {}".format(filename))
		corpus[split] = {"s1": [], "s2": [], "label": []}
		for line in open(filename):
			s1, s2, label = line[:-1].split("\t")
			corpus[split]["s1"].append(s1)
			corpus[split]["s2"].append(s2)
			corpus[split]["label"].append(label)
	return corpus

def get_most_common_class_accuracies():
	corpus = read_corpus(params.corpus)
	train_counter = Counter(corpus["train"]["label"])
	print(train_counter)
	test_counter = Counter(corpus["test"]["label"])
	print(test_counter)
	most_common_train_class = train_counter.most_common()[0][0]
	n_correct = test_counter[most_common_train_class]
	n_total = sum([test_counter[label] for label in test_counter])
	print(float(n_correct)/n_total)

class BoWCorpus:
	def __init__(self, corpus_label):
		self.corpus = read_corpus(corpus_label)
		self.X = {split: [] for split in self.corpus}
		self.unigrams = None
		self.bigrams = None
		self.trigrams = None
	def featurize(self, split):
		if self.unigrams==None:
			print("featurizing corpus")
			self.unigrams = Counter()
			self.bigrams = Counter()
			self.trigrams = Counter()
			for split in self.corpus:
				for sentence_label in ["s1", "s2"]:
					for sentence in self.corpus[split][sentence_label]:
						unigrams = sentence.split(" ")
						bigrams = [" ".join(unigrams[(i-2):i]) for i in range(2, len(unigrams))]
						trigrams = [" ".join(unigrams[(i-3):i]) for i in range(3, len(unigrams))]
						self.unigrams.update(unigrams)
						self.bigrams.update(bigrams)
						self.trigrams.update(trigrams)
			self.unigrams = [unigram for unigram in self.unigrams.most_common() if self.unigrams[unigram]>params.min_count]
			self.bigrams = [bigram for bigram in self.bigrams.most_common() if self.bigrams[bigram]>params.min_count]
			self.trigrams = [trigram for trigram in self.trigrams.most_common() if self.trigrams[trigram]>params.min_count]
		if len(self.X[split])==0:
			print("featurizing {}".format(split))
			for i in range(len(self.corpus[split]["s1"])):
				s1 = self.corpus[split]["s1"][i]
				s2 = self.corpus[split]["s2"][i]
				s1_c = Counter(s1.split(" "))
				s2_c = Counter(s2.split(" "))
				s1_features = [s1_c[unigram] for unigram in self.unigrams] + \
							[s1_c[bigram] for bigram in self.bigrams] + \
							[s1_c[trigram] for trigram in self.trigrams]
				s2_features = [s2_c[unigram] for unigram in self.unigrams] + \
							[s2_c[bigram] for bigram in self.bigrams] + \
							[s2_c[trigram] for trigram in self.trigrams]
				feature_pair = s1_features + s2_features
				self.X[split].append(feature_pair)

		return (self.X[split], self.corpus[split]["label"])

def get_logistic_regression_performance():
	if params.features=="bow":
		corpus = BoWCorpus(params.corpus)
	train_X, train_y = corpus.featurize("train")
	test_X, test_y = corpus.featurize("test")

	model = sklearn.linear_model.LogisticRegression(max_iter=100, verbose=0)

	print("fitting model")
	model.fit(train_X, train_y)
	print("testing model")
	output = model.score(test_X, test_y)
	print(output)

if __name__ == '__main__':
	if params.features == "mostcommonclass":
		get_most_common_class_accuracies()
	get_logistic_regression_performance()


