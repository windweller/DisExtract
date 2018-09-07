# -*- coding: utf-8 -*-

import spacy
import logging
import string

en_nlp = spacy.load("en_core_web_sm")
printable = set(string.printable)
file_names = ['news.{}.en.shuffled'.format(y) for y in range(2007, 2017+1)]


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    sentences = []

    for f in file_names:
        logger.info(f)
        with open(f) as f:
            for i, line in enumerate(f):
                # Tokenize the paragraph into words
		processed = filter(lambda x: x in printable, line).strip()
                tokens = en_nlp.tokenizer(unicode(processed))
                words = [str(token) for token in tokens if not str(token).isspace()]
                if len(words) < 3:
                    continue
                sentences.append(' '.join(words))
                if i % 10000 == 0:
                    logger.info("processed {} sentences".format(i))

        with open("./news_crawl_0717_flattened.txt", 'w') as f:
            for s in sentences:
                f.write(s + "\n")
