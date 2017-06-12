
import logging
import glob
import codecs
import tensorflow as tf
import nltk
import re


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


books = sorted(glob.glob("/home/atul/PycharmProjects/word2vec/Word2VecModel/*.txt"))

print("Books : ", books)

corpus_raw = u""

for book in books:
    print("Reading book '{0}'...".format(book))
    with codecs.open(book, "r", "utf-8") as book_body:
        corpus_raw += book_body.read()
    print("Corpus now contains {0} charecters".format(len(corpus_raw)))


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

raw_sentences = tokenizer.tokenize(corpus_raw)

def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words

