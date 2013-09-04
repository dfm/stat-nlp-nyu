#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

__all__ = []

import os
import re
from math import log
from collections import defaultdict


STOP = "</S>"
UNKNOWN = "*UNKNOWN*"
word_re = re.compile(r"\s+")


def read_sentence(sentence):
    return [w.lower() for w in word_re.split(sentence) if len(w)]


def load_sentence_collection(fn):
    with open(fn) as f:
        collection = [read_sentence(line) for line in f]
    return collection


def load_nbest_list(basepath, vocabulary):
    # For fast lookups.
    vocab_set = set(vocabulary)

    # Read the correct sentences.
    correct_sentences = dict([(s[-1].strip("()"), s[:-1])
                              for s in load_sentence_collection(os.path.join(
                                  basepath, "REF.HUB1"))])

    # Loop over sentences and load the best N sentences and scores.
    nbest, tokenized = [], []
    for sid, correct_sentence in correct_sentences.items():
        sentences = load_sentence_collection(os.path.join(basepath, sid))
        with open(os.path.join(basepath, sid + ".acc")) as f:
            scores = [sum(map(float, line.split())) for line in f]

        # Discard sentences where any of the words are *not* in the vocabulary.
        try:
            sentences, scores = zip(*[(sentence, score)
                                      for sentence, score
                                      in zip(sentences, scores)
                                      if not any([w not in vocab_set
                                                  for w in sentence])])

        except ValueError:
            # No sentences had only vocabulary words.
            continue

        # FIXME: in the current dataset all the sentences are unique.
        # unique_sentences = set([" ".join(s) for s in sentences])
        # assert len(sentences) == len(unique_sentences)

        # Find the tokenized sentence and skip if it doesn't exist.
        try:
            tokenized.append(sentences[map("".join, sentences)
                                       .index("".join(correct_sentence))])
        except ValueError:
            continue

        nbest.append(dict([(" ".join(s), sc)
                     for s, sc in zip(sentences, scores)]))

    return tokenized, nbest


class WordCounter(object):

    def __init__(self):
        self.total = 0.0
        self.counts = defaultdict(float)

    def incr(self, w, n=1.0):
        self.total += n
        self.counts[w] += n

    def normalize(self):
        self.counts = dict([(k, v / self.total)
                            for k, v in self.counts.items()])
        self.total = 1.0

    def words(self):
        return self.counts.keys()

    def get(self, word):
        return self.counts.get(word, self.counts.get(UNKNOWN))


class LanguageModel(object):

    def __init__(self, sentence_collection):
        self.sentence_collection = sentence_collection

        # Count the unigrams.
        self.unigrams = WordCounter()
        [self.unigrams.incr(w) for s in sentence_collection for w in s]
        self.unigrams.incr(STOP, len(sentence_collection))
        self.unigrams.incr(UNKNOWN)
        self.unigrams.normalize()

        # Save the vocabulary list.
        self.vocabulary = self.unigrams.words()

    def get_word_lnprobability(self, word):
        return log(self.unigrams.get(word))

    def get_sentence_lnprobability(self, sentence):
        return sum(map(self.get_word_lnprobability, sentence + [STOP]))

    def get_perplexity(self, sentence_collection):
        lp = sum(map(self.get_sentence_lnprobability, sentence_collection))
        lp /= log(2.0)
        ns = sum(map(len, sentence_collection))
        return 0.5 ** (lp / ns)


if __name__ == "__main__":
    train_collection = load_sentence_collection(
        "data/treebank-sentences-spoken-train.txt")
    validation_collection = load_sentence_collection(
        "data/treebank-sentences-spoken-validate.txt")
    test_collection = load_sentence_collection(
        "data/treebank-sentences-spoken-test.txt")

    model = LanguageModel(train_collection)
    nbest, scores = load_nbest_list("data/wsj_n_bst", model.vocabulary)
    print(len(nbest))

    print(model.get_perplexity(test_collection))
    print(model.get_perplexity(nbest))
