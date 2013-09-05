#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import re
import numpy as np
from math import log
from collections import defaultdict

import _edit


INSERT_COST = 1.0
DELETE_COST = 1.0
SUBSTITUTE_COST = 1.0

START = "<S>"
STOP = "</S>"
UNKNOWN = "*UNKNOWN*"

word_re = re.compile(r"\s+")


def read_sentence(sentence):
    return [w.lower() for w in word_re.split(sentence) if len(w)]


def load_sentence_collection(fn):
    with open(fn) as f:
        collection = [read_sentence(line) for line in f]
    return collection


class NBestList(object):

    def __init__(self, basepath, vocabulary):
        # For fast lookups.
        vocab_set = set(vocabulary)

        # Read the correct sentences.
        correct_sentences = dict([(s[-1].strip("()"), s[:-1])
                                  for s in load_sentence_collection(
                                      os.path.join(basepath, "REF.HUB1"))])

        # Loop over sentences and load the best N sentences and scores.
        nbest, tokenized = [], []
        for sid, correct_sentence in correct_sentences.items():
            sentences = load_sentence_collection(os.path.join(basepath, sid))
            with open(os.path.join(basepath, sid + ".acc")) as f:
                scores = [sum(map(float, line.split())) for line in f]

            # Discard sentences where any of the words are *not* in the
            # vocabulary.
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

            nbest.append(zip(sentences, scores))

        self.correct = tokenized
        self.guesses = nbest


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
        return self.counts.get(word, self.counts.get(UNKNOWN, 0.0))


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

    def get_word_error_rate(self, nbest, verbose=True):
        total_distance = 0.0
        total_words = 0.0
        for correct, guesses in zip(nbest.correct, nbest.guesses):
            best_guess = None
            best_score = -np.inf
            nwbest = 0.0
            dwbest = 0.0
            for guess, ac_score in guesses:
                score = self.get_sentence_lnprobability(guess) + ac_score/16.0
                distance = _edit.distance(correct, guess)
                if score == best_score:
                    nwbest += 1
                    dwbest += distance
                if score > best_score or best_guess is None:
                    best_score = score
                    best_guess = guess
                    dwbest = distance
                    nwbest = 1.0
            total_distance += dwbest / nwbest
            total_words += len(correct)
            if verbose:
                print()
                self.display_hypothesis("GUESS", best_guess, guesses)
                self.display_hypothesis("GOLD", correct, guesses)

        return total_distance / total_words

    def display_hypothesis(self, title, guess, guesses):
        for g, ac_score in guesses:
            if g == guess:
                lnprob = self.get_sentence_lnprobability(guess)
                print("{0}:\tAM: {1:.2e}\tLM: {2:.2e}\tTotal: {3:.2e}\t{4}"
                      .format(title, ac_score/16., lnprob,
                              ac_score/16.+lnprob, guess))
                break


class BigramModel(LanguageModel):

    def __init__(self, sentence_collection, factor=0.6):
        super(BigramModel, self).__init__(sentence_collection)
        self.factor = factor

        # Compute the empirical bigram frequencies.
        self.bigrams = defaultdict(WordCounter)
        [self.bigrams[w].incr(s[i])
         if i < len(s) else self.bigrams[w].incr(STOP)
         for s in sentence_collection
         for i, w in enumerate([START] + s)]
        [bg.normalize() for bg in self.bigrams.values()]

    def get_bigram_lnprobability(self, prev, word):
        ug = self.unigrams.get(word)
        bg = self.bigrams[prev]
        return log((1 - self.factor) * ug + self.factor*bg.get(word))

    def get_sentence_lnprobability(self, sentence):
        prev = START
        lnprob = 0.0
        for word in sentence + [STOP]:
            lnprob += self.get_bigram_lnprobability(prev, word)
            prev = word
        return lnprob


if __name__ == "__main__":
    import sys

    train_collection = load_sentence_collection(
        "data/treebank-sentences-spoken-train.txt")
    validation_collection = load_sentence_collection(
        "data/treebank-sentences-spoken-validate.txt")
    test_collection = load_sentence_collection(
        "data/treebank-sentences-spoken-test.txt")

    model = BigramModel(train_collection)
    # model = LanguageModel(train_collection)
    nbest = NBestList("data/wsj_n_bst", model.vocabulary)

    print("WSJ Perplexity: {0}".format(model.get_perplexity(test_collection)))
    print("HUB Perplexity: {0}".format(model.get_perplexity(nbest.correct)))
    wer = model.get_word_error_rate(nbest, verbose=("--verbose" in sys.argv))
    print("HUB Word Error Rate: {0}".format(wer))
