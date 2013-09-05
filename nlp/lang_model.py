#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

__all__ = []

import os
import re
import numpy as np
from math import log
from collections import defaultdict


INSERT_COST = 1.0
DELETE_COST = 1.0
SUBSTITUTE_COST = 1.0

STOP = "</S>"
UNKNOWN = "*UNKNOWN*"

word_re = re.compile(r"\s+")


def read_sentence(sentence):
    return [w.lower() for w in word_re.split(sentence) if len(w)]


def load_sentence_collection(fn):
    with open(fn) as f:
        collection = [read_sentence(line) for line in f]
    return collection


def _get_distance(first_list, second_list, first_pos, second_pos, best_dists):
    if first_pos > len(first_list) or second_pos > len(second_list):
        return np.inf
    if first_pos == len(first_list) and second_pos == len(second_list):
        return 0.0

    if np.isnan(best_dists[first_pos][second_pos]):
        dist = np.inf
        dist = np.min([dist, INSERT_COST + _get_distance(first_list,
                                                         second_list,
                                                         first_pos+1,
                                                         second_pos,
                                                         best_dists)])
        dist = np.min([dist, DELETE_COST + _get_distance(first_list,
                                                         second_list,
                                                         first_pos,
                                                         second_pos+1,
                                                         best_dists)])
        dist = np.min([dist, SUBSTITUTE_COST + _get_distance(first_list,
                                                             second_list,
                                                             first_pos+1,
                                                             second_pos+1,
                                                             best_dists)])
        if (first_pos < len(first_list) and second_pos < len(second_list) and
                first_list[first_pos] == second_list[second_pos]):
            dist = np.min([dist, _get_distance(first_list, second_list,
                                               first_pos+1, second_pos+1,
                                               best_dists)])
        best_dists[first_pos][second_pos] = dist

    return best_dists[first_pos][second_pos]


def get_edit_distance(list1, list2):
    return _get_distance(list1, list2, 0, 0, np.nan + np.zeros((len(list1)+1,
                                                                len(list2)+1)))


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

    def get_word_error_rate(self, nbest, verbose=True):
        total_distance = 0.0
        total_words = 0.0
        for correct, guesses in zip(nbest.correct, nbest.guesses):
            print("CORRECT: " + " ".join(correct))
            best_guess = None
            best_score = -np.inf
            nwbest = 0.0
            dwbest = 0.0
            for guess, ac_score in guesses:
                score = self.get_sentence_lnprobability(guess) + ac_score/16.0
                distance = get_edit_distance(correct, guess)
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
                self.display_hypothesis("GUESS", best_guess, nbest)
                self.display_hypothesis("GOLD", correct, nbest)

        return total_distance / total_words

    def display_hypothesis(self, title, guess, nbest):
        pass


if __name__ == "__main__":
    train_collection = load_sentence_collection(
        "data/treebank-sentences-spoken-train.txt")
    validation_collection = load_sentence_collection(
        "data/treebank-sentences-spoken-validate.txt")
    test_collection = load_sentence_collection(
        "data/treebank-sentences-spoken-test.txt")

    model = LanguageModel(train_collection)
    nbest = NBestList("data/wsj_n_bst", model.vocabulary)

    print("WSJ Perplexity: {0}".format(model.get_perplexity(test_collection)))
    print("HUB Perplexity: {0}".format(model.get_perplexity(nbest.correct)))
    wer = model.get_word_error_rate(nbest)
    print("HUB Word Error Rate: {0}".format(wer))
