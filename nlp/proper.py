#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

__all__ = ["Distribution", "Dataset", "UnigramModel"]

import numpy as np
from collections import defaultdict


class Distribution(object):

    def __init__(self):
        self.norm = 0.0
        self.counts = defaultdict(float)

    def __getitem__(self, k):
        return self.counts.get(k, 0.0) / self.norm

    def incr(self, k, v=1.0):
        self.counts[k] += v
        self.norm += v

    def argmax(self):
        keys, values = zip(*(self.counts.items()))
        return keys[np.argmax(values)]


class Dataset(object):

    def __init__(self, fn):
        with open(fn) as f:
            self.data = [line.strip().split("\t")
                         if line[0] != "\t"
                         else ["", line.strip()]
                         for line in f]
        self.classes, tmp = zip(*self.data)
        self.classes = set(self.classes)

    def __getitem__(self, i):
        return self.data[i]


class UnigramModel(object):

    def __init__(self, training_set):
        self.prior = Distribution()
        self.distributions = dict([(c, Distribution())
                                   for c in training_set.classes])
        for c, w in training_set:
            self.prior.incr(c)
            [self.distributions[c].incr(l.lower()) for l in w if l != " "]

    def classify(self, word):
        prob = defaultdict(lambda: 1.0)
        for char in word.lower():
            if char == " ":
                continue
            for c, dist in self.distributions.items():
                prob[c] *= dist[char]

        classes, probabilities = zip(*(prob.items()))
        ind = np.argmax(probabilities)
        norm = np.sum(probabilities)
        if norm > 0.0:
            return classes[ind], probabilities[ind] / np.sum(probabilities)
        return self.prior.argmax(), 0.0

    def test(self, test_set, outfile=None):
        if outfile is not None:
            open(outfile, "w")
        success = 0
        total = 0
        for correct, word in test_set:
            guess, confidence = self.classify(word)
            if outfile is not None:
                with open(outfile, "a") as f:
                    f.write(("Example:\t{word}\tguess={guess}"
                             "\tgold={correct}\tconfidence={confidence}\n")
                            .format(word=word, guess=guess,
                                    confidence=confidence, correct=correct))
            if correct == guess:
                success += 1
            total += 1
        return success / total


class BigramModel(UnigramModel):

    pass


if __name__ == "__main__":
    training_set = Dataset("data/pnp-train.txt")
    validation_set = Dataset("data/pnp-validate.txt")
    test_set = Dataset("data/pnp-test.txt")

    model = UnigramModel(training_set)
    print("Validation accuracy: {0}".format(
        model.test(validation_set)))
    model.test(test_set, outfile="hw2/output.txt")
