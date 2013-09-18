#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

__all__ = ["FeatureExtractor", "UnigramExtractor", "BigramExtractor",
           "SuffixExtractor", "MaximumEntropyClassifier",
           "DigitExtractor"]

import re
import numpy as np

from . import _maxent


START = "|"
STOP = "\\"


def logsumexp(arr):
    a = np.max(arr)
    return a + np.log(np.sum(np.exp(arr - a)))


class FeatureExtractor(object):

    def setup(self, features):
        self.features = list(set(features))
        self.nfeatures = len(self.features)

    def __call__(self, instance):
        f = np.zeros(self.nfeatures)
        for feature in instance[1]:
            f[self.features.index(feature)] += 1
        return f


class UnigramExtractor(FeatureExtractor):

    def setup(self, training_data):
        # Find all the character unigrams.
        unigrams = [c for label, word in training_data for c in word]
        super(UnigramExtractor, self).setup(unigrams)

        # Allow for unknown characters.
        self.nfeatures += 1

    def __call__(self, instance):
        f = np.zeros(self.nfeatures)
        for char in instance[1]:
            try:
                f[self.features.index(char)] += 1
            except ValueError:
                # Unknown character.
                f[-1] += 1
        return f


class BigramExtractor(FeatureExtractor):

    def setup(self, training_data):
        # Then, find the bigrams.
        bigrams = []
        for label, w in training_data:
            word = START + w + STOP
            bigrams += [c+word[i+1] for i, c in enumerate(word[:-1])]

        # Initialize the extractor.
        super(BigramExtractor, self).setup(bigrams)

    def __call__(self, instance):
        f = np.zeros(self.nfeatures)
        word = START + instance[1] + STOP
        for i, char in enumerate(word[:-1]):
            try:
                f[self.features.index(char+word[i+1])] += 1
            except ValueError:
                # Unknown character.
                pass
        return f


class SuffixExtractor(FeatureExtractor):

    def __init__(self, length):
        self.length = length

    def setup(self, training_data):
        suffs = [w[-self.length:] for label, w in training_data]
        super(SuffixExtractor, self).setup(suffs)

    def __call__(self, instance):
        f = np.zeros(self.nfeatures)
        try:
            f[self.features.index(instance[1][-self.length:])] = 1
        except ValueError:
            pass
        return f


class DigitExtractor(FeatureExtractor):

    _re = re.compile("[0-9]")

    def __init__(self, number=10):
        super(DigitExtractor, self).setup(range(number))

    def setup(self, training_data):
        pass

    def __call__(self, inst):
        f = np.zeros(self.nfeatures)
        try:
            f[len(self._re.findall(inst[1]))] = 1
        except IndexError:
            f[-1] = 1
        return f


class MaximumEntropyClassifier(object):

    def __init__(self, classes, extractors, sigma=1.0):
        self.classes = list(classes)
        self.sigma = sigma
        self._hinvsig2 = 0.5 / sigma / sigma
        self.extractors = extractors
        self.wshape = (len(classes), sum([e.nfeatures for e in extractors]))
        self.weights = np.zeros(self.wshape)

    @property
    def vector(self):
        return self.weights.flatten()

    @vector.setter
    def vector(self, v):
        self.weights = v.reshape(self.wshape)

    def extract(self, inst):
        return np.concatenate([e(inst) for e in self.extractors])

    def train(self, data, maxiter=40):
        label_indicies = [self.classes.index(inst[0]) for inst in data]
        feature_vector_list = [self.extract(inst) for inst in data]
        nlp, self.vector = _maxent.optimize(self.vector, label_indicies,
                                            feature_vector_list, self.sigma,
                                            maxiter)

    def online(self, data, maxiter=40, rate=0.1):
        np.random.suffle(data)
        label_indicies = [self.classes.index(inst[0]) for inst in data]
        feature_vector_list = [self.extract(inst) for inst in data]

        v = self.vector
        _maxent.online(v, label_indicies, feature_vector_list, self.sigma,
                       maxiter, rate)
        self.vector = v

    def test(self, test_set, outfile=None):
        if outfile is not None:
            open(outfile, "w")

        success = 0
        total = 0
        for correct, word in test_set:
            p = self.get_log_probabilities(word)
            ind = np.argmax(p)
            guess = self.classes[ind]
            if outfile is not None:
                with open(outfile, "a") as f:
                    f.write(("Example:\t{word}\tguess={guess}"
                             "\tgold={correct}\tconfidence={confidence}\n")
                            .format(word=word, guess=guess,
                                    confidence=np.exp(p[ind]),
                                    correct=correct))
            if correct == guess:
                success += 1
            total += 1
        return success / total

    def get_log_probabilities(self, feature_vector):
        p = self.weights.dot(self.extract((None, feature_vector)))
        return p - logsumexp(p)

if __name__ == "__main__":
    training_data = [("cat", ["fuzzy", "claws", "small"]),
                     ("bear", ["fuzzy", "claws", "big"]),
                     ("cat", ["claws", "medium"])]
    test_datum = ("cat", ["claws", "small"])

    features = [f for datum in training_data for f in datum[1]]
    extractor = FeatureExtractor(features)

    classifier = MaximumEntropyClassifier(["cat", "bear"], [extractor])
    classifier.train(training_data)
    print(classifier.test([test_datum]))
    print(classifier.classes,
          np.exp(classifier.get_log_probabilities(test_datum)))
