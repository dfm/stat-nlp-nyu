#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

__all__ = []

import numpy as np

from . import _maxent


START = "|"
STOP = "\\"


def logsumexp(arr):
    a = np.max(arr)
    return a + np.log(np.sum(np.exp(arr - a)))


class FeatureExtractor(object):

    def __init__(self, features):
        self.features = list(set(features))
        self.nfeatures = len(self.features)

    def __call__(self, instance):
        f = np.zeros(self.nfeatures)
        for feature in instance[1]:
            f[self.features.index(feature)] += 1
        return f


class UnigramExtractor(FeatureExtractor):

    def __init__(self, unigrams):
        super(UnigramExtractor, self).__init__(unigrams)
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


class BigramExtractor(UnigramExtractor):

    def __init__(self, unigrams, bigrams):
        super(BigramExtractor, self).__init__(unigrams)
        self.features += list(set(bigrams))
        self.nfeatures = len(self.features) + 1

    def __call__(self, instance):
        f = super(BigramExtractor, self).__call__(instance)
        word = START + instance[1] + STOP
        for i, char in enumerate(word[:-1]):
            try:
                f[self.features.index(char+word[i+1])] += 1
            except ValueError:
                # Unknown character.
                pass
        return f


class MaximumEntropyClassifier(object):

    def __init__(self, classes, extractor, sigma=1.0):
        self.classes = list(classes)
        self.sigma = sigma
        self._hinvsig2 = 0.5 / sigma / sigma
        self.extractor = extractor
        self.wshape = (len(classes), extractor.nfeatures)
        self.weights = np.zeros(self.wshape)

    @property
    def vector(self):
        return self.weights.flatten()

    @vector.setter
    def vector(self, v):
        self.weights = v.reshape(self.wshape)

    def train(self, data, maxiter=40):
        label_indicies = [self.classes.index(inst[0]) for inst in data]
        feature_vector_list = [self.extractor(inst) for inst in data]
        nlp, self.vector = _maxent.optimize(self.vector, label_indicies,
                                            feature_vector_list, self.sigma,
                                            maxiter)

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
        p = self.weights.dot(self.extractor((None, feature_vector)))
        return p - logsumexp(p)

if __name__ == "__main__":
    training_data = [("cat", ["fuzzy", "claws", "small"]),
                     ("bear", ["fuzzy", "claws", "big"]),
                     ("cat", ["claws", "medium"])]
    test_datum = ("cat", ["claws", "small"])

    features = [f for datum in training_data for f in datum[1]]
    extractor = FeatureExtractor(features)

    classifier = MaximumEntropyClassifier(["cat", "bear"], extractor)
    classifier.train(training_data)
    print(classifier.test([test_datum]))
    print(classifier.classes,
          np.exp(classifier.get_log_probabilities(test_datum)))
