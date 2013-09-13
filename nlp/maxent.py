#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

__all__ = []

import numpy as np
import scipy.optimize as op

from . import _maxent


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

    def __init__(self, features):
        super(UnigramExtractor, self).__init__(features)
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

    def train(self, data):
        label_indicies = [self.classes.index(inst[0]) for inst in data]
        feature_vector_list = [self.extractor(inst) for inst in data]
        results = op.minimize(_maxent.objective, self.vector, jac=True,
                              args=(label_indicies, feature_vector_list,
                                    self.sigma))
        print(results)
        self.vector = results.x

    def test(self, data):
        correct = 0
        total = 0
        for instance in data:
            p = self.get_log_probabilities(instance)
            if np.argmax(p) == self.classes.index(instance[0]):
                correct += 1
            total += 1
        return correct / total

    def get_log_probabilities(self, feature_vector):
        p = self.weights.dot(self.extractor(feature_vector))
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
