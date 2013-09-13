#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

__all__ = []

import numpy as np
import scipy.optimize as op


def logsumexp(arr):
    a = np.max(arr)
    return a + np.log(np.sum(np.exp(arr - a)))


class LabeledInstance(object):

    def __init__(self, label, instance):
        self.label = label
        self.instance = instance


class FeatureExtractor(object):

    def __init__(self, features):
        self.features = list(set(features))
        self.nfeatures = len(self.features)

    def __call__(self, instance):
        f = np.zeros(self.nfeatures)
        for feature in instance.instance:
            f[self.features.index(feature)] += 1
        return f


class MaximumEntropyClassifier(object):

    def __init__(self, classes, sigma, iterations, extractor):
        self.classes = classes
        self.sigma = sigma
        self.iterations = iterations
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
        label_indicies = [self.classes.index(inst.label) for inst in data]
        feature_vector_list = [self.extractor(inst) for inst in data]
        results = op.minimize(self.objective, self.vector,
                              args=(label_indicies, feature_vector_list))
        self.vector = results.x

    def objective(self, weights, label_indicies, feature_vector_list):
        w = weights.reshape(self.wshape)

        lnprob = 0.0
        for i, f in zip(label_indicies, feature_vector_list):
            p = w.dot(f)
            norm = logsumexp(p)
            lnprob += p[i] - norm

        return -lnprob

    def get_probabilities(self, feature_vector):
        p = self.weights.dot(self.extractor(feature_vector))
        return dict(zip(self.classes, np.exp(p - logsumexp(p))))

if __name__ == "__main__":
    training_data = [LabeledInstance("cat", ["fuzzy", "claws", "small"]),
                     LabeledInstance("bear", ["fuzzy", "claws", "big"]),
                     LabeledInstance("cat", ["claws", "medium"])]
    test_datum = LabeledInstance("cat", ["claws", "small"])

    features = [f for datum in training_data for f in datum.instance]
    extractor = FeatureExtractor(features)

    classifier = MaximumEntropyClassifier(["cat", "bear"], 1.0, 20, extractor)
    classifier.train(training_data)

    print("Probabilities on test instance: {0}"
          .format(classifier.get_probabilities(test_datum)))
