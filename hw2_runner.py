#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import sys
import argparse
import numpy as np
from nlp import _maxent
from nlp.proper import Dataset
from nlp.maxent import (FeatureExtractor, UnigramExtractor,
                        MaximumEntropyClassifier)

np.random.seed(123)

parser = argparse.ArgumentParser(
    description="Proper noun classifier.")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Display all the results.")
parser.add_argument("-d", "--data", default="data",
                    help="The base path for the data files.")
parser.add_argument("--test-grad", action="store_true",
                    help="Test the gradient computation.")
parser.add_argument("--debug", action="store_true",
                    help="Use the debug dataset?")
parser.add_argument("-i", "--iterations", default=40, type=int,
                    help="The maximum number of optimizer iterations to run.")


if __name__ == "__main__":
    # Parse the command line arguments.
    args = parser.parse_args()

    if args.debug:
        training_data = [("cat", ["fuzzy", "claws", "small"]),
                         ("bear", ["fuzzy", "claws", "big"]),
                         ("cat", ["claws", "medium"])]
        validation_data = [("cat", ["claws", "small"])]
        test_data = [(None, ["claws", "small"])]

        features = [f for datum in training_data for f in datum[1]]
        extractor = FeatureExtractor(features)
        labels = ["bear", "cat"]
    else:
        # Load the datasets.
        training_data = Dataset(os.path.join(args.data, "pnp-train.txt"))
        validation_data = Dataset(os.path.join(args.data, "pnp-validate.txt"))
        test_data = Dataset(os.path.join(args.data, "pnp-test.txt"))

        # Figure out the list of features in the training data.
        features = [c for label, word in training_data for c in word]
        extractor = UnigramExtractor(features)
        labels = training_data.classes

    # Initialize the classifier.
    classifier = MaximumEntropyClassifier(labels, extractor)

    if args.test_grad:
        data = training_data
        label_indicies = [classifier.classes.index(inst[0]) for inst in data]
        feature_vector_list = [extractor(inst) for inst in data]
        classifier.vector = np.random.randn(len(classifier.vector))

        # Compute initial position and gradient.
        p0, g0 = _maxent.objective(classifier.vector, label_indicies,
                                   feature_vector_list, 1.0)

        v = np.array(classifier.vector)
        eps = 1e-8
        for i in range(len(classifier.vector)):
            v[i] += eps
            p1, g1 = _maxent.objective(v, label_indicies,
                                       feature_vector_list, 1.0)
            v[i] -= 2 * eps
            p2, g2 = _maxent.objective(v, label_indicies,
                                       feature_vector_list, 1.0)
            v[i] += eps

            print(g0[i], 0.5 * (p1 - p2) / eps)

        sys.exit(0)

    # Train.
    classifier.train(training_data, maxiter=args.iterations)

    # Test.
    print(classifier.test(validation_data))
    if args.debug:
        print(classifier.classes,
              np.exp(classifier.get_log_probabilities(test_data[0])))
