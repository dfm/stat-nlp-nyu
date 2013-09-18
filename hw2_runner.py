#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import argparse
import numpy as np
from nlp.proper import Dataset
from nlp.maxent import (FeatureExtractor, BigramExtractor,
                        MaximumEntropyClassifier)

np.random.seed(123)

parser = argparse.ArgumentParser(
    description="Proper noun classifier.")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Display all the results.")
parser.add_argument("-d", "--data", default="data",
                    help="The base path for the data files.")
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
        extractors = [FeatureExtractor(features)]
        labels = ["bear", "cat"]

    else:
        # Load the datasets.
        training_data = Dataset(os.path.join(args.data, "pnp-train.txt"))
        validation_data = Dataset(os.path.join(args.data, "pnp-validate.txt"))
        test_data = Dataset(os.path.join(args.data, "pnp-test.txt"))

        extractors = [BigramExtractor(training_data)]
        labels = training_data.classes

    # Initialize the classifier.
    classifier = MaximumEntropyClassifier(labels, extractors)

    # Train.
    classifier.train(training_data, maxiter=args.iterations)

    # Test.
    print("Validation accuracy: {0}".format(classifier.test(validation_data)))
    classifier.test(test_data, outfile="hw2/output.txt")

    if args.debug:
        print(classifier.classes,
              np.exp(classifier.get_log_probabilities(test_data[0])))
