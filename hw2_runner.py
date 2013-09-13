#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import sys
import argparse
from nlp.proper import Dataset
from nlp.maxent import UnigramExtractor, MaximumEntropyClassifier
from nlp import _maxent

parser = argparse.ArgumentParser(
    description="Proper noun classifier.")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Display all the results.")
parser.add_argument("-d", "--data", default="data",
                    help="The base path for the data files.")
parser.add_argument("--test-grad", action="store_true",
                    help="Test the gradient computation.")


if __name__ == "__main__":
    # Parse the command line arguments.
    args = parser.parse_args()

    # Load the datasets.
    training_data = Dataset(os.path.join(args.data, "pnp-train.txt"))
    validation_data = Dataset(os.path.join(args.data, "pnp-validate.txt"))
    test_data = Dataset(os.path.join(args.data, "pnp-test.txt"))

    # Figure out the list of features in the training data.
    features = [c for label, word in training_data for c in word]
    extractor = UnigramExtractor(features)

    # Initialize the classifier.
    classifier = MaximumEntropyClassifier(training_data.classes, extractor)

    if args.test_grad:
        print(_maxent.objective)
        sys.exit(0)

    # Train.
    classifier.train(training_data)

    # Test.
    print(classifier.test(validation_data))
