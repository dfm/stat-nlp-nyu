#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import sys
import argparse
import numpy as np
import cPickle as pickle
from nlp.proper import Dataset
from nlp.maxent import *

np.random.seed(123)

parser = argparse.ArgumentParser(
    description="Proper noun classifier.")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Display all the results.")
parser.add_argument("--online", action="store_true",
                    help="Run an online algorithm.")
parser.add_argument("-r", "--rate", default=0.1, type=float,
                    help="The learning rate.")
parser.add_argument("-c", "--const", default=1, type=float,
                    help="The learning rate decay constant.")
parser.add_argument("-d", "--data", default="data",
                    help="The base path for the data files.")
parser.add_argument("-s", "--sigma", default=1.0, type=float,
                    help="The L2 coefficient.")
parser.add_argument("--debug", action="store_true",
                    help="Use the debug dataset?")
parser.add_argument("-i", "--iterations", default="25, 50, 100, 150, 200",
                    help="The schedule of optimizer iterations to run.")
parser.add_argument("-e", "--extractors", default="UnigramExtractor()",
                    help="A comma separated list of extractors to use.")
parser.add_argument("-o", "--outdir", default="hw2",
                    help="The output directory to save to.")


if __name__ == "__main__":
    # Parse the command line arguments.
    args = parser.parse_args()

    try:
        os.makedirs(args.outdir)
    except os.error:
        pass

    with open(os.path.join(args.outdir, "args"), "w") as fl:
        fl.write(" ".join(sys.argv))

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
        labels = training_data.classes

        print("Number of parameters:")
        extractors = []
        for e in args.extractors.split("|"):
            extractors.append(eval("{0}".format(e.strip())))
            extractors[-1].setup(training_data)
            print("    {0}: {1}".format(e.strip(), extractors[-1].nfeatures))

    # Initialize the classifier.
    classifier = MaximumEntropyClassifier(labels, extractors, sigma=args.sigma)

    # Train.
    iterations = map(int, args.iterations.split(","))
    if args.online:
        classifier.online(training_data, maxiter=max(iterations),
                          rate=args.rate, C=args.const)
    else:
        classifier.train(training_data, validation_set=validation_data,
                         outdir=args.outdir,
                         schedule=iterations)

    # Save the final classifier.
    pickle.dump(classifier, open(os.path.join(args.outdir, "model.pkl"), "wb"),
                -1)

    # Test.
    print("Validation accuracy: {0}".format(
        classifier.test(validation_data,
                        outfile=os.path.join(args.outdir, "validation.txt"))))
    classifier.test(test_data,
                    outfile=os.path.join(args.outdir, "test.txt"))

    if args.debug:
        print(classifier.classes,
              np.exp(classifier.get_log_probabilities(test_data[0])))
