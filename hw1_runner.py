#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import sys
import argparse
from nlp.lang_model import NBestList, load_sentence_collection

parser = argparse.ArgumentParser(
    description="Run language model for homework 1.")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Display all the results.")
parser.add_argument("-m", "--model", default="LanguageModel",
                    help="The name of the Python class that implements the "
                    "language model.")
parser.add_argument("-d", "--data", default="data",
                    help="The base path for the data files.")


if __name__ == "__main__":
    # Parse the command line arguments.
    args = parser.parse_args()

    # Import the requested model.
    exec("from nlp.lang_model import {0}".format(args.model))

    # Load the WSJ data.
    train_collection = load_sentence_collection(
        os.path.join(args.data, "treebank-sentences-spoken-train.txt"))
    validation_collection = load_sentence_collection(
        os.path.join(args.data, "treebank-sentences-spoken-validate.txt"))
    test_collection = load_sentence_collection(
        os.path.join(args.data, "treebank-sentences-spoken-test.txt"))

    # Train the model.
    model = eval("{0}(train_collection)".format(args.model))

    # Load the HUB data.
    nbest = NBestList(os.path.join(args.data, "wsj_n_bst"), model.vocabulary)

    # Print the results.
    print("WSJ Perplexity: {0}".format(model.get_perplexity(test_collection)))
    print("HUB Perplexity: {0}".format(model.get_perplexity(nbest.correct)))
    wer = model.get_word_error_rate(nbest, verbose=args.verbose)
    print("HUB Word Error Rate: {0}".format(wer))
