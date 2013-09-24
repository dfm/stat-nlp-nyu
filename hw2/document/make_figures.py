#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import re
import sys
import numpy as np
import matplotlib.pyplot as pl
from collections import defaultdict
from matplotlib.ticker import MaxNLocator


def plot_convergence(model):
    data = np.loadtxt("results/{0}/convergence.txt".format(model))

    fig, axes = pl.subplots(2, 1)
    fig.subplots_adjust(left=0.15, bottom=0.1, right=0.95, top=0.95,
                        wspace=0.0, hspace=0.0)

    axes[0].plot(data[:, 0], data[:, 1], "k", lw=2)
    axes[0].annotate("training set", xy=(data[-1, 0], data[-1, 1]),
                     xytext=(-5, 5), textcoords="offset points",
                     ha="right", va="bottom", xycoords="data")
    axes[0].set_xticklabels([])
    axes[0].set_ylabel("negative log-probability")
    axes[0].yaxis.set_major_locator(MaxNLocator(5))
    axes[0].yaxis.set_label_coords(-0.1, 0.5)

    axes[1].plot(data[:, 0], data[:, 2]*100, "k", lw=2)
    axes[1].plot(data[:, 0], data[:, 3]*100, "--k", lw=2)
    axes[1].annotate("training set", xy=(data[-1, 0], 100*data[-1, 2]),
                     xytext=(-5, -5), textcoords="offset points",
                     ha="right", va="top", xycoords="data")
    axes[1].annotate("validation set", xy=(data[-1, 0], 100*data[-1, 3]),
                     xytext=(-5, -5), textcoords="offset points",
                     ha="right", va="top", xycoords="data")

    axes[1].set_ylabel("percent accuracy")
    axes[1].set_xlabel("L-BFGS iterations")
    axes[1].yaxis.set_major_locator(MaxNLocator(5))
    axes[1].yaxis.set_label_coords(-0.1, 0.5)

    fig.savefig("{0}_convergence.pdf".format(model))


def plot_confusion_matrix(model):
    norm = defaultdict(int)
    matrix = defaultdict(lambda: defaultdict(int))
    extractor = re.compile(r"Example:(?:\s*)(.*?)(?:\s*?)guess=(\w*?)(?:\s*?)"
                           r"gold=(\w*)(?:\s*)confidence=(.*)")
    correct_confidences = []
    incorrect_confidences = []
    open("results/{0}/mistakes.txt".format(model), "w")
    with open("results/{0}/validation.txt".format(model)) as f:
        for line in f:
            word, guess, gold, confidence = extractor.findall(
                line.decode("utf-8", errors="ignore"))[0]
            norm[gold] += 1
            matrix[gold][guess] += 1
            if gold != guess:
                incorrect_confidences.append(float(confidence.strip()))
                with open("results/{0}/mistakes.txt".format(model),
                          "a") as fout:
                    fout.write("{0} -> {1}: {2} ({3})\n"
                               .format(guess, gold, word, confidence.strip()))
            else:
                correct_confidences.append(float(confidence.strip()))

    keys = norm.keys()
    conf = np.zeros((len(keys), len(keys)))
    for i, gold in enumerate(keys):
        for j, guess in enumerate(keys):
            conf[i, j] = matrix[gold][guess] / norm[gold]

    fig = pl.figure(figsize=(8, 8))
    fig.subplots_adjust(left=0.2, bottom=0.2, right=0.99, top=0.99,
                        wspace=0.0, hspace=0.0)

    ax = pl.gca()
    ax.imshow(np.max(conf) - conf, cmap="gray", interpolation="nearest")

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys)
    [l.set_rotation(90) for l in ax.get_xticklabels()]
    ax.set_xlabel("guess", fontsize=36)

    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys)
    ax.set_ylabel("gold", fontsize=36)

    pl.savefig("{0}_confusion.pdf".format(model))

    # Plot histograms.
    pl.figure()
    bins = np.linspace(0, 1, 50)
    nc, b, p = pl.hist(correct_confidences, bins, histtype="step", color="g")
    ni, b, p = pl.hist(incorrect_confidences, bins, histtype="step", color="r")
    pl.xlabel("confidence")
    pl.ylabel("number of examples")
    pl.savefig("{0}_confidence_hist.pdf".format(model))

    pl.clf()
    pl.plot(0.5*(bins[1:]+bins[:-1]), nc / (nc + ni), ".k")
    pl.plot([0, 1], [0, 1], "--k", lw=2)
    pl.xlabel("confidence")
    pl.ylabel("fraction correct")
    pl.savefig("{0}_confidence_scale.pdf".format(model))


if __name__ == "__main__":
    model = sys.argv[1]
    plot_convergence(model)
    plot_confusion_matrix(model)
