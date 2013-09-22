#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import re
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
    axes[0].set_ylabel("negative log-likelihood")
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
                           r"gold=(\w*)\b")
    with open("results/{0}/validation.txt".format(model)) as f:
        for line in f:
            word, guess, gold = extractor.findall(
                line.decode("utf-8", errors="ignore"))[0]
            norm[gold] += 1
            matrix[gold][guess] += 1
            if gold != guess:
                print("{0} -> {1}: {2}".format(guess, gold, word))

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


if __name__ == "__main__":
    plot_confusion_matrix("unigram")
