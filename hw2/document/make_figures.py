#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator

data = np.loadtxt("results/full_500/convergence.txt")

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

fig.savefig("full_500_convergence.pdf")
