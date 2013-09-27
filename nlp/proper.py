#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

__all__ = ["Distribution", "Dataset", "UnigramModel"]

import numpy as np
from collections import defaultdict


START = "|"
STOP = "\\"


class Distribution(object):

    def __init__(self):
        self.norm = 0.0
        self.counts = defaultdict(float)

    def __getitem__(self, k):
        return self.counts.get(k, 0.0) / self.norm

    def incr(self, k, v=1.0):
        self.counts[k] += v
        self.norm += v

    def argmax(self):
        keys, values = zip(*(self.counts.items()))
        return keys[np.argmax(values)]


class Dataset(object):

    def __init__(self, fn):
        with open(fn) as f:
            self.data = [line.strip().split("\t")
                         if line[0] != "\t"
                         else ["", line.strip()]
                         for line in f]
        self.classes, tmp = zip(*self.data)
        self.classes = set(self.classes)

    def __getitem__(self, i):
        return self.data[i]


class UnigramModel(object):

    def __init__(self, training_set):
        self.prior = Distribution()
        self.distributions = dict([(c, Distribution())
                                   for c in training_set.classes])
        for c, w in training_set:
            self.prior.incr(c)
            [self.distributions[c].incr(l.lower()) for l in w if l != " "]

    def get_probabilities(self, word):
        prob = defaultdict(lambda: 1.0)
        for char in word.lower():
            if char == " ":
                continue
            for c, dist in self.distributions.items():
                prob[c] *= dist[char]
        return prob

    def classify(self, word):
        prob = self.get_probabilities(word)
        classes, probabilities = zip(*(prob.items()))
        probabilities = [self.prior[k] * p
                         for k, p in zip(classes, probabilities)]
        ind = np.argmax(probabilities)
        norm = np.sum(probabilities)
        if norm > 0.0:
            return classes[ind], probabilities[ind] / np.sum(probabilities)
        return self.prior.argmax(), 0.0

    def test(self, test_set, outfile=None):
        if outfile is not None:
            open(outfile, "w")
        success = 0
        total = 0
        for correct, word in test_set:
            guess, confidence = self.classify(word)
            if outfile is not None:
                with open(outfile, "a") as f:
                    f.write(("Example:\t{word}\tguess={guess}"
                             "\tgold={correct}\tconfidence={confidence}\n")
                            .format(word=word, guess=guess,
                                    confidence=confidence, correct=correct))
            if correct == guess:
                success += 1
            total += 1
        return success / total


class BigramModel(UnigramModel):

    def __init__(self, training_set, f2=1.0):
        super(BigramModel, self).__init__(training_set)
        self.f2 = f2
        self.bigrams = dict([(c, defaultdict(Distribution))
                             for c in training_set.classes])
        for cls, word in training_set:
            dist = self.bigrams[cls]
            for i, char in enumerate(START+word):
                (dist[char.lower()].incr(word[i].lower())
                 if i < len(word) else dist[char.lower()].incr(STOP))

    def get_probabilities(self, word):
        prob = defaultdict(lambda: 1.0)
        for i, char in enumerate(START+word.lower()):
            for cls, dist in self.bigrams.items():
                if char not in dist:
                    prob[cls] = 0.0
                    continue

                if i < len(word):
                    prob[cls] *= dist[char][word[i].lower()]
                else:
                    prob[cls] *= dist[char][STOP]

        # Linearly interpolate.
        unigram_prob = super(BigramModel, self).get_probabilities(word)
        [prob.__setitem__(k, self.f2*prob[k]+(1-self.f2)*unigram_prob[k])
         for k in prob]

        return prob


class TrigramModel(BigramModel):

    def __init__(self, training_set, f2=0.0, f3=1.0):
        super(TrigramModel, self).__init__(training_set, f2=f2)
        self.f3 = f3
        self.trigrams = dict([(c, defaultdict(Distribution))
                              for c in training_set.classes])
        for cls, word in training_set:
            dist = self.trigrams[cls]
            w = START+START+word.lower()+STOP+STOP
            [dist[w[i:i+2]].incr(char) for i, char in enumerate(w[2:])]

    def get_probabilities(self, word):
        prob = defaultdict(lambda: 1.0)
        w = START+START+word.lower()+STOP+STOP
        for i, char in enumerate(w[2:]):
            ind = w[i:i+2]
            for cls, dist in self.trigrams.items():
                if ind not in dist:
                    prob[cls] = 0.0
                    continue
                prob[cls] *= dist[ind][char]

        # Linearly interpolate.
        bigram_prob = super(TrigramModel, self).get_probabilities(word)
        [prob.__setitem__(k, self.f3*prob[k]+(1-self.f3)*bigram_prob[k])
         for k in prob]

        return prob


if __name__ == "__main__":
    training_set = Dataset("data/pnp-train.txt")
    validation_set = Dataset("data/pnp-validate.txt")
    test_set = Dataset("data/pnp-test.txt")

    model = UnigramModel(training_set)
    print("Unigram validation accuracy: {0}".format(
        model.test(validation_set)))

    model = BigramModel(training_set)
    print("Bigram validation accuracy: {0}".format(
        model.test(validation_set)))

    model = TrigramModel(training_set, f3=0.9, f2=0.9)
    print("Trigram validation accuracy: {0}".format(
        model.test(validation_set, outfile="hw2/validation.txt")))

    model.test(test_set, outfile="hw2/output_empirical.txt")
