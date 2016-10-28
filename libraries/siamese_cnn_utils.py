
import numpy as np
import random


""" A bunch of Siamese CNN-specific methods.
I don't think they're useful anywhere else, so these get their own library.
Ref:
https://github.com/fchollet/keras/blob/master/examples/mnist_siamese_graph.py
"""

def create_pairs(x, digit_indices):
    """Positive and negative pair creation.
    Alternates between positive and negative pairs.

    The original create_pairs() method, workable but not elegant.
    Nevertheless, LEAVE THIS UNMODIFIED.

    My interpretation: select the smallest class, then pair it with members from the
    same classes, and a randomly selected member of another class, in an alternating fashion.
    If so, the max. number of similar pairs should be half the size of the smallest class.

    I think this assumes that all classes are of the same size.

    Params
    ------
    x: np array. The training data itself
    digit_indices: the digit indices of the numeric labels of x.
    """
    pairs = []
    labels = []
    n_classes = len(digit_indices)

    n = min([len(digit_indices[d]) for d in range(n_classes)]) - 1
    for d in range(n_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, n_classes)
            dn = (d + inc) % n_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]

    return np.array(pairs), np.array(labels)
