
import numpy as np
#import scipy
import os
import sys
import time
import re


"""Various methods that are frequently used,
and aren't specific to any module.
Probably methods that aren't specific to any particular part of the pipeline?
"""
def swap_image_format(image):
    """changes from image.shape(height, width, ch) to (ch, height, width)
    """
    image_new = np.array([image[:,:,0],
                         image[:,:,1],
                         image[:,:,2]])
    return image_new

def swap_image_format_list(image_array):
    """performs the image format swap for each image in an array of images
    """
    img_array_new = []
    for i in range(len(image_array)):
        img_new = swap_image_format(image_array[i])
        img_array_new.append(img_new)

    img_array_new = np.array(img_array_new)

    return img_array_new


def flatten_sparse_matrix(x, onehot = True):
    """Sets all nonzero entries of x to 1,
    then compresses to a 1D vector by taking the sum across rows
    i.e. squishes a rectangular matrix vertically.

    Params
    ------
    x: Array of float, shape (m, n)
    onehot: if true, converts any nonzero entry into 1

    Returns
    -------
    x_vec: 1D array of float, shape (n,)"""

    x_vec = x
    #Set all the nonzero elements of x to 1
    if onehot:
        x_vec[x_vec != 0] = 1
    x_vec = np.sum(x_vec, axis = 0)

    return x_vec


def sparse_vec_format(v):
    """Returns the indices and values of the nonzero elements
    of a sparse vector, v
    """

    C = []
    idx = []
    for i in range(len(v)):
        if v[i] != 0:
            C.append(v[i])
            idx.append(i)

    return C, idx


def my_one_hot_binary(indices):
    """Converts an array of category labels, represented as integers
    into a binary vector
    e.g.
    >>>my_labels = np.array([0,1,3,4])
    >>>my_one_hot_vectorize(my_labels)
    array([[ 1.,  0.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  0.,  1.]])"""
    encoded = np.zeros((len(indices), max(indices)+1))
    encoded[np.arange(len(indices)),indices] = 1

    return encoded


def my_one_hot_index(labels, label_ref):
    """Converts an array of labels (string) into integer indices
    According to a reference list label_ref

    e.g.
    >>label_ref=['dog', 'cat']
    >>labels=np.array(['dog', 'dog', 'dog', 'cat'])
    >>my_one_hot(labels, label_ref)
    array([0, 0, 0, 1], dtype=uint8)

    Params
    ------
    labels: array of the data labels
    label_ref: a list of the labels to look up from

    Returns
    -------
    indices: array of integers: the labels as integer indices
    """
    indices = np.zeros((len(labels)),dtype='uint8')

    for i in range(len(labels)):
        idx = label_ref.index(labels[i])
        indices[i] = idx

    return indices
