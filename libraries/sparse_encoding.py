
import numpy as np
import pickle
import time
import pandas as pd

from sklearn.decomposition import sparse_encode


""" A bunch of methods in the sparse-encoding step. Mostly wrapper and utility
functions. Put in s module to hide text, basically.
"""
def patch_and_sparse_encode(d0, V, alpha_ls, method='lasso_lars', verbose=1):
    """
    WRAPPER.
    Extract image data from dataframe d0,
    unravel patches, do Gaussian normalization.
    Then sparse encode that image data based on V, using method.

    Params
    ------
    d0: pandas dataframe, cols: img, pw (plate-well), cpd, cc, moa, img_idx.
    pw is not used for your purposes.
    V: Some dictionary learnt on slurm.
    alpha_ls: list of values of alpha to use for sparse encoding
    method: 'lasso_lars', or 'omp'. Algorithm used by the sparse encoding method.

    Returns
    -------
    X_dict: dictionary of sparse encodings, for different values of alpha:
        X_dict = {1:, 2:, 4:, 8:, 16:, 32:}
    """
    # first lets get the raw image data
    img_data = np.array(list(d0['img']))
    img_data = img_data.reshape(img_data.shape[0],-1)
    img_data -= np.mean(img_data, axis = 0)
    img_data /= np.std(img_data, axis = 0)
    img_data[np.isnan(img_data)] = 0.0
    if verbose > 0:
        print("Img data shape = %s" % (img_data.shape,))
        print("Dictionary shape = %s" % (V.shape,))

    # Initialize X_dict
    X_dict = {}
    # Just in case
    alpha_ls.sort()

    for a in alpha_ls:
        t0 = time.time()
        if verbose > 0:
            print(a, end = ", ")
        if method == 'lasso_lars':
            X = sparse_encode(img_data, V,
                      algorithm='lasso_lars',
                      alpha=a,
                     )
        elif method == 'omp':
            X = sparse_encode(img_data, V,
                      algorithm='omp',
                      n_nonzero_coefs=a
                     )
        print("%.2fs" % (time.time() - t0))
        X_dict[a] = X

    return X_dict


def get_img_data(dfr, normalize=True):
    """Extract image data from a pd.Dataframe of choice, then does two steps
    of normalization: Gaussian, then 0-1 (optional).

    Params
    ------
    dfr: pd.dataframe. Must have the image data under the colname 'img'.

    Returns
    -------
    img_data: array shape (n_samples, unravelled patch length)
    """
    img_data = np.array(list(dfr['img']))
    img_data = img_data.reshape(img_data.shape[0],-1)
    img_data -= np.mean(img_data, axis = 0)
    img_data /= np.std(img_data, axis = 0)
    img_data[np.isnan(img_data)] = 0.0

    if normalize == True:
        img_data = (img_data - img_data.min())/(img_data.max()-img_data.min())

    return img_data



def create_quilt(i0, n_patches, im_data, ht):
    """Creates a quilt comprising n_patches patches from a subset of im_data

    Params
    ------
    i0: starting index of patch to visualize
    n_patches: total no. of patches to viz
    im_data: raw image data, shape (n_samples, p_len, p_len, 3)
    ht: height of output quilt

    Returns
    -------
    quilt: array of float, shape (n_patches/ht, ht)

    NOTE: You're encountering index-out-of-bounds error unexpected early. Why?
    """
    quilt = []
    for j in range(i0, n_patches, ht):
        row = im_data[j:j+ht]
        row = np.vstack(row)
        quilt.append(row)
    quilt = np.hstack(np.array(quilt))

    return quilt
