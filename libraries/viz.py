import matplotlib.pyplot as plt
import numpy as np
import pickle


""" A bunch of image reconstruction and visualization methods.
"""

def bicluster(data, linkage_method='average', distance_metric='correlation'):
    """Juan's code.Cluster the rows and the columns of a matrix.

    Parameters
    ----------
    data : 2D ndarray
        The input data to bicluster.
    linkage_method : string, optional
        Method to be passed to `linkage`.
    distance_metric : string, optional
        Distance metric to use for clustering. See the documentation
        for ``scipy.spatial.distance.pdist`` for valid metrics.

    Returns
    -------
    y_rows : linkage matrix
        The clustering of the rows of the input data.
    y_cols : linkage matrix
        The clustering of the cols of the input data.
    """
    y_rows = linkage(data, method=linkage_method, metric=distance_metric)
    y_cols = linkage(data.T, method=linkage_method, metric=distance_metric)
    return y_rows, y_cols


def plot_bicluster(data, row_linkage, col_linkage, x_label, y_label,
                   row_nclusters=10, col_nclusters=3):
    """Juan's code.Perform a biclustering, plot a heatmap with dendrograms on each axis.

    Parameters
    ----------
    data : array of float, shape (M, N)
        The input data to bicluster.
    row_linkage : array, shape (M-1, 4)
        The linkage matrix for the rows of `data`.
    col_linkage : array, shape (N-1, 4)
        The linkage matrix for the columns of `data`.
    n_clusters_r, n_clusters_c : int, optional
        Number of clusters for rows and columns.
    xlabel : str; x-axis label.
    ylabel : str; y-axis label.
    """
    fig = plt.figure(figsize=(10, 10))

    # Compute and plot row-wise dendrogram
    # `add_axes` takes a "rectangle" input to add a subplot to a figure.
    # The figure is considered to have side-length 1 on each side, and its
    # bottom-left corner is at (0, 0).
    # The measurements passed to `add_axes` are the left, bottom, width, and
    # height of the subplot. Thus, to draw the left dendrogram (for the rows),
    # we create a rectangle whose bottom-left corner is at (0.09, 0.1), and
    # measuring 0.2 in width and 0.6 in height.
    ax1 = fig.add_axes([0.09, 0.1, 0.2, 0.6])
    # For a given number of clusters, we can obtain a cut of the linkage
    # tree by looking at the corresponding distance annotation in the linkage
    # matrix.
    threshold_r = (row_linkage[-row_nclusters, 2] +
                   row_linkage[-row_nclusters+1, 2]) / 2
    dendrogram(row_linkage, orientation='left', color_threshold=threshold_r)

    # Compute and plot column-wise dendogram
    # See notes above for explanation of parameters to `add_axes`
    ax2 = fig.add_axes([0.3, 0.71, 0.6, 0.2])
    threshold_c = (col_linkage[-col_nclusters, 2] +
                   col_linkage[-col_nclusters+1, 2]) / 2
    dendrogram(col_linkage, color_threshold=threshold_c)

    # Hide axes labels
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Plot data heatmap
    ax = fig.add_axes([0.3, 0.1, 0.6, 0.6])

    # Sort data by the dendogram leaves
    idx_rows = leaves_list(row_linkage)
    data = data[idx_rows, :]
    idx_cols = leaves_list(col_linkage)
    data = data[:, idx_cols]

    im = ax.matshow(data, aspect='auto', origin='lower', cmap='YlGnBu_r')
    ax.set_xticks([])
    ax.set_yticks([])

    # Axis labels
    plt.xlabel(x_label)
    plt.ylabel(y_label, labelpad=125)

    # Plot legend
    axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.6])
    plt.colorbar(im, cax=axcolor)

    # display the plot
    plt.show()


def form_quilt(quilt_w, patch_data):
    """Creates a grid ('quilt') out of an bunch of input patches in patch_data.

    PARAMS
    ------
    patch_data : array of float; shape (n_patches, patch_len, patch_len, ch)
    quilt_w : int; width of the quilt

    RETURNS
    -------
    quilt : array of float; shape (patch_len*quilt_width,
    patch_len*quilt_height, ch)
    """
    n_patches = len(patch_data)

    quilt_ls = []
    for i in range(0, len(patch_data), quilt_w):
        # start the row to have something to hstack to
        q_row = patch_data[i]
        for j in range(i+1, i+quilt_w):
            q_row = np.hstack((q_row, patch_data[j]))
        quilt_ls.append(q_row)

    quilt = quilt_ls[0]
    for i in range(1, len(quilt_ls)):
        quilt = np.vstack((quilt, quilt_ls[i]))

    return quilt


def get_reconstruction_error(XV, original, alpha):
    """Compute the reconstruction error of the array `XV`, which should be
    just np.dot(X, V). Both XV and original should be 0-1 normalized.

    PARAMS
    ------
    XV : array of float; shape (n_samples, im_size).
        The reconstruction: np.dot(sparse_encodings, dictionary)
    Original : array of float; shape (n_samples, im_size)
    method : distance metric. Only Euclidean so far.
    alpha : value of alpha used in the previous sparse-encoding step

    RETURNS
    -------
    err_arr :  array of float of length n_samples.
        Reconstruction error for each sample.
    err_mean : average of err_arr
    err_sd : standard deviation of err_arr
    loss_arr : array of DL loss function error, with regularization param alpha
    """

    err_arr = []
    loss_arr = []
    for i in range(len(XV)):
        err = np.sqrt(np.sum((XV[i]-original[i])**2))
        err_arr.append(err)
    err_arr = np.array(err_arr)

    # Get the stats
    err_mean = np.average(err_arr)
    err_sd = np.sqrt(np.var(err_arr))

    return err_arr, err_mean, err_sd


def get_loss(X, XV, original, alpha):
    """Compute the error of the DL objective function, taking into account
    sparsity param alpha. WOW this needs a better name.

    Params
    ------
    XV : array of float; shape (n_samples, im_size).
        The reconstruction: np.dot(sparse_encodings, dictionary)
    X : The sparse encodings; shape (n_samples, n_atoms).
    original : the original input data, shape (n_samples, im_size).
    alpha : sparse encoding parameter.

    Returns
    -------
    loss_arr: array of float; shape (n_samples)
        The error for each patch. Note that you'll end up taking the average.

    see: http://scikit-learn.org/stable/modules/decomposition.html#dictionarylearning
    """
    loss_arr = []
    for i in range(len(XV)):
        loss = 0.5*np.sqrt((original[i]-XV[i])**2) + alpha*len(np.flatnonzero(X[i]))
        loss_arr.append(loss)

    return loss_arr


def show_with_diff(original, reconstruction, title, cmap=plt.cm.gray):
    """Displays 2 subplots: the original image image, the reconstruction.
    Adapted from:
    http://scikit-learn.org/stable/auto_examples/decomposition/plot_image_denoising.html

    Params
    ------
    original: np array of float. Original image
    reconstruction: np array of float
    Reconstructed image. Or any other image, really.
    title: string; Do I really need to explain what a title is?
    cmap: string. Colour map argument to be passed to imshow().
    """

    im_h, im_w, ch = original.shape
    #Subplot 1: Show original
    plt.figure(figsize=(10, 6.6))
    plt.subplot(1, 3, 1)
    plt.title('Original')
    plt.imshow(original, cmap=cmap, interpolation='nearest')
    plt.xticks(())
    plt.yticks(())

    #Subplot 2: Show reconstruction
    plt.subplot(1, 3, 2)
    plt.title('Reconstruction')
    plt.imshow(reconstruction, cmap=cmap, interpolation='nearest')
    plt.xticks(())
    plt.yticks(())

    difference = original - reconstruction
    diff_euc = np.sqrt(np.sum(difference ** 2))
    diff_px = diff_euc/(im_h*im_w*ch)
    print('Difference = %.2f' % diff_euc)
    print("Ave. pixel difference = %.5f" % diff_px)
