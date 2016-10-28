
import numpy as np
import pickle


""" A bunch of DL-specific methods.
I don't think they're useful anywhere else, so these get their own library.
More to hide a few dozen lines of code, really.
"""

def read_p_list(path, g_normalize = True, verbose=True):
    """Pickle.loads a p_list from the given path.
    """
    data_list = pickle.load(open(path, "rb"))
    if verbose:
        print(data_list.shape)

    # Get different levels of labels
    moa_labels = []
    for row in data_list:
        moa_labels.append(row[3])
    print("No. of MOA labels = %s" % len(moa_labels))

    #Retrieve a list of compound labels
    cpd_labels = []
    for row in data_list:
        cpd_labels.append(row[1])
    print("No. of cpd labels = %s" % len(cpd_labels))

    #Get a list of compound-concentration labels
    cc_labels = []
    for row in data_list:
        label = row[1]+str(row[2])
        cc_labels.append(label)
    print("No. of cc labels = %s" % len(cc_labels))

    # Load patch data
    mydata = []
    for row in data_list:
        mydata.append(row[0])

    mydata = np.array(mydata)
    n_patches, patch_len, patch_len, ch = mydata.shape
    if verbose:
        print("No. of patches = %s" % n_patches)
        print("Patch shape = (%s, %s, %s)" % (patch_len, patch_len, ch))

    # Unravel each patch
    mydata = mydata.reshape(mydata.shape[0],-1)

    # Standard normalization
    if g_normalize:
        mydata -= np.mean(mydata, axis = 0)
        mydata /= np.std(mydata, axis = 0)
        mydata[np.isnan(mydata)] = 0.0

    return data_list, mydata, moa_labels, cpd_labels, cc_labels
