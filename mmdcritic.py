import numpy as np
import pandas as pd
import sklearn as sk

from sklearn.metrics.pairwise import rbf_kernel

# Global kernel
def calculate_kernel(X, g=None):
    return rbf_kernel(X, gamma=g)


# Local kernel
def calculate_kernel_individual(X, y, g=None):
    kernel = np.zeros((np.shape(X)[0], np.shape(X)[0]))
    sortind = np.argsort(y).values
    X = X.loc[sortind, :]
    y = y.loc[sortind]

    for i in np.arange(y.nunique()):
        ind = np.where(y == i)[0]
        startind = np.min(ind)
        endind = np.max(ind) + 1
        kernel[startind:endind, startind:endind] = rbf_kernel(
            X.iloc[startind:endind, :], gamma=g
        )
    return kernel


def greedy_select_protos(K, candidate_indices, m):
    ##############################################################################################################################
    # Function choose m of all rows by MMD as per kernelfunc
    # ARGS:
    # K : kernel matrix
    # candidate_indices : array of potential choices for selections, returned values are chosen from these  indices
    # m: number of selections to be made
    # RETURNS: subset of candidate_indices which are selected as prototypes
    ##############################################################################################################################

    if len(candidate_indices) != np.shape(K)[0]:
        K = K[:, candidate_indices][candidate_indices, :]

    n = len(candidate_indices)

    colsum = 2 * np.sum(K, axis=0) / n

    selected = np.array([], dtype=int)
    for i in range(m):
        argmax = -1
        candidates = np.setdiff1d(range(n), selected)

        s1array = colsum[candidates]
        if len(selected) > 0:
            temp = K[selected, :][:, candidates]
            s2array = np.sum(temp, axis=0) * 2 + np.diagonal(K)[candidates]
            s2array /= len(selected) + 1
            s1array -= s2array
        else:
            s1array -= np.abs(np.diagonal(K)[candidates])

        argmax = candidates[np.argmax(s1array)]
        selected = np.append(selected, argmax)
        KK = K[selected, :][:, selected]

    return candidate_indices[selected]


def select_criticism_regularized(K, selectedprotos, m, reg="logdet"):
    ##############################################################################################################################
    # function to select criticisms
    # ARGS:
    # K: Kernel matrix
    # selectedprotos: prototypes already selected
    # m : number of criticisms to be selected
    # reg: regularizer type.
    # RETURNS: indices selected as criticisms
    ##############################################################################################################################

    n = np.shape(K)[0]
    available_reg = {None, "logdet", "iterative"}
    assert (
        reg in available_reg
    ), f'Unknown regularizer: "{reg}". Available regularizers: {available_reg}'

    selected = np.array([], dtype=int)
    candidates2 = np.setdiff1d(range(n), selectedprotos)
    inverse_of_prev_selected = None  # should be a matrix

    colsum = np.sum(K, axis=0) / n
    inverse_of_prev_selected = None

    for i in range(m):
        argmax = -1
        candidates = np.setdiff1d(candidates2, selected)

        s1array = colsum[candidates]

        temp = K[selectedprotos, :][:, candidates]
        s2array = np.sum(temp, axis=0)
        s2array /= len(selectedprotos)
        s1array -= s2array

        if reg == "logdet":
            if inverse_of_prev_selected is not None:  # first call has been made already
                temp = K[selected, :][:, candidates]

                # hadamard product
                temp2 = np.array(np.dot(inverse_of_prev_selected, temp))
                regularizer = temp2 * temp
                regcolsum = np.sum(regularizer, axis=0)
                regularizer = np.log(abs(np.diagonal(K)[candidates] - regcolsum))
                s1array += regularizer
            else:
                s1array -= np.log(abs(np.diagonal(K)[candidates]))

        argmax = candidates[np.argmax(s1array)]
        selected = np.append(selected, argmax)

        if reg == "logdet":
            KK = K[selected, :][:, selected]
            inverse_of_prev_selected = np.linalg.pinv(KK)  # shortcut

        if reg == "iterative":
            selectedprotos = np.append(selectedprotos, argmax)

    return selected


def mmd_critic(X, y, gamma, ktype, n, m, crit=True):
    ##############################################################################################################################
    # this function makes selects prototypes/criticisms and outputs the indices of the selected instances.
    # ARGS:
    # X: datas
    # y: labels
    # gamma: parameter for the kernel exp(- gamma * \| x1 - x2 \|_2)
    # ktype: kernel type, 0 for global, 1 for local
    # n : number of prototypes wanted
    # m : number of criticisms wanted
    # RETURNS: returns indices of  selected prototypes and criticisms
    ##############################################################################################################################

    if ktype == 0:
        kernel = calculate_kernel(X, gamma)
    else:
        kernel = calculate_kernel_individual(X, y, gamma)

    selected = greedy_select_protos(kernel, np.array(range(np.shape(kernel)[0])), n)

    critselected = np.array([], dtype=int)
    if crit:
        critselected = select_criticism_regularized(kernel, selected, m, reg="logdet")

    return np.concatenate((selected, critselected))



