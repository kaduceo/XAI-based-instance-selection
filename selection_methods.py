import numpy as np
import pandas as pd
import sklearn as sk
import sklearn_extra.cluster

from mmdcritic import mmd_critic


def random_selection(X, percentage):
    n_random = int(round(X.shape[0] * percentage, 0))

    return X.sample(n_random).index


def kmedoids_selection(X_datas, percentage_):

    n_medoides = int(round(X_datas.shape[0] * percentage_, 0))

    # Clustering
    kmedoids = sklearn_extra.cluster.KMedoids(
        n_clusters=n_medoides,
        metric="euclidean",
        init="k-medoids++",
        max_iter=1000,
        random_state=6,
    )
    kmedoids.fit(X_datas)

    return kmedoids.medoid_indices_


def mmdcritic_selection(X, y, p_select, p_proto, gamma, ktype):
    """
    Adaptation of the MMD-critic method proposed by Kim et al. [2016]

    Parameters
    ----------
    X : Pandas DataFrame
        Raw datas or influences to compute.
    y : Pandas.Series

    p_select : float
        Desired percentage of the total dataset desired as recommanded instances.
    p_proto : float
        Desired percentage of the prototypes desired. 1 means no criticisms.
    gamma : float
        parameter for the kernel exp(- gamma * \| x1 - x2 \|_2)
    ktype : bool
        kernel type, 0 for global, 1 for local

    Returns
    -------
    list of recommanded instances indices.

    """

    n_proto = int(round(X.shape[0] * p_select * p_proto, 0))
    n_critic = int(round(X.shape[0] * p_select * (1 - p_proto), 0))
    if n_critic == 0:
        crit = False
    else:
        crit = True

    return mmd_critic(X, y, gamma, ktype, n_proto, n_critic, crit=crit)


def submodular_pick_selection(X, percentage):
    """
    Adaptation of the Submodular Pick method proposed by Ribeiro et al. [2016]

    Parameters
    ----------
    X : Pandas DataFrame
        Raw datas or influences to compute.
    percentage : float,
        Desired percentage of the total dataset desired as recommanded instances.

    Returns
    -------
    V : list
        list of recommanded instances indices.

    """
    n_instances = int(round(X.shape[0] * percentage, 0))

    importance = np.sum(abs(X), axis=0) ** 0.5
    remaining_indices = set(X.index)
    V = []
    for _ in range(n_instances):
        best = 0
        best_ind = 0
        current = 0
        for i in remaining_indices:
            current = np.dot(
                (np.sum(abs(X.loc[V + [i]]), axis=0) > 0), importance
            )  # original coverage function
            if current >= best:
                best = current
                best_ind = i
        V.append(best_ind)
        remaining_indices -= {best_ind}
    return V

