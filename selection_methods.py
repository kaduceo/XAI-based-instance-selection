"""
selection_methods.py
Copyright (C) 2020 Elodie Escriva, Kaduceo <elodie.escriva@kaduceo.com>

Submodular Pick (https://arxiv.org/abs/1602.04938)
Copyright (c) 2016, Marco Tulio Correia Ribeiro

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import sklearn_extra.cluster

from mmdcritic import mmd_critic


def random_selection(X, percentage):
    """
    Selects random instances from a pandas DataFrame.

    Parameters
    ----------
    X : Pandas DataFrame
        Raw datas or influences to compute.
    percentage : float
        Desired percentage of the total dataset desired as selected instances..

    Returns
    -------
    Pandas Index
        Indices of the selected instances.

    """
    n_random = int(round(X.shape[0] * percentage, 0))

    return X.sample(n_random).index


def kmedoids_selection(X, percentage):
    """
    Selects instances based on the medoids of a Kmedoids clustering.

    Parameters
    ----------
    X : Pandas DataFrame
        Raw datas or influences to compute.
    percentage : float
        Desired percentage of the total dataset desired as selected instances.

    Returns
    -------
    sklearn_extra.cluster.KMedoids
        Medoids created on X.

    """

    n_medoides = int(round(X.shape[0] * percentage, 0))
    if n_medoides < 2:
        n_medoides = 2

    # Clustering
    kmedoids = sklearn_extra.cluster.KMedoids(
        n_clusters=n_medoides,
        metric="euclidean",
        init="k-medoids++",
        max_iter=1000,
        random_state=6,
    )
    kmedoids.fit(X)

    return kmedoids


def mmdcritic_selection(X, y, p_select, p_proto, gamma=None, ktype=0):
    """
    Adaptation of the MMD-critic method proposed by Kim et al. [2016]

    Parameters
    ----------
    X : Pandas DataFrame
        Raw datas or influences to compute.
    y : Pandas.Series
        Labels of the instances in the input dataset.
    p_select : float
        Desired percentage of the total dataset desired as selected instances.
    p_proto : float
        Desired percentage of the prototypes desired. 1 means no criticisms.
    gamma : float, optional
        parameter for the kernel exp(- gamma * \| x1 - x2 \|_2). Default None.
    ktype : bool, optional
        kernel type, 0 for global, 1 for local. Default 0.

    Returns
    -------
    Numpy Array
        list of recommanded instances indices.

    """

    n_proto = int(round(X.shape[0] * p_select * p_proto, 0))
    n_critic = int(round(X.shape[0] * p_select * (1 - p_proto), 0))
    if n_critic == 0:
        crit = False
    else:
        crit = True

    return mmd_critic(X, y, n_proto, n_critic, gamma, ktype, crit)


def submodular_pick_selection(X, percentage):
    """
    Adaptation of the Submodular Pick method proposed by Ribeiro et al. [2016]

    Parameters
    ----------
    X : Pandas DataFrame
        Raw datas or influences to compute.
    percentage : float,
        Desired percentage of the total dataset desired as selected instances.

    Returns
    -------
    selected_indices : list
        list of recommanded instances indices.

    """
    n_instances = int(round(X.shape[0] * percentage, 0))

    importance = np.sum(abs(X), axis=0) ** 0.5
    remaining_indices = set(X.index)
    selected_indices = []
    for _ in range(n_instances):
        best = 0
        best_ind = 0
        current = 0
        for i in remaining_indices:
            current = np.dot(
                (np.sum(abs(X.loc[selected_indices + [i]]), axis=0) > 0), importance
            )  # original coverage function
            if current >= best:
                best = current
                best_ind = i
        selected_indices.append(best_ind)
        remaining_indices -= {best_ind}
    return selected_indices
