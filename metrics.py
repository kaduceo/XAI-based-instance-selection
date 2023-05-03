"""
metrics.py
Copyright (C) 2020 Elodie Escriva, Kaduceo <elodie.escriva@kaduceo.com>



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


def compute_intra_inertia(datas, center):
    """
    Compute intra-inertia of a cluster, based on the data and the center.

    Parameters
    ----------
    datas : pandas.DataFrame
        Datas of all element of the studied cluster.
    center : pandas.Series
        Data of the center of the studied cluster.

    Returns
    -------
    inertia : float
        Intra-cluster inertia of the studied cluster.

    """
    inertia = 0
    for i in datas.index:
        inertia += np.linalg.norm(datas.loc[i] - center)
    return inertia


def compute_inter_inertia(centers, len_cluster):
    """
    Compute inter-inertia of all cluster, based on the centers.

    Parameters
    ----------
    centers : pandas.DataFrame
        Center of each cluster.
    len_cluster : list
        Size of all clusters.

    Returns
    -------
    inertia : float
        Inter-cluster inertia of the clusters.

    """
    inertia = 0
    barycenter = (1 / len(centers)) * np.sum(centers)
    for i in range(centers.shape[0]):
        inertia += len_cluster[i] * np.linalg.norm(centers.iloc[i] - barycenter)
    return inertia


def compute_clusters_inertie(datas, clusters_labels, centers):
    """
    Compute intra and inter inertia for all clusters in parameter.

    Parameters
    ----------
    datas : pandas.DataFrame
        Datas of the elements used for clustering.
    clusters_labels : dictionary
        Labels of instances.
    centers : pandas.DataFrame
        Center of each cluster

    Returns
    -------
    intra_inertia : dictionary
        Intra inertia of the clustering.
    inter_inertia : dictionary
        Inter inertia of the clustering.

    """
    len_cluster = []
    intra_inertia = 0

    for k in range(len(centers)):
        datas_cluster = datas.loc[clusters_labels == k]
        len_cluster.append(len(datas_cluster))
        intra_inertia += compute_intra_inertia(datas_cluster, X.iloc[centers[k]])
    inter_inertia = compute_inter_inertia(datas.iloc[centers], len_cluster)

    return intra_inertia, inter_inertia


def compute_entropy_cluster(datas_y, nb_class):
    """
    Compute entropy for one cluster, as defined by Conrad et al. (2005)

    Parameters
    ----------
    datas_y : pandas.DataFrame
        labels of instances from the studied cluster.
    nb_class : int
        Number of classes


    Returns
    -------
    entropy : float
        The entropy value for the studied cluster.

    """
    entropy = 0
    n_i = datas_y.value_counts()
    n = datas_y.shape[0]
    for i in n_i.index:
        if n_i[i] != 0:
            entropy += (n_i[i] / n) * (np.log(n_i[i]) - np.log(n))
    return np.abs((-1 / np.log(nb_class)) * entropy)


def compute_purity_cluster(datas_y):
    """
    Compute purity for one cluster, as defined by Conrad et al. (2005)

    Parameters
    ----------
    datas_y : pandas.DataFrame
        labels of instance from the studied cluster.

    Returns
    -------
    purity : float
        The purity value for the studied cluster.

    """
    return (1 / datas_y.shape[0]) * np.max(datas_y.value_counts())


def compute_clusters_purity_entropy(clusters_labels, y_labels):
    """
    Compute purity and entropy for all clusters in parameter.

    Parameters
    ----------
    clusters_labels : dictionary
        Labels of instances.
    y_labels : pandas.DataFrame
        Labels of all datas.

    Returns
    -------
    entropy : dictionary
        Entropy of the clustering.
    purity : dictionary
        Purity of the clustering.

    """
    entropy = 0
    purity = 0
    for i in range(max(clusters_labels)):
        q = y_labels.unique().shape[0]
        datas_cluster = y_labels.loc[clusters_labels == i]
        if datas_cluster.shape[0] != 0:
            penalisation = datas_cluster.shape[0] / y_labels.shape[0]
            entropy += penalisation * compute_entropy_cluster(datas_cluster, q)
            purity += penalisation * compute_purity_cluster(datas_cluster)
    return purity, entropy
