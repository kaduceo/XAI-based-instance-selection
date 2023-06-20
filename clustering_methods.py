"""
clustering_methods.py
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

import sklearn_extra.cluster
import pandas as pd
import numpy as np

import six
import sys

sys.modules["sklearn.externals.six"] = six
from skrules import SkopeRules


def kmedoids_clustering(datas, percentage):
    """
    Clusters instances with a Kmedoids clustering.
    Number of cluster is defined as a percentage of the global dataset,
    with a minimum of two clusters.

    Parameters
    ----------
    datas : Pandas DataFrame
        Datas to compute.
    percentage : float
        Desired percentage of the total dataset desired as k-medoid instances.

    Returns
    -------
    sklearn_extra.cluster.KMedoids
        Clusters created on datas.

    """

    n_medoides = int(round(datas.shape[0] * percentage, 0))
    if n_medoides < 2:
        n_medoides = 2

    kmedoids = sklearn_extra.cluster.KMedoids(
        n_clusters=n_medoides,
        metric="euclidean",
        init="k-medoids++",
        max_iter=1000,
        random_state=6,
    )
    kmedoids.fit(datas)

    return kmedoids


def cluster_multiple_percentage(datas, list_percentages):
    """
    Apply k-medoid clustering on the data, for each percentage in the list.


    Parameters
    ----------
    datas : pandas.DataFrame
        Datas to cluster.
    list_percentages : list
        List of percentages to used for computing the number of clusters.

    Returns
    -------
    dict
        Clusters for each percentage.

    """

    return {p: kmedoids_clustering(datas, p) for p in list_percentages}


def clustering_skoperules(datas, labels, max_depth=2):
    """
    Apply Skope Rules algorithm to pre-defined clusters.


    Parameters
    ----------
    datas : pandas.DataFrame
        Datas used to compute clusters.
    labels : list
        List of clusters labels.
    max_depth : int. Default 2.
        Maximum of comparison terms for the rules.

    Returns
    -------
    dict
        Clusters for each percentage.

    """

    df = pd.DataFrame(columns=["Cluster", "Rule", "Precision", "Recall"])
    for cluster in np.unique(labels):
        if cluster >= 0:  # -1 is the non clusterized for dbscan
            # create target variable for individual cluster
            yc = (labels == cluster) * 1
            # use SkopeRules to identify rules with a maximum of comparison terms set in parameters
            sr = SkopeRules(
                feature_names=datas.columns,
                max_depth=max_depth,
                precision_min=0.1,
                recall_min=0.01,
                n_estimators=10,
            ).fit(datas.values, yc)
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [
                            [
                                cluster,
                                sr.rules_[0][0],
                                sr.rules_[0][1][0],
                                sr.rules_[0][1][1],
                            ]
                        ],
                        columns=df.columns,
                    ),
                ],
                ignore_index=True,
            )
    return df
