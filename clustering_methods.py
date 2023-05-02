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


def kmedoids_selection(datas, percentage):
    """
    Clusters instances with a Kmedoids clustering, based on the percentage of the global dataset,
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
