"""
utils.py
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

import pandas as pd

import shap
import lime
from lime import lime_tabular
import openml as oml

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector


def process_dataset_openml(dataset_id, variable_pred):
    """
    Retrieves dataset from openml based on the task id in parameters,
    and pre-process data to deal with categorical attributs.

    Parameters
    ----------
    task_id : int
        Index of the dataset in Open ML.

    Returns
    -------
    X_process : pandas.DataFrame
        Process datas from the dataset.
    y_process : pandas.DataFrame
        Process datas from the dataset.
    """
    dataset = oml.datasets.get_dataset(dataset_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OrdinalEncoder()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, selector(dtype_exclude="category")),
            ("cat", categorical_transformer, selector(dtype_include="category")),
        ]
    )

    X_process = preprocessor.fit_transform(X)
    X_process = pd.DataFrame(X_process, columns=X.columns)

    label_encoder = LabelEncoder()
    y_process = label_encoder.fit_transform(y)
    y_process = pd.DataFrame(y_process, columns=[variable_pred])

    return X_process, y_process


def TreeSHAP_positive_class(modele_, X_, y_):
    tree_shap_explainer_global = shap.TreeExplainer(
        modele_,
        data=X_.values,
        feature_perturbation="interventional",
        model_output="probability",
    )
    treeshap_values = tree_shap_explainer_global.shap_values(
        X_.values, check_additivity=True
    )[
        1
    ]  # On veut uniquement la classe positive, donc [1]
    return pd.DataFrame(treeshap_values, columns=X_.columns, index=X_.index)


def KernelSHAP_positive_class(modele_, X_, y_, num_kmeans=5):
    kernel_shap_explainer_global = shap.KernelExplainer(
        modele_.predict_proba, data=shap.kmeans(X_.values, num_kmeans)
    )
    kernelshap_values = kernel_shap_explainer_global.shap_values(
        X_.values, check_additivity=True
    )[
        1
    ]  # On veut uniquement la classe positive, donc [1]
    return pd.DataFrame(kernelshap_values, columns=X_.columns, index=X_.index)


def LIME_positive_class(
    modele_, X, y, mode="classification", look_at=1, num_samples=100
):
    explainer = lime_tabular.LimeTabularExplainer(
        X.values, mode=mode, feature_names=X.columns
    )
    inf_values = np.array(
        [
            [
                v
                for (k, v) in sorted(
                    explainer.explain_instance(
                        X.values[i],
                        modele_.predict_proba,
                        labels=(look_at,),
                        num_samples=num_samples,
                        num_features=X.shape[1],
                    ).as_map()[look_at]
                )
            ]
            for i in range(X.shape[0])
        ]
    )

    # Generate explanation compatible with shap
    explanation = shap.Explanation(
        inf_values,
        base_values=np.zeros(X.shape[0]),
        data=X.values,
        feature_names=X.columns.to_list(),
    )

    shap_values = pd.DataFrame(inf_values, columns=X.columns, index=X.index)

    return shap_values
