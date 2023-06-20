"""
utils.py
Copyright (C) 2020 Elodie Escriva, Kaduceo <elodie.escriva@kaduceo.com>

LIME (https://arxiv.org/abs/1602.04938)
Copyright (c) 2016, Marco Tulio Correia Ribeiro

KernelSHAP (https://arxiv.org/abs/1705.07874)
Copyright (c) 2017, Scott Lundberg
TreeSHAP (https://doi.org/10.1038/s42256-019-0138-9)
Copyright (c) 2020, Scott Lundberg


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
import pandas as pd
import sklearn as sk
import openml as oml

import shap
from lime import lime_tabular

import xgboost as xgb

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold, GridSearchCV, RepeatedStratifiedKFold


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
        Process data from the dataset.
    y_process : pandas.DataFrame
        Process data from the dataset.
    """
    dataset = oml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(
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


def fct_RF_gridsearch(
    X, y, n_folds=5, n_repeats=1, scoring="balanced_accuracy", param_grid=None
):
    """
    Search the best parameters for Random Forest, based on the data and labels in parameters.

    Parameters
    ----------
    X : pandas.DataFrame
        Data to train the model.
    y : pandas.DataFrame
        Labels of the data.
    n_folds : int. Default 5.
        Number of fold for the Reapeated Cross-Validation.
    n_repeats : int. Default 1.
        Number of times cross-validator needs to be repeated.
    scoring : string. Default balanced accuracy.
        Strategy to evaluate the performance of the cross-validated model on the test set.
    param_grid : dictionnary. Default None.
        Grid of parameters to test during Grid Search

    Returns
    -------
    Sklearn.ensemble.RandomForestClassifier
        Model with the best parameters based on the Grid Search Repeated k-folds Cross-Validation.
    """
    type_CV = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats)

    if param_grid is None:
        param_grid = {
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [None, 2, 5, 10, 15],
            "min_samples_split": [2, 5, 10],
        }

    gs_repeatedCV = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=param_grid,
        cv=type_CV,
        scoring=scoring,
        refit=False,
    )

    gs_repeatedCV.fit(X.values, y.values)

    return RandomForestClassifier(**gs_repeatedCV.best_params_)


def fct_XGB_gridsearch(
    X, y, n_folds=5, n_repeats=2, scoring="roc_auc", param_grid=None
):
    """
    Search the best parameters for XGBoost, based on the data and labels in parameters.

    Parameters
    ----------
    X : pandas.DataFrame
        Data to train the model.
    y : pandas.DataFrame
        Labels of the data.
    n_folds : int. Default 5.
        Number of fold for the Reapeated Cross-Validation.
    n_repeats : int. Default 2.
        Number of times cross-validator needs to be repeated.
    scoring : string. Default roc-auc.
        Strategy to evaluate the performance of the cross-validated model on the test set.
    param_grid : dictionnary. Default None.
        Grid of parameters to test during Grid Search

    Returns
    -------
    Sklearn.ensemble.RandomForestClassifier
        Model with the best parameters based on the Grid Search Repeated k-folds Cross-Validation.
    """
    type_CV = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats)

    sqrt_colsample_bytree = np.sqrt(X.shape[1]) / X.shape[1]
    balanced_scale_pos_weight = (y.squeeze() == 0).sum() / (y.squeeze() == 1).sum()

    if param_grid is None:
        param_grid = {
            "objective": ["binary:logistic"],
            "eval_metric": ["error"],
            "max_depth": [4, 5, 6],
            "n_estimators": [200],
            "learning_rate": [0.1, 0.025],
            "min_child_weight": [5, 10],
            "subsample": [0.7, 1],
            "colsample_bytree": [1, 0.5, sqrt_colsample_bytree],
            "scale_pos_weight": [1, balanced_scale_pos_weight],
            "use_label_encoder": [False],
        }

    gs_repeatedCV = GridSearchCV(
        estimator=xgb.XGBClassifier(),
        param_grid=param_grid,
        cv=type_CV,
        scoring=scoring,
        refit=False,
    )

    gs_repeatedCV.fit(X, y)

    return xgb.XGBClassifier(**gs_repeatedCV.best_params_)


def TreeSHAP_oneclass(model, X, look_at=1):
    """
    Explains the modelling with the TreeSHAP methods for one class. (Lundberg, 2020)

    Parameters
    ----------
    model
        Model trained during the modelling.
    X : pandas.DataFrame
        Data used for the modelling.
    look_at : int. Default 1.
        Class to look at for explanations.

    Returns
    -------
    pandas.DataFrame
        TreeSHAP explanations of the modelling for the selected class.
    """
    tree_shap_explainer_global = shap.TreeExplainer(
        model,
        data=X.values,
        feature_perturbation="interventional",
        model_output="probability",
    )

    treeshap_values = tree_shap_explainer_global.shap_values(
        X.values, check_additivity=True
    )

    if isinstance(treeshap_values, list):
        treeshap_values = treeshap_values[look_at]

    return pd.DataFrame(treeshap_values, columns=X.columns, index=X.index)


def KernelSHAP_oneclass(model, X, look_at=1, num_kmeans=5):
    """
    Explains the modelling with the KernelSHAP methods for one class. (Lundberg, 2017)

    Parameters
    ----------
    model
        Model trained during the modelling.
    X : pandas.DataFrame
        Data used for the modelling.
    look_at : int. Default 1.
        Class to look at for explanations.
    num_kmeans : int. Default 5.
        Number of clusters for summarising the data.

    Returns
    -------
    pandas.DataFrame
        KernelSHAP explanations of the modelling for the selected class.
    """
    kernel_shap_explainer_global = shap.KernelExplainer(
        model.predict_proba, data=shap.kmeans(X.values, num_kmeans)
    )
    kernelshap_values = kernel_shap_explainer_global.shap_values(
        X.values, check_additivity=True
    )[look_at]
    return pd.DataFrame(kernelshap_values, columns=X.columns, index=X.index)


def LIME_oneclass(model, X, mode="classification", look_at=1, num_samples=100):
    """
    Explains the modelling with the LIME methods for one class. (Ribeiro, 2016)

    Parameters
    ----------
    model
        Model trained during the modelling.
    X : pandas.DataFrame
        Data used for the modelling.
    mode : 'classification' or 'regression'
        Type of modelling.
    look_at : int. Default 1.
        Class to look at for explanations.
    num_samples : int. Default 100
        Size of the neighborhood to learn the linear model

    Returns
    -------
    lime_values : pandas.DataFrame
        LIME explanations of the modelling for the selected class.
    explanations : shap.Explanation
        Explanation compatible with shap
    """
    explainer = lime_tabular.LimeTabularExplainer(
        X.values, mode=mode, feature_names=X.columns
    )
    inf_values = np.array(
        [
            [
                v
                for (_, v) in sorted(
                    explainer.explain_instance(
                        X.values[i],
                        model.predict_proba,
                        labels=(look_at,),
                        num_samples=num_samples,
                        num_features=X.shape[1],
                    ).as_map()[look_at]
                )
            ]
            for i in range(X.shape[0])
        ]
    )

    lime_values = pd.DataFrame(inf_values, columns=X.columns, index=X.index)

    explanation = shap.Explanation(
        inf_values,
        base_values=np.zeros(X.shape[0]),
        data=X.values,
        feature_names=X.columns.to_list(),
    )

    return lime_values
