"""
This is a boilerplate pipeline 'train_wine_model'
generated using Kedro 0.19.8
"""

import mlflow
import matplotlib.pyplot as plt
import mlflow.xgboost

from xgboost import XGBClassifier
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import datasets

import tempfile
import pandas as pd
import typing as t
import os

import seaborn as sns
import matplotlib.pyplot as plt


def _get_X_y(df: pd.DataFrame) -> t.Tuple:
    X = df.iloc[:, :-2]
    y = df["target"].tolist()

    return X, y


def _create_confusion_matrix(y_test, y_pred, labels, temp_dir):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=labels, yticklabels=labels
    )
    plt.ylabel("Previsão", fontsize=12)
    plt.xlabel("Verdadeiro", fontsize=12)
    plt.title("Matriz de Confusão", fontsize=16)

    output_path = os.path.join(temp_dir, "confusion_matrix.png")
    plt.savefig(output_path)

    return output_path


def get_data() -> pd.DataFrame:
    dataset = datasets.load_wine()
    dataset_df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    dataset_df["target"] = dataset.target
    dataset_df["label"] = [dataset.target_names.tolist()[t] for t in dataset.target]
    return dataset_df


def split_train_test(df: pd.DataFrame, parameters: t.Dict) -> t.Tuple:
    train_set, test_set = train_test_split(
        df,
        test_size=parameters["test_size"],
        random_state=parameters["random_state"],
        stratify=df["target"],
    )
    return train_set, test_set


def train_model(
    train_set: pd.DataFrame, test_set: pd.DataFrame, parameters: t.Dict
) -> XGBClassifier:
    X_train, y_train = _get_X_y(train_set)
    X_test, y_test = _get_X_y(test_set)

    labels = test_set.sort_values(["target"])["label"].unique()
    f1_scorer = make_scorer(f1_score, average="weighted")

    xgb_model = XGBClassifier()
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=parameters["xgboost_params"],
        scoring=f1_scorer,
        cv=5,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    mlflow.xgboost.log_model(grid_search.best_estimator_, "model")

    for param_name, param_value in grid_search.best_params_.items():
        mlflow.log_param(param_name, param_value)

    best_score = grid_search.best_score_
    mlflow.log_metric("best_f1_score", best_score)
    mlflow.log_param("model_type", "xgboost")

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision_weighted = precision_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision_weighted", precision_weighted)
    mlflow.log_metric("recall_weighted", recall_weighted)
    mlflow.log_metric("f1_weighted", f1_weighted)

    with tempfile.TemporaryDirectory() as temp_dir:
        train_file_path = os.path.join(temp_dir, "train.parquet")
        test_file_path = os.path.join(temp_dir, "test.parquet")

        train_set.to_parquet(train_file_path, index=False)
        test_set.to_parquet(test_file_path, index=False)

        mlflow.log_artifact(train_file_path, artifact_path="datasets")
        mlflow.log_artifact(test_file_path, artifact_path="datasets")

        cm_file_name = _create_confusion_matrix(y_test, y_pred, labels, temp_dir)
        mlflow.log_artifact(cm_file_name, artifact_path="stats")

    return best_model
