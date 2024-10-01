"""
This is a boilerplate pipeline 'train_wine_model'
generated using Kedro 0.19.8
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import get_data, split_train_test, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(inputs=None, func=get_data, outputs="dataset"),
            node(
                func=split_train_test,
                inputs=["dataset", "params:model_inputs"],
                outputs=["train_set", "test_set"],
            ),
            node(
                func=train_model,
                inputs=["train_set", "test_set", "params:model_inputs"],
                outputs="classifier",
            ),
        ]
    )
