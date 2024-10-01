import os

def get_path_last_version(model_type: str) -> str:
    path = f"./data/ml/{model_type}"
    versions = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    versions.sort()
    last_version = versions[-1]
    return f"{path}/{last_version}/model.xgb"