from datetime import datetime
from pydantic import BaseModel
from typing import Literal, Any
import pickle


def get_datetime_str(format="%Y-%m-%d_%H-%M-%S"):
    return datetime.now().strftime(format)


def pickle_dump(obj: Any, filename: str):
    with open(filename, "wb") as fp:
        pickle.dump(obj, fp)


def pickle_load(filename: str):
    with open(filename, "rb") as fp:
        obj = pickle.load(fp)

    return obj


class BaseModelConfig(BaseModel):
    model_name: str = ""
    repo_name: str = ""

    test_transform: Any = None


class BaseTrainingConfig(BaseModel):
    # General
    device: str = "cuda:0"
    random_state: int = 42
    project_name: str = "project"

    # Dataset Loader
    batch_size: int = 32
    test_batch_size: int = 64
    num_workers: int = 4

    # Optimizer
    num_epochs: int = 100
    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler: Literal["cos", None] = "cos"
    lr_warmup_epochs: int = 0

    train_transform: Any = None
