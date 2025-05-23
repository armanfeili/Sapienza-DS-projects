from dataclasses import dataclass
from typing import Tuple
import json


@dataclass
class VisualConfig:
    def to_json(self, path: str):
        with open(path, "w") as fp:
            json.dump(self.__dict__, fp, indent=2)

    @classmethod
    def from_json(self, path: str):
        with open(path, "r") as fp:
            json_obj = json.load(fp)

        return VisualConfig(**json_obj)

    # General
    project_name: str = "project"
    random_state: int = 42
    device: str = "cuda"
    seed: int = 42

    # Model
    num_classes: int = 4
    model_name: str = "resnet18"
    pretrained: bool = True

    # Preprocessing
    train_input_size: Tuple[int, int] = (224, 224)
    test_input_size: Tuple[int, int] = (224, 224)

    aug_color_jitter_b: float = 0.1
    aug_color_jitter_c: float = 0.1
    aug_color_jitter_s: float = 0.1

    norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    norm_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Datase
    # # If fold = -1 no evaluation will be made and all the train dataset will be considered
    fold: int = 0
    csv_train_file: str = "dataset/train.csv"
    csv_test_file: str = "dataset/test.csv"
    csv_split_file: str = "dataset/fold.csv"
    root_train_images: str = "dataset/images/train"
    root_test_images: str = "dataset/images/test"

    # Optimizer
    num_epochs: int = 15
    batch_size: int = 32
    test_batch_size: int = 64
    num_workers: int = 0

    lr: float = 1.0e-3
    weight_decay: float = 1e-4
