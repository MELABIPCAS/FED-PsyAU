"""默认超参数。"""

import argparse
from dataclasses import dataclass

from models.dpk_gat import AU_NAMES

# AU 标签向量对应的 12 个 AU
LABEL_MAPPING = {au: index for index, au in enumerate(AU_NAMES)}


@dataclass
class Config:
    excel_path: str = "../data_coding/DFME_data_coding.xlsx"
    data_path: str = "../data/dfme_augment"

    num_classes: int = 7
    batch_size: int = 512
    learning_rate: float = 5e-5
    epochs: int = 25
    eval_interval: int = 1
    test_ratio: float = 0.3
    seed: int = 1
    device: str = "cuda"
    runs: int = 10

    alpha1: float = 1.0
    alpha2: float = 0.5
    alpha3: float = 0.5

    checkpoint: str = "../experiments/fed_psyau.pth"


def parse_args(description):
    defaults = Config()
    parser = argparse.ArgumentParser(description=description)
    for name, value in vars(defaults).items():
        parser.add_argument(f"--{name}", type=type(value), default=value)
    return Config(**vars(parser.parse_args()))
