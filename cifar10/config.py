from . import PROJECT_ROOT
import os
from dataclasses import dataclass, field
import yaml

@dataclass
class Config:
    model_name: str = field(default="model")
    num_epochs: int = field(default=None)
    batch_size: int = field(default=None)
    initial_lr: float = field(default=None)
    num_layers: list = field(default=None)
    data_dir: str = field(default="data")
    csv_path: str = field(default="loss_and_error")
    plot_path: str = field(default="loss_and_error")
    weight_dir: str = field(default="weights")
    weight_path: str = field(default=None)
    suffix: str = field(default="")

    def __post_init__(self):
        if self.data_dir:
            self.data_dir = os.path.join(PROJECT_ROOT, self.data_dir)
        if self.weight_dir:
            self.weight_dir = os.path.join(PROJECT_ROOT, self.weight_dir)


def load_config(path):
    yaml_path = os.path.join(PROJECT_ROOT, path)

    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    if not config_dict:
        raise ValueError("Config file is empty.")

    error_log = []

    if config_dict.get("num_epochs") is None:
        error_log.append("`num_epochs` must be provided in config file.")
    if config_dict.get("batch_size") is None:
        error_log.append("`batch_size` must be provided in config file.")
    if config_dict.get("initial_lr") is None:
        error_log.append("`initial_lr` must be provided in config file.")

    if error_log:
        raise ValueError("\n\t".join([""] + error_log))

    config = Config(**config_dict)

    print(config)

    return config
