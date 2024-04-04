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
    milestones: list = field(default_factory=lambda: [0.5, 0.75])
    num_layers: list = field(default=None)
    channels: list = field(default=None)
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

    def validate(self):
        error_log = []

        if self.num_epochs is None:
            error_log.append("`num_epochs` must be provided in config file.")

        if self.num_epochs < 0:
            error_log.append("`num_epochs` must be a positive integer.")

        if self.batch_size is None:
            error_log.append("`batch_size` must be provided in config file.")

        if self.batch_size < 0:
            error_log.append("`batch_size` must be a positive integer.")

        if self.initial_lr is None:
            error_log.append("`initial_lr` must be provided in config file.")

        if self.initial_lr <= 0:
            error_log.append("`initial_lr` must be a positive float.")

        for i in range(1, len(self.milestones)):
            if self.milestones[i-1] >= self.milestones[i]:
                error_log.append("`milestones` must be monotoniously increasing.")

        if self.channels is None:
            error_log.append("`channels` must be provided in config file.")

        if self.num_layers is not None and len(self.channels) != len(self.num_layers):
            error_log.append("`channels` and `num_layers` must have the same length.")

        return error_log

def load_config(path):
    yaml_path = os.path.join(PROJECT_ROOT, path)

    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    if not config_dict:
        raise ValueError("Config file is empty.")

    config = Config(**config_dict)

    error_log = config.validate()

    if error_log:
        raise ValueError("\n\t".join([""] + error_log))

    print(config)

    return config
