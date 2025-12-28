import numpy as np

import torch
from torch.optim.adamw import AdamW

from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

from introduction_to_world_model.model.world_model import WorldModel
from introduction_to_world_model.env.env import Env


def train_policy_model(
    agent: WorldModel,
    n_episodes: int,
    batch_size: int,
    device: str = "cpu",
    save_path: str | None = None,
):
    # TODO
    ...