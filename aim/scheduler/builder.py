from typing import Union

from timm.scheduler.scheduler import Scheduler
from torch.optim.lr_scheduler import LRScheduler

from aim.engine import recursive_build


def build_scheduler(cfg: dict, optimizer: Union[
    LRScheduler, Scheduler
]):
    cfg['optimizer'] = optimizer
    return recursive_build(cfg)
