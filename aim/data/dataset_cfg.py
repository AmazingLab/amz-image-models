import timm
from timm.data.dataset import ImageDataset
# from timm.data.dataset_factory import create_dataset
from torchvision.datasets import *
from .transforms_cfg import (
    imagenet_eval_cfg, imagenet_train_cfg, timm_imagenet_eval_cfg, timm_imagenet_train_cfg
)

timm_imagenet_dataset_train = {
    'type': 'timm.data.create_dataset',
    'name': 'torch/imagenet',
    'root': '~/Downloads/imagenet',
    'split': 'train',
    # TIMM will create transforms in loader
    # in timm.data.loader line 287
    # 'transform': timm_imagenet_train_cfg
}

timm_imagenet_dataset_val = {
    'type': 'timm.data.create_dataset',
    'name': 'torch/imagenet',
    'root': '~/Downloads/imagenet',
    'split': 'val',
    # TIMM will create transforms in loader
    # in timm.data.loader line 287
    # 'transform': timm_imagenet_eval_cfg
}

imagenet_dataset_train = {
    'type': 'torchvision.datasets.ImageNet',
    'root': '~/Downloads/imagenet',
    'split': 'train',
    'transform': imagenet_train_cfg
}

imagenet_dataset_val = {
    'type': 'torchvision.datasets.ImageNet',
    'root': '~/Downloads/imagenet',
    'split': 'val',
    'transform': imagenet_eval_cfg
}
