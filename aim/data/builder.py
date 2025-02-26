from typing import Union

from timm.data.distributed_sampler import OrderedDistributedSampler
from torchvision.transforms import transforms

from timm.data.loader import fast_collate, PrefetchLoader
from torch.utils.data import Dataset, Sampler, DataLoader

from aim.engine import recursive_build, build, resolve_dict


def build_transforms(cfg: dict):
    """
    >>> transform_cfg = {
    >>>     'type': 'torchvision.transforms.Compose',
    >>>     'transforms': [
    >>>         {'type': 'torchvision.transforms.ToTensor'},
    >>>         {
    >>>             'type': 'torchvision.transforms.Normalize',
    >>>             'mean': [0.485, 0.456, 0.406],
    >>>             'std': [0.229, 0.224, 0.225]
    >>>         }
    >>>     ]
    >>> }
    >>> transform = build_transforms(transform_cfg)
    >>> print(transform)
    """
    return recursive_build(cfg)


def build_dataset(cfg: dict) -> Dataset:
    """
    >>> from torchvision.datasets import MNIST
    >>> dataset_cfg = {
    >>>     'type': MNIST,
    >>>     'root': '~/Downloads/',
    >>>     'train': True,
    >>>     'download': True,
    >>>     'transform': {
    >>>         'type': 'torchvision.transforms.ToTensor'
    >>>     }
    >>> }
    >>> dataset = build_dataset(dataset_cfg)
    >>> print(dataset)
    :param cfg:
    :return:
    """
    return recursive_build(cfg)


def build_sampler(cfg: dict, dataset: Union[Dataset, dict]) -> Sampler:
    """
    >>> # Example 1: Build distributed sampler (using dataset config)
    >>> dataset_cfg = {
    >>>     'type': 'torchvision.datasets.MNIST',
    >>>     'root': '~/Downloads/',
    >>>     'train': True,
    >>>     'download': True,
    >>>     'transform': {'type': 'torchvision.transforms.ToTensor'}
    >>> }
    >>> sampler_cfg = {
    >>>     'type': 'torch.utils.data.distributed.DistributedSampler',
    >>>     'shuffle': True  # Controls whether to shuffle the data
    >>> }
    >>> sampler = build_sampler(sampler_cfg, dataset_cfg)
    >>> print(sampler)
    >>>
    >>> # Example 2: Build random sampler (using pre-built dataset instance)
    >>> from torch.utils.data import RandomSampler
    >>> dataset = build_dataset(dataset_cfg)  # Assuming dataset is built via build_dataset
    >>> sampler_cfg = {
    >>>     'type': RandomSampler,
    >>>     'replacement': True,    # Allows sampling with replacement
    >>>     'num_samples': 1000     # Specifies number of samples to draw
    >>> }
    >>> sampler = build_sampler(sampler_cfg, dataset)
    >>> print(sampler)
    """
    dataset = recursive_build(dataset)
    name, args, kwargs = resolve_dict(cfg)
    # Torch RandomSampler(data_source, ...)
    # Torch DistributedSampler(dataset, ...)
    # TIMM OrderedDistributedSampler(dataset, ...)
    return build(name, dataset, *args, **kwargs)


def build_dataloader(cfg: dict, dataset: Dataset) -> Union[DataLoader, PrefetchLoader]:
    """
    Builds a DataLoader with flexible configuration support, including timm prefetcher.

    >>> # Example 1: Standard PyTorch DataLoader
    >>> dataset = build_dataset(dataset_cfg)  # Assume dataset is pre-built
    >>> loader_cfg = {
    >>>     'type': 'torch.utils.data.DataLoader',
    >>>     'batch_size': 32,
    >>>     'sampler': {
    >>>         'type': 'torch.utils.data.RandomSampler',
    >>>         'replacement': False
    >>>     },
    >>>     'num_workers': 4,
    >>>     'timm_prefetcher': None  # Disable prefetcher
    >>> }
    >>> dataloader = build_dataloader(loader_cfg, dataset)
    >>>
    >>> # Example 2: Using timm PrefetchLoader with fast collate
    >>> prefetch_cfg = {
    >>>     'type': 'torch.utils.data.DataLoader',
    >>>     'batch_size': 128,
    >>>     'sampler': {
    >>>         'type': 'torch.utils.data.SequentialSampler'
    >>>     },
    >>>     'timm_prefetcher': {
    >>>         'num_workers': 4,
    >>>         'pin_memory': True
    >>>     }
    >>> }
    >>> prefetch_loader = build_dataloader(prefetch_cfg, dataset)

    Key Features:
    1. Automatic sampler building: The 'sampler' config will be resolved using build_sampler
    2. Prefetcher integration: When 'timm_prefetcher' is specified:
       - Uses timm's fast_collate as collate_fn
       - Wraps the loader with PrefetchLoader
    3. Configuration inheritance: Original DataLoader params (batch_size, num_workers, etc.)
       are preserved when using prefetcher
    """
    cfg['dataset'] = dataset
    cfg['sampler'] = build_sampler(cfg['sampler'], dataset)
    timm_prefetcher = cfg.pop('timm_prefetcher')
    if timm_prefetcher is not None:
        cfg['collate_fn'] = fast_collate
        dataloader = recursive_build(cfg)
        return PrefetchLoader(loader=dataloader, **timm_prefetcher)

    return recursive_build(cfg)
