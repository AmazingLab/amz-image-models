

timm_imagenet_dataloader_train = {
    'type': 'timm.data.create_dataset',
    'name': 'torch/imagenet',
    'root': '~/Downloads/imagenet',
    'split': 'train',
    # TIMM will create transforms in loader
    # in timm.data.loader line 287
    # 'transform': timm_imagenet_train_cfg
}


import torch
import timm
def create_loader(
        dataset: Union[ImageDataset, IterableImageDataset],
        batch_size: int,
        is_training: bool = False,
        re_split: bool = False,
        num_aug_repeats: int = 0,
        num_aug_splits: int = 0,
        num_workers: int = 1,
        distributed: bool = False,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        use_prefetcher: bool = True,
        use_multi_epochs_loader: bool = False,
        persistent_workers: bool = True,
        worker_seeding: str = 'all',
):
    re_num_splits = 0
    if re_split:
        # apply RE to second half of batch if no aug split otherwise line up with aug split
        re_num_splits = num_aug_splits or 2

    if isinstance(dataset, IterableImageDataset):
        # give Iterable datasets early knowledge of num_workers so that sample estimates
        # are correct before worker processes are launched
        dataset.set_loader_cfg(num_workers=num_workers)

    sampler = None
    if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        if is_training:
            if num_aug_repeats:
                sampler = RepeatAugSampler(dataset, num_repeats=num_aug_repeats)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)
    else:
        assert num_aug_repeats == 0, "RepeatAugment not currently supported in non-distributed or IterableDataset use"

    if collate_fn is None:
        collate_fn = fast_collate if use_prefetcher else torch.utils.data.dataloader.default_collate

    loader_class = torch.utils.data.DataLoader
    if use_multi_epochs_loader:
        loader_class = MultiEpochsDataLoader

    loader_args = dict(
        batch_size=batch_size,
        shuffle=not isinstance(dataset, torch.utils.data.IterableDataset) and sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training,
        worker_init_fn=partial(_worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers
    )
    try:
        loader = loader_class(dataset, **loader_args)
    except TypeError as e:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = loader_class(dataset, **loader_args)

    return loader

torch.utils.data.DataLoader