timm_imagenet_dataloader_train = {

}

dataloader_train = {
    'type': 'torch.utils.data.DataLoader',
    'batch_size': 128,
    'sampler': {
        'type': 'torch.utils.data.SequentialSampler'
    },
    'timm_prefetcher': {
        'num_workers': 4,
        'pin_memory': True
    }
}

create_loader_cfg = {
    'type': 'torch.utils.data.DataLoader',
    'batch_size': 32,  # 必填参数，需用户指定
    'num_workers': 1,
    'pin_memory': False,
    'persistent_workers': True,
    'sampler': {'type': 'torch.utils.data.SequentialSampler'},
    'distributed': False,
    'timm_prefetcher': {
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
        'channels': 3,
        'device': 'cuda',
        'img_dtype': 'torch.float32',
        're_prob': 0.0,
        're_mode': 'const',
        're_count': 1,
        're_num_splits': 0
    },
    'num_aug_repeats': 0,
    'num_aug_splits': 0,
    'collate_fn': None,
    'use_multi_epochs_loader': False,
    'worker_seeding': 'all'
}
