import torch
from aim.data.builder import build_dataloader, build_dataset

imagenet_train_cfg = {
    'type': 'torchvision.transforms.Compose',
    'transforms': [
        {'type': 'timm.data.RandomResizedCropAndInterpolation',
         'size': 224, 'scale': (0.08, 1.0), 'ratio': (3. / 4., 4. / 3.), 'interpolation': 'random'},
        {'type': 'torchvision.transforms.RandomHorizontalFlip', 'p': 0.5},
        {'type': 'torchvision.transforms.ColorJitter', 'brightness': 0.4, 'contrast': 0.4, 'saturation': 0.4},
        {'type': 'torchvision.transforms.ToTensor'},
        {'type': 'torchvision.transforms.Normalize', 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    ]
}

imagenet_dataset_train = {
    'type': 'timm.data.ImageDataset',
    'root': '/home/stephen/Downloads/imagenet',
    'split': 'train',
    'transform': imagenet_train_cfg
}

loader_cfg = {
    'type': torch.utils.data.DataLoader,
    'dataset': imagenet_dataset_train,
    'batch_size': 32,
    'num_workers': 1,
    'pin_memory': False,
    'persistent_workers': True,
    # 'shuffle': True,
    # Enable `RandomSampler`, the same as set shuffle=True
    'sampler': {
        'type': 'torch.utils.data.RandomSampler',
        'data_source': imagenet_dataset_train,
    },
    'collate_fn': None,
    'timm_prefetcher': {
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
        'channels': 3,
        'device': torch.device('cuda'),
        'img_dtype': torch.float32,
        're_prob': 0.0,
        're_mode': 'const',
        're_count': 1,
        're_num_splits': 0
    },
}

if __name__ == "__main__":
    loader = build_dataloader(loader_cfg)
    print(loader)
    for img, label in loader:
        print(img.shape)
        print(label)
        break
