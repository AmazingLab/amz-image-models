timm_noaug_train_cfg = {
    'type': 'timm.data.transforms_factory.transforms_noaug_train',
    'img_size': 224,
    'interpolation': 'bilinear',
    'mean': (0.485, 0.456, 0.406),
    'std': (0.229, 0.224, 0.225),
    'use_prefetcher': False,
    'normalize': True,
}

timm_imagenet_train_cfg = {
    'type': 'timm.data.transforms_factory.transforms_imagenet_train',
    'img_size': 224,
    'scale': None,
    'ratio': None,
    'train_crop_mode': None,
    'hflip': 0.5,
    'vflip': 0.0,
    'color_jitter': 0.4,
    'color_jitter_prob': None,
    'force_color_jitter': False,
    'grayscale_prob': 0.0,
    'gaussian_blur_prob': 0.0,
    'auto_augment': None,
    'interpolation': 'random',
    'mean': (0.485, 0.456, 0.406),  # 替换 IMAGENET_DEFAULT_MEAN
    'std': (0.229, 0.224, 0.225),  # 替换 IMAGENET_DEFAULT_STD
    're_prob': 0.0,
    're_mode': 'const',
    're_count': 1,
    're_num_splits': 0,
    'use_prefetcher': False,
    'normalize': True,
    'separate': False,
}

timm_imagenet_eval_cfg = {
    'type': 'timm.data.transforms_factory.transforms_imagenet_eval',
    'img_size': 224,
    'crop_pct': None,  # 函数内部默认用 DEFAULT_CROP_PCT，但配置保留用户传入的默认值 None
    'crop_mode': None,  # 函数内部默认用 'center'，但配置保持定义时的默认值 None
    'crop_border_pixels': None,
    'interpolation': 'bilinear',
    'mean': (0.485, 0.456, 0.406),  # 替换 IMAGENET_DEFAULT_MEAN
    'std': (0.229, 0.224, 0.225),  # 替换 IMAGENET_DEFAULT_STD
    'use_prefetcher': False,
    'normalize': True,
}

noaug_train_cfg = {
    'type': 'torchvision.transforms.Compose',
    'transforms': [
        {'type': 'torchvision.transforms.Resize', 'size': 224},
        {'type': 'torchvision.transforms.CenterCrop', 'size': 224},
        # Enable this and disable Normalize if using timm PrefetchLoader
        # in timm.data.loader line 76 and timm.data.transforms_factory line 47
        # {'type': 'torchvision.transforms.ToNumpy'},
        {'type': 'torchvision.transforms.ToTensor'},
        {'type': 'torchvision.transforms.Normalize', 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
    ]
}

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

imagenet_eval_cfg = {
    'type': 'torchvision.transforms.Compose',
    'transforms': [
        {'type': 'torchvision.transforms.Resize', 'size': 256, 'interpolation': 'bilinear'},
        {'type': 'torchvision.transforms.CenterCrop', 'size': 224},
        {'type': 'torchvision.transforms.ToTensor'},
        {'type': 'torchvision.transforms.Normalize', 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    ]
}
