import random
from numbers import Number
from typing import List, Tuple, Optional, Dict
from typing import Union, Sequence

import math
import numpy as np
import torch
from timm.data.transforms import (
    ToNumpy, ToTensor, str_to_interp_mode, str_to_pil_interp, interp_mode_to_str,
    RandomResizedCropAndInterpolation, CenterCropOrPad, center_crop_or_pad, crop_or_pad,
    RandomCropOrPad, RandomPad, ResizeKeepRatio, TrimBorder, MaybeToTensor, MaybePILToTensor
)
from torch import Tensor
from torchvision.transforms import (
    RandomHorizontalFlip, InterpolationMode, RandomCrop, Resize, RandomErasing
)
from torchvision.transforms import functional as F

__all__ = [
    "ToFloatTensor", "RandomHorizontalFlipDVS", "ResizeDVS", "RandomCropDVS", "NeuromorphicDataAugmentation",
    "TimeSample", "RandomTimeShuffle", "RandomSliding", "RandomCircularSliding"
]


def to_float_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return torch.FloatTensor(data)
    elif isinstance(data, np.ndarray):
        return torch.FloatTensor(torch.from_numpy(data.copy()))
    elif isinstance(data, int):
        return torch.FloatTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(
            f'Type {type(data)} cannot be converted to tensor.'
            'Supported types are: `numpy.ndarray`, `torch.Tensor`, '
            '`Sequence`, `int` and `float`')


class ToFloatTensor(object):

    def __call__(self, data):
        return to_float_tensor(data)


class RandomHorizontalFlipDVS(object):

    def __init__(self, prob=0.5):
        self.flip = RandomHorizontalFlip(p=prob)

    def __call__(self, data):
        return self.flip(data)


class ResizeDVS(object):

    def __init__(self, scale=(48, 48),
                 interpolation=InterpolationMode.BILINEAR):
        self.resize = Resize(scale, interpolation, antialias=True)

    def __call__(self, data):
        return self.resize(data)


class RandomCropDVS(object):

    def __init__(self,
                 crop_size: Union[Sequence, int],
                 padding: Optional[Union[Sequence, int]] = None,
                 pad_if_needed: bool = False,
                 pad_val: Union[Number, Sequence[Number]] = 0,
                 padding_mode: str = 'constant'):
        self.tv_rand_crop = RandomCrop(crop_size, padding, pad_if_needed, pad_val, padding_mode)

    def __call__(self, data) -> dict:
        return self.tv_rand_crop(data)


class NeuromorphicDataAugmentation(torch.nn.Module):
    def __init__(self, num_magnitude_bins: int = 31, interpolation: InterpolationMode = InterpolationMode.NEAREST,
                 fill: Optional[List[float]] = None, augmentation_space: dict = None) -> None:
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill
        self.cutout = RandomErasing(p=1, scale=(0.001, 0.11), ratio=(1, 1))  # todo cutout N holes
        # https://github.com/ChaotengDuan/TEBN/blob/8f82f6a307093faddeb87127aa432a66d65352ea/dataloader.py#L8
        self.augmentation_space = augmentation_space

    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        if self.augmentation_space is None:
            self.augmentation_space = {
                # op_name: (magnitudes, signed)
                "Identity": (torch.tensor(0.0), False),
                "ShearX": (torch.linspace(-0.3, 0.3, num_bins), True),
                "ShearY": (torch.linspace(-0.3, 0.3, num_bins), True),
                "TranslateX": (torch.linspace(-5.0, 5.0, num_bins), True),
                "TranslateY": (torch.linspace(-5.0, 5.0, num_bins), True),
                "Rotate": (torch.linspace(-30.0, 30.0, num_bins), True),
                "Cutout": (torch.linspace(1.0, 30.0, num_bins), True)
            }
        return self.augmentation_space

    def _apply_op(self, img: Tensor, op_name: str, magnitude: float):

        if op_name == "ShearX":
            img = F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[math.degrees(magnitude), 0.0],
                           interpolation=self.interpolation, fill=self.fill)
        elif op_name == "ShearY":
            img = F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[0.0, math.degrees(magnitude)],
                           interpolation=self.interpolation, fill=self.fill)
        elif op_name == "TranslateX":
            img = F.affine(img, angle=0.0, translate=[int(magnitude), 0], scale=1.0,
                           interpolation=self.interpolation, shear=[0.0, 0.0], fill=self.fill)
        elif op_name == "TranslateY":
            img = F.affine(img, angle=0.0, translate=[0, int(magnitude)], scale=1.0,
                           interpolation=self.interpolation, shear=[0.0, 0.0], fill=self.fill)
        elif op_name == "Rotate":
            img = F.rotate(img, magnitude, interpolation=self.interpolation)
        elif op_name == "Identity":
            pass
        else:
            raise ValueError("The provided operator {} is not recognized.".format(op_name))
        return img

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        op_meta = self._augmentation_space(self.num_magnitude_bins)
        op_index = int(torch.randint(len(op_meta), (1,)).item())
        op_name = list(op_meta.keys())[op_index]
        magnitudes, signed = op_meta[op_name]
        magnitude = float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item()) \
            if magnitudes.ndim > 0 else 0.0
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0
        if op_name == "Cutout":
            return self.cutout(img)
        else:
            return self._apply_op(img, op_name, magnitude)


class TimeSample(object):

    def __init__(self, time_step: int, sample_step: int, use_rand=True):
        self.time_step = time_step
        self.sample_step = sample_step
        self.use_rand = use_rand

    def __call__(self, data):
        sample_step = random.randint(self.sample_step, self.time_step) if self.use_rand else self.sample_step
        indices = random.sample(
            range(self.time_step), sample_step
        )
        indices.sort()

        data = data[indices]

        if self.use_rand:
            zero = np.zeros((self.time_step - sample_step, *data.shape[1:]), dtype=data.dtype)
            data = np.concatenate((data, zero), axis=0)

        return data


class RandomTimeShuffle(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        if np.random.rand(1) > self.p:
            return data

        time_step = data.shape[0]
        indices = np.arange(time_step)
        np.random.shuffle(indices)
        data = data[indices]

        return data


class RandomSliding(object):
    """
        {x: [1, 2, 3, 4, 5], sliding: +2} -> [3, 4, 5, 0, 0]
        {x: [1, 2, 3, 4, 5], sliding: -2} -> [0, 0, 1, 2, 3]
    """

    def __init__(self, max_sliding: int, p=0.5):
        self.max_sliding = max_sliding
        self.p = p

    def __call__(self, data):
        if np.random.rand(1) > self.p:
            return data

        sliding = np.random.randint(-self.max_sliding, self.max_sliding + 1)
        front_part = data[:sliding]
        rear_part = data[sliding:]
        if sliding == 0:
            return data
        elif sliding > 0:
            data = np.concatenate((rear_part, np.zeros_like(front_part)), axis=0)
        else:
            data = np.concatenate((np.zeros_like(rear_part), front_part), axis=0)

        return data


class RandomCircularSliding(object):
    """
        {x: [1, 2, 3, 4, 5], sliding: +2} -> [3, 4, 5, 1, 2]
        {x: [1, 2, 3, 4, 5], sliding: -2} -> [4, 5, 1, 2, 3]
    """

    def __init__(self, max_sliding: int, p=0.5):
        self.max_sliding = max_sliding
        self.p = p

    def __call__(self, data):
        if np.random.rand(1) > self.p:
            return data

        sliding = np.random.randint(
            -self.max_sliding, self.max_sliding + 1)
        data = np.concatenate(
            (data[sliding:], data[:sliding]), axis=0
        )

        return data
