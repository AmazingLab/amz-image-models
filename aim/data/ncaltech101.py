import os

import numpy as np
from spikingjelly.datasets.n_caltech101 import NCaltech101 as SJNCaltech101
from tqdm import tqdm
from torch.utils.data import Dataset

__all__ = ['NCaltech101']
GLOBAL_DVS_DATASET = None


def load_ncaltech101_spikingjelly(data_prefix, test_mode, time_step, data_type='frame', split_by='number'):
    global GLOBAL_DVS_DATASET
    if GLOBAL_DVS_DATASET is None:
        GLOBAL_DVS_DATASET = SJNCaltech101(
            root=data_prefix,
            data_type=data_type,
            frames_number=time_step,
            split_by=split_by)
        print(f'[INFO] [AMZCLS] Processing {"testing" if test_mode else "training"} dataset...')
    return GLOBAL_DVS_DATASET


def load_npz(file_name, data_type='frames'):
    return np.load(file_name, allow_pickle=True)[data_type].astype(np.float32)


# implement without checkpoint
def process_dataset(dvs_dataset, use_train, split_rate=0.9, shuffle=True, pre_load_to_memory=False):
    filepath = dvs_dataset.root
    class_list = os.listdir(filepath)
    class_list.sort()
    image_list = []
    label_list = []
    for class_index, class_name in enumerate(class_list):
        file_list = os.listdir(os.path.join(filepath, class_name))
        if shuffle:
            np.random.seed(class_index)
            np.random.shuffle(file_list)
        split = int(len(file_list) * split_rate)
        file_split = file_list[:split] if use_train else file_list[split:]
        for file in file_split:
            image_path = os.path.join(filepath, class_name, file)
            image_list.append(load_npz(image_path) if pre_load_to_memory else image_path)
            label_list.append(np.array(class_index, dtype=np.int64))
    return image_list, label_list


class NCaltech101(Dataset):
    def __init__(self, root, use_train, time_step=16, data_type='frame', split_by='number', train_ratio=0.9,
                 shuffle=True, pre_load_to_memory=False, transform=None, target_transform=None, **kwargs):
        super().__init__()
        dvs_dataset = load_ncaltech101_spikingjelly(root, use_train, time_step, data_type, split_by)
        self.transform = transform
        self.target_transform = target_transform
        self.data_type = data_type
        self.pre_load_to_memory = pre_load_to_memory
        self.images, self.targets = process_dataset(dvs_dataset, use_train, train_ratio, shuffle, pre_load_to_memory)

        print(f'[INFO] [AMZCLS] Unused parameters {kwargs}')

    def __len__(self):
        assert len(self.images) == len(self.targets)
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.images[idx], self.targets[idx]
        image = image if self.pre_load_to_memory else load_npz(image)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label
