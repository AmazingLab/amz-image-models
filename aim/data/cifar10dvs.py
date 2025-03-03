import numpy as np
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS as SJCIFAR10DVS
from tqdm import tqdm
from torch.utils.data import Dataset

__all__ = ['CIFAR10DVS']
GLOBAL_DVS_DATASET = None
TRAIN_INDICES = None
TEST_INDICES = None


def load_cifar10dvs_spikingjelly(data_prefix, use_train, time_step, data_type='frame', split_by='number'):
    global GLOBAL_DVS_DATASET
    print(f'[INFO] [AMZCLS] Processing {"training" if use_train else "testing"} dataset...')
    if GLOBAL_DVS_DATASET is None:
        GLOBAL_DVS_DATASET = SJCIFAR10DVS(
            root=data_prefix,
            data_type=data_type,
            frames_number=time_step,
            split_by=split_by)
        print(f'[INFO] [AMZCLS] Loading {"training" if use_train else "testing"} dataset...')
    return GLOBAL_DVS_DATASET


def cifar10dvs_indices(train_ratio: float, use_train: bool, shuffle: bool = True):
    global TEST_INDICES
    global TRAIN_INDICES
    if (TRAIN_INDICES is None) and (TEST_INDICES is None):
        indices = [[i for i in range(j * 1000, (j + 1) * 1000)] for j in range(10)]
        if shuffle:
            for index in indices:
                np.random.seed(index)
                np.random.shuffle(index)
            # ========================== v0 ==========================
            TEST_INDICES = [indices[j][int(1000 * train_ratio):] for j in range(10)]
            TRAIN_INDICES = [indices[j][:int(1000 * train_ratio)] for j in range(10)]
        else:
            # ======================== From PSN =======================
            split_index = int(1000 * (1. - train_ratio))
            TEST_INDICES = [indices[j][:split_index] for j in range(10)]
            TRAIN_INDICES = [indices[j][split_index:] for j in range(10)]
            raise NotImplementedError

    return TRAIN_INDICES if use_train else TEST_INDICES


def load_npz(file_name):
    return np.load(file_name, allow_pickle=True)['frames'].astype(np.float32)


class CIFAR10DVS(Dataset):
    def __init__(self, root, use_train, time_step=16, data_type='frame', split_by='number', train_ratio=0.9,
                 shuffle=True, pre_load_to_memory=False, transform=None, target_transform=None, **kwargs):
        super().__init__()
        dvs_dataset = load_cifar10dvs_spikingjelly(root, use_train, time_step, data_type, split_by)
        self.transform = transform
        self.target_transform = target_transform
        self.data_type = data_type
        self.pre_load_to_memory = pre_load_to_memory
        self.images = []
        self.targets = []
        for class_indices in cifar10dvs_indices(train_ratio, use_train, shuffle):
            for index in tqdm(class_indices):
                sample = dvs_dataset.samples[index]
                self.images.append(load_npz(sample[0]) if pre_load_to_memory else sample[0]),
                self.targets.append(sample[1])

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
