import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImagePairDataset(Dataset):
    def __init__(self, image_dir, correspondence_dir, dataset_name='Custom', transform=None):
        self.image_dir = image_dir
        self.correspondence_dir = correspondence_dir
        self.dataset_name = dataset_name
        self.transform = transform
        self.pairs = self._load_pairs()

    def _load_pairs(self):
        pairs = []
        for corr_file in os.listdir(self.correspondence_dir):
            if corr_file.endswith('.txt'):
                base_name = corr_file.replace('.txt', '')
                img1_name = f"{base_name}_1.png"
                img2_name = f"{base_name}_2.png"
                pairs.append((img1_name, img2_name, corr_file))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_name, img2_name, corr_file = self.pairs[idx]
        img1_path = os.path.join(self.image_dir, img1_name)
        img2_path = os.path.join(self.image_dir, img2_name)
        corr_path = os.path.join(self.correspondence_dir, corr_file)

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        else:
            img1 = transforms.ToTensor()(img1)
            img2 = transforms.ToTensor()(img2)
        
        correspondences = np.loadtxt(corr_path, delimiter=' ')
        correspondences = torch.from_numpy(correspondences).float()

        # Create uniform depth maps
        depth0 = torch.ones((img1.shape[1], img1.shape[2])) * 2.0  # uniform depth map for img1
        depth1 = torch.ones((img2.shape[1], img2.shape[2])) * 2.0  # uniform depth map for img2

        # Create dummy transformation matrices
        T_0to1 = torch.eye(4)
        T_1to0 = torch.eye(4)

        # Dummy intrinsics (identity matrices)
        K0 = torch.eye(3)
        K1 = torch.eye(3)

        return {
            'image0': img1,
            'image1': img2,
            'correspondences': correspondences,
            'depth0': depth0,
            'depth1': depth1,
            'dataset_name': self.dataset_name,
            'T_0to1': T_0to1,
            'T_1to0': T_1to0,
            'K0': K0,
            'K1': K1
        }


def get_dataloader(image_dir, correspondence_dir, batch_size, num_workers, transform=None):
    dataset = ImagePairDataset(image_dir, correspondence_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader


def get_local_split(items: list, world_size: int, rank: int, seed: int):
    """ The local rank only loads a split of the dataset. """
    n_items = len(items)
    items_permute = np.random.RandomState(seed).permutation(items)
    if n_items % world_size == 0:
        padded_items = items_permute
    else:
        padding = np.random.RandomState(seed).choice(
            items,
            world_size - (n_items % world_size),
            replace=True)
        padded_items = np.concatenate([items_permute, padding])
        assert len(padded_items) % world_size == 0, \
            f'len(padded_items): {len(padded_items)}; world_size: {world_size}; len(padding): {len(padding)}'
    n_per_rank = len(padded_items) // world_size
    local_items = padded_items[n_per_rank * rank: n_per_rank * (rank + 1)]

    return local_items
