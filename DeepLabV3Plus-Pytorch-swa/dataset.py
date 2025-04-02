import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np

class MapSegmentationDataset(Dataset):
    def __init__(self, root, split='train', transforms=None):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.images_dir = os.path.join(root, split, 'images')
        self.masks_dir = os.path.join(root, split, 'masks')
        self.images = list(sorted(os.listdir(self.images_dir)))
        self.masks = list(sorted(os.listdir(self.masks_dir)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transforms is not None:
            image = self.transforms(image)
            mask = np.array(mask)
            mask = self.map_mask_values(mask)
            mask = torch.tensor(mask, dtype=torch.long)

        return image, mask
    
    def map_mask_values(self, mask):
        mask[mask == 100] = 1
        mask[mask == 200] = 2
        return mask

transforms = transforms.Compose([
    transforms.ToTensor()
])
