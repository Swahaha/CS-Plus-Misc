# import os
# import math
# from collections import abc
# from loguru import logger
# from torch.utils.data.dataset import Dataset
# from tqdm import tqdm
# from os import path as osp
# from pathlib import Path
# from joblib import Parallel, delayed

# import pytorch_lightning as pl
# from torch import distributed as dist
# from torch.utils.data import (
#     Dataset,
#     DataLoader,
#     ConcatDataset,
#     DistributedSampler,
#     RandomSampler,
#     dataloader
# )

# from src.utils.augment import build_augmentor
# from src.utils.dataloader import get_local_split
# from src.utils.misc import tqdm_joblib
# from src.utils import comm
# from src.datasets.megadepth import MegaDepthDataset
# from src.datasets.scannet import ScanNetDataset
# from src.datasets.sampler import RandomConcatSampler


# class MultiSceneDataModule(pl.LightningDataModule):
#     """ 
#     For distributed training, each training process is assgined
#     only a part of the training scenes to reduce memory overhead.
#     """
#     def __init__(self, args, config):
#         super().__init__()

#         # 1. data config
#         # Train and Val should from the same data source
#         self.trainval_data_source = config.DATASET.TRAINVAL_DATA_SOURCE
#         self.test_data_source = config.DATASET.TEST_DATA_SOURCE
#         # training and validating
#         self.train_data_root = config.DATASET.TRAIN_DATA_ROOT
#         self.train_pose_root = config.DATASET.TRAIN_POSE_ROOT  # (optional)
#         self.train_npz_root = config.DATASET.TRAIN_NPZ_ROOT
#         self.train_list_path = config.DATASET.TRAIN_LIST_PATH
#         self.train_intrinsic_path = config.DATASET.TRAIN_INTRINSIC_PATH
#         self.val_data_root = config.DATASET.VAL_DATA_ROOT
#         self.val_pose_root = config.DATASET.VAL_POSE_ROOT  # (optional)
#         self.val_npz_root = config.DATASET.VAL_NPZ_ROOT
#         self.val_list_path = config.DATASET.VAL_LIST_PATH
#         self.val_intrinsic_path = config.DATASET.VAL_INTRINSIC_PATH
#         # testing
#         self.test_data_root = config.DATASET.TEST_DATA_ROOT
#         self.test_pose_root = config.DATASET.TEST_POSE_ROOT  # (optional)
#         self.test_npz_root = config.DATASET.TEST_NPZ_ROOT
#         self.test_list_path = config.DATASET.TEST_LIST_PATH
#         self.test_intrinsic_path = config.DATASET.TEST_INTRINSIC_PATH

#         # 2. dataset config
#         # general options
#         self.min_overlap_score_test = config.DATASET.MIN_OVERLAP_SCORE_TEST  # 0.4, omit data with overlap_score < min_overlap_score
#         self.min_overlap_score_train = config.DATASET.MIN_OVERLAP_SCORE_TRAIN
#         self.augment_fn = build_augmentor(config.DATASET.AUGMENTATION_TYPE)  # None, options: [None, 'dark', 'mobile']

#         # MegaDepth options
#         self.mgdpt_img_resize = config.DATASET.MGDPT_IMG_RESIZE  # 840
#         self.mgdpt_img_pad = config.DATASET.MGDPT_IMG_PAD   # True
#         self.mgdpt_depth_pad = config.DATASET.MGDPT_DEPTH_PAD   # True
#         self.mgdpt_df = config.DATASET.MGDPT_DF  # 8
#         self.coarse_scale = 1 / config.LOFTR.RESOLUTION[0]  # 0.125. for training loftr.

#         # 3.loader parameters
#         self.train_loader_params = {
#             'batch_size': args.batch_size,
#             'num_workers': args.num_workers,
#             'pin_memory': getattr(args, 'pin_memory', True)
#         }
#         self.val_loader_params = {
#             'batch_size': 1,
#             'shuffle': False,
#             'num_workers': args.num_workers,
#             'pin_memory': getattr(args, 'pin_memory', True)
#         }
#         self.test_loader_params = {
#             'batch_size': 1,
#             'shuffle': False,
#             'num_workers': args.num_workers,
#             'pin_memory': True
#         }
        
#         # 4. sampler
#         self.data_sampler = config.TRAINER.DATA_SAMPLER
#         self.n_samples_per_subset = config.TRAINER.N_SAMPLES_PER_SUBSET
#         self.subset_replacement = config.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT
#         self.shuffle = config.TRAINER.SB_SUBSET_SHUFFLE
#         self.repeat = config.TRAINER.SB_REPEAT
        
#         # (optional) RandomSampler for debugging

#         # misc configurations
#         self.parallel_load_data = getattr(args, 'parallel_load_data', False)
#         self.seed = config.TRAINER.SEED  # 66

#     def setup(self, stage=None):
#         """
#         Setup train / val / test dataset. This method will be called by PL automatically.
#         Args:
#             stage (str): 'fit' in training phase, and 'test' in testing phase.
#         """

#         assert stage in ['fit', 'test'], "stage must be either fit or test"

#         try:
#             self.world_size = dist.get_world_size()
#             self.rank = dist.get_rank()
#             logger.info(f"[rank:{self.rank}] world_size: {self.world_size}")
#         except AssertionError as ae:
#             self.world_size = 1
#             self.rank = 0
#             logger.warning(str(ae) + " (set wolrd_size=1 and rank=0)")

#         if stage == 'fit':
#             self.train_dataset = self._setup_dataset(
#                 self.train_data_root,
#                 self.train_npz_root,
#                 self.train_list_path,
#                 self.train_intrinsic_path,
#                 mode='train',
#                 min_overlap_score=self.min_overlap_score_train,
#                 pose_dir=self.train_pose_root)
#             # setup multiple (optional) validation subsets
#             if isinstance(self.val_list_path, (list, tuple)):
#                 self.val_dataset = []
#                 if not isinstance(self.val_npz_root, (list, tuple)):
#                     self.val_npz_root = [self.val_npz_root for _ in range(len(self.val_list_path))]
#                 for npz_list, npz_root in zip(self.val_list_path, self.val_npz_root):
#                     self.val_dataset.append(self._setup_dataset(
#                         self.val_data_root,
#                         npz_root,
#                         npz_list,
#                         self.val_intrinsic_path,
#                         mode='val',
#                         min_overlap_score=self.min_overlap_score_test,
#                         pose_dir=self.val_pose_root))
#             else:
#                 self.val_dataset = self._setup_dataset(
#                     self.val_data_root,
#                     self.val_npz_root,
#                     self.val_list_path,
#                     self.val_intrinsic_path,
#                     mode='val',
#                     min_overlap_score=self.min_overlap_score_test,
#                     pose_dir=self.val_pose_root)
#             logger.info(f'[rank:{self.rank}] Train & Val Dataset loaded!')
#         else:  # stage == 'test
#             self.test_dataset = self._setup_dataset(
#                 self.test_data_root,
#                 self.test_npz_root,
#                 self.test_list_path,
#                 self.test_intrinsic_path,
#                 mode='test',
#                 min_overlap_score=self.min_overlap_score_test,
#                 pose_dir=self.test_pose_root)
#             logger.info(f'[rank:{self.rank}]: Test Dataset loaded!')

#     def _setup_dataset(self,
#                        data_root,
#                        split_npz_root,
#                        scene_list_path,
#                        intri_path,
#                        mode='train',
#                        min_overlap_score=0.,
#                        pose_dir=None):
#         """ Setup train / val / test set"""
#         with open(scene_list_path, 'r') as f:
#             npz_names = [name.split()[0] for name in f.readlines()]

#         if mode == 'train':
#             local_npz_names = get_local_split(npz_names, self.world_size, self.rank, self.seed)
#         else:
#             local_npz_names = npz_names
#         logger.info(f'[rank {self.rank}]: {len(local_npz_names)} scene(s) assigned.')
        
#         dataset_builder = self._build_concat_dataset_parallel \
#                             if self.parallel_load_data \
#                             else self._build_concat_dataset
#         return dataset_builder(data_root, local_npz_names, split_npz_root, intri_path,
#                                 mode=mode, min_overlap_score=min_overlap_score, pose_dir=pose_dir)

#     def _build_concat_dataset(
#         self,
#         data_root,
#         npz_names,
#         npz_dir,
#         intrinsic_path,
#         mode,
#         min_overlap_score=0.,
#         pose_dir=None
#     ):
#         datasets = []
#         augment_fn = self.augment_fn if mode == 'train' else None
#         data_source = self.trainval_data_source if mode in ['train', 'val'] else self.test_data_source
#         if str(data_source).lower() == 'megadepth':
#             npz_names = [f'{n}.npz' for n in npz_names]
#         for npz_name in tqdm(npz_names,
#                              desc=f'[rank:{self.rank}] loading {mode} datasets',
#                              disable=int(self.rank) != 0):
#             # `ScanNetDataset`/`MegaDepthDataset` load all data from npz_path when initialized, which might take time.
#             npz_path = osp.join(npz_dir, npz_name)
#             if data_source == 'ScanNet':
#                 datasets.append(
#                     ScanNetDataset(data_root,
#                                    npz_path,
#                                    intrinsic_path,
#                                    mode=mode,
#                                    min_overlap_score=min_overlap_score,
#                                    augment_fn=augment_fn,
#                                    pose_dir=pose_dir))
#             elif data_source == 'MegaDepth':
#                 datasets.append(
#                     MegaDepthDataset(data_root,
#                                      npz_path,
#                                      mode=mode,
#                                      min_overlap_score=min_overlap_score,
#                                      img_resize=self.mgdpt_img_resize,
#                                      df=self.mgdpt_df,
#                                      img_padding=self.mgdpt_img_pad,
#                                      depth_padding=self.mgdpt_depth_pad,
#                                      augment_fn=augment_fn,
#                                      coarse_scale=self.coarse_scale))
#             else:
#                 raise NotImplementedError()
#         return ConcatDataset(datasets)
    
#     def _build_concat_dataset_parallel(
#         self,
#         data_root,
#         npz_names,
#         npz_dir,
#         intrinsic_path,
#         mode,
#         min_overlap_score=0.,
#         pose_dir=None,
#     ):
#         augment_fn = self.augment_fn if mode == 'train' else None
#         data_source = self.trainval_data_source if mode in ['train', 'val'] else self.test_data_source
#         if str(data_source).lower() == 'megadepth':
#             npz_names = [f'{n}.npz' for n in npz_names]
#         with tqdm_joblib(tqdm(desc=f'[rank:{self.rank}] loading {mode} datasets',
#                               total=len(npz_names), disable=int(self.rank) != 0)):
#             if data_source == 'ScanNet':
#                 datasets = Parallel(n_jobs=math.floor(len(os.sched_getaffinity(0)) * 0.9 / comm.get_local_size()))(
#                     delayed(lambda x: _build_dataset(
#                         ScanNetDataset,
#                         data_root,
#                         osp.join(npz_dir, x),
#                         intrinsic_path,
#                         mode=mode,
#                         min_overlap_score=min_overlap_score,
#                         augment_fn=augment_fn,
#                         pose_dir=pose_dir))(name)
#                     for name in npz_names)
#             elif data_source == 'MegaDepth':
#                 # TODO: _pickle.PicklingError: Could not pickle the task to send it to the workers.
#                 raise NotImplementedError()
#                 datasets = Parallel(n_jobs=math.floor(len(os.sched_getaffinity(0)) * 0.9 / comm.get_local_size()))(
#                     delayed(lambda x: _build_dataset(
#                         MegaDepthDataset,
#                         data_root,
#                         osp.join(npz_dir, x),
#                         mode=mode,
#                         min_overlap_score=min_overlap_score,
#                         img_resize=self.mgdpt_img_resize,
#                         df=self.mgdpt_df,
#                         img_padding=self.mgdpt_img_pad,
#                         depth_padding=self.mgdpt_depth_pad,
#                         augment_fn=augment_fn,
#                         coarse_scale=self.coarse_scale))(name)
#                     for name in npz_names)
#             else:
#                 raise ValueError(f'Unknown dataset: {data_source}')
#         return ConcatDataset(datasets)

#     def train_dataloader(self):
#         """ Build training dataloader for ScanNet / MegaDepth. """
#         assert self.data_sampler in ['scene_balance']
#         logger.info(f'[rank:{self.rank}/{self.world_size}]: Train Sampler and DataLoader re-init (should not re-init between epochs!).')
#         if self.data_sampler == 'scene_balance':
#             sampler = RandomConcatSampler(self.train_dataset,
#                                           self.n_samples_per_subset,
#                                           self.subset_replacement,
#                                           self.shuffle, self.repeat, self.seed)
#         else:
#             sampler = None
#         dataloader = DataLoader(self.train_dataset, sampler=sampler, **self.train_loader_params)
#         return dataloader
    
#     def val_dataloader(self):
#         """ Build validation dataloader for ScanNet / MegaDepth. """
#         logger.info(f'[rank:{self.rank}/{self.world_size}]: Val Sampler and DataLoader re-init.')
#         if not isinstance(self.val_dataset, abc.Sequence):
#             sampler = DistributedSampler(self.val_dataset, shuffle=False)
#             return DataLoader(self.val_dataset, sampler=sampler, **self.val_loader_params)
#         else:
#             dataloaders = []
#             for dataset in self.val_dataset:
#                 sampler = DistributedSampler(dataset, shuffle=False)
#                 dataloaders.append(DataLoader(dataset, sampler=sampler, **self.val_loader_params))
#             return dataloaders

#     def test_dataloader(self, *args, **kwargs):
#         logger.info(f'[rank:{self.rank}/{self.world_size}]: Test Sampler and DataLoader re-init.')
#         sampler = DistributedSampler(self.test_dataset, shuffle=False)
#         return DataLoader(self.test_dataset, sampler=sampler, **self.test_loader_params)


# def _build_dataset(dataset: Dataset, *args, **kwargs):
#     return dataset(*args, **kwargs)

# import os
# import numpy as np
# from PIL import Image
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from pytorch_lightning import LightningDataModule

# class ImagePairDataset(Dataset):
#     def __init__(self, image_dir, correspondence_dir, dataset_name='Custom', transform=None):
#         self.image_dir = image_dir
#         self.correspondence_dir = correspondence_dir
#         self.dataset_name = dataset_name
#         self.transform = transform
#         self.pairs = self._load_pairs()

#     def _load_pairs(self):
#         pairs = []
#         for corr_file in os.listdir(self.correspondence_dir):
#             if corr_file.endswith('.txt'):
#                 base_name = corr_file.replace('.txt', '')
#                 img1_name = f"{base_name}_1.png"
#                 img2_name = f"{base_name}_2.png"
#                 pairs.append((img1_name, img2_name, corr_file))
#         return pairs

#     def __len__(self):
#         return len(self.pairs)

#     def __getitem__(self, idx):
#         img1_name, img2_name, corr_file = self.pairs[idx]
#         img1_path = os.path.join(self.image_dir, img1_name)
#         img2_path = os.path.join(self.image_dir, img2_name)
#         corr_path = os.path.join(self.correspondence_dir, corr_file)

#         img1 = Image.open(img1_path).convert("RGB")
#         img2 = Image.open(img2_path).convert("RGB")
        
#         if self.transform:
#             img1 = self.transform(img1)
#             img2 = self.transform(img2)
#         else:
#             img1 = transforms.ToTensor()(img1)
#             img2 = transforms.ToTensor()(img2)
        
#         correspondences = np.loadtxt(corr_path, delimiter=' ')
#         correspondences = torch.from_numpy(correspondences).float()

#         # Create matching matrix from correspondences
#         h, w = img1.shape[1], img1.shape[2]
#         matching_matrix = torch.zeros(h, w, 2)
#         for i in range(correspondences.shape[0]):
#             x0, y0, x1, y1 = correspondences[i]
#             matching_matrix[int(y0), int(x0), :] = torch.tensor([x1, y1])

#         return {
#             'image0': img1,
#             'image1': img2,
#             'correspondences': correspondences,
#             'matching_matrix': matching_matrix,
#             'dataset_name': self.dataset_name
#         }

# class ImagePairDataModule(LightningDataModule):
#     def __init__(self, args, config):
#         super().__init__()
#         self.image_dir = config.DATASET.TRAIN_DATA_ROOT
#         self.correspondence_dir = config.DATASET.CORRESPONDENCES_ROOT
#         self.batch_size = args.batch_size
#         self.num_workers = args.num_workers
#         self.pin_memory = args.pin_memory
#         self.dataset_name = config.DATASET.TRAINVAL_DATA_SOURCE

#         # Define the transform to convert images to grayscale
#         self.transform = transforms.Compose([
#             transforms.Grayscale(num_output_channels=1),
#             transforms.ToTensor(),
#         ])

#     def setup(self, stage=None):
#         self.train_dataset = ImagePairDataset(self.image_dir, self.correspondence_dir, dataset_name=self.dataset_name, transform=self.transform)
#         self.val_dataset = ImagePairDataset(self.image_dir, self.correspondence_dir, dataset_name=self.dataset_name, transform=self.transform)

#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
#                           num_workers=self.num_workers, pin_memory=self.pin_memory)

#     def val_dataloader(self):
#         return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
#                           num_workers=self.num_workers, pin_memory=self.pin_memory)

# import os
# import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from pytorch_lightning import LightningDataModule
# import torch.nn.functional as F
# from PIL import Image

# class ImagePairDataset(Dataset):
#     def __init__(self, pkl_file, image_dir, dataset_name, transform=None):
#         self.image_dir = image_dir
#         self.data = pd.read_pickle(pkl_file)
#         self.dataset_name = dataset_name
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
#         img1_name = row['img1_name']
#         img2_name = row['img2_name']
#         matchings = row['matchings']

#         img1_path = os.path.join(self.image_dir, img1_name)
#         img2_path = os.path.join(self.image_dir, img2_name)

#         img1 = Image.open(img1_path).convert("RGB")
#         img2 = Image.open(img2_path).convert("RGB")

#         if self.transform:
#             img1 = self.transform(img1)
#             img2 = self.transform(img2)
#         else:
#             img1 = transforms.ToTensor()(img1)
#             img2 = transforms.ToTensor()(img2)

#         h1, w1 = img1.shape[1], img1.shape[2]
#         h2, w2 = img2.shape[1], img2.shape[2]

#         matching_matrix_0 = torch.zeros(h1, w1, 2)
#         matching_matrix_1 = torch.zeros(h2, w2, 2)

#         for (x0, y0), (x1, y1) in matchings.items():
#             if 0 <= y0 < h1 and 0 <= x0 < w1:
#                 matching_matrix_0[int(y0), int(x0), :] = torch.tensor([x1, y1])
#             if 0 <= y1 < h2 and 0 <= x1 < w2:
#                 matching_matrix_1[int(y1), int(x1), :] = torch.tensor([x0, y0])

#         return {
#             'image0': img1,
#             'image1': img2,
#             'matchings': matchings,
#             'matching_matrix_0': matching_matrix_0,
#             'matching_matrix_1': matching_matrix_1,
#             'dataset_name': self.dataset_name
#         }

# class ImagePairDataModule(LightningDataModule):
#     def __init__(self, args, config):
#         super().__init__()
#         self.train_pkl_file = config.DATASET.TRAIN_PKL_FILE
#         self.val_pkl_file = config.DATASET.VAL_PKL_FILE
#         self.test_pkl_file = config.DATASET.TEST_PKL_FILE
#         self.image_dir = config.DATASET.TRAIN_DATA_ROOT
#         self.batch_size = args.batch_size
#         self.num_workers = args.num_workers
#         self.pin_memory = args.pin_memory

#         self.transform = transforms.Compose([
#             transforms.Grayscale(num_output_channels=1),
#             transforms.ToTensor(),
#         ])

#     def setup(self, stage=None):
#         if stage == 'fit' or stage is None:
#             self.train_dataset = ImagePairDataset(self.train_pkl_file, self.image_dir, dataset_name='train', transform=self.transform)
#             self.val_dataset = ImagePairDataset(self.val_pkl_file, self.image_dir, dataset_name='val', transform=self.transform)

#         if stage == 'test' or stage is None:
#             self.test_dataset = ImagePairDataset(self.test_pkl_file, self.image_dir, dataset_name='test', transform=self.transform)

#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
#                           num_workers=self.num_workers, pin_memory=self.pin_memory, collate_fn=self.custom_collate_fn)

#     def val_dataloader(self):
#         return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
#                           num_workers=self.num_workers, pin_memory=self.pin_memory, collate_fn=self.custom_collate_fn)

#     def test_dataloader(self):
#         return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
#                           num_workers=self.num_workers, pin_memory=self.pin_memory, collate_fn=self.custom_collate_fn)

#     @staticmethod
#     def pad_tensor(tensor, pad_height, pad_width):
#         c, h, w = tensor.shape
#         padding = (0, pad_width - w, 0, pad_height - h)
#         return F.pad(tensor, padding, "constant", 0)

#     @staticmethod
#     def pad_matching_matrix(tensor, pad_height, pad_width):
#         h, w, d = tensor.shape
#         padding = (0, 0, 0, pad_width - w, 0, pad_height - h)
#         return F.pad(tensor, padding, "constant", 0)

#     @staticmethod
#     def custom_collate_fn(batch):
#         batch = list(filter(lambda x: x is not None, batch))

#         max_channels = max(d['image0'].shape[0] for d in batch)
#         max_height = max(max(d['image0'].shape[1], d['image1'].shape[1]) for d in batch)
#         max_width = max(max(d['image0'].shape[2], d['image1'].shape[2]) for d in batch)

#         for d in batch:
#             d['image0'] = ImagePairDataModule.pad_tensor(d['image0'], max_height, max_width)
#             d['image1'] = ImagePairDataModule.pad_tensor(d['image1'], max_height, max_width)
#             d['matching_matrix_0'] = ImagePairDataModule.pad_matching_matrix(d['matching_matrix_0'], max_height, max_width)
#             d['matching_matrix_1'] = ImagePairDataModule.pad_matching_matrix(d['matching_matrix_1'], max_height, max_width)

#         for d in batch:
#             assert d['matching_matrix_0'].shape[2] == 2, f"matching_matrix_0 shape {d['matching_matrix_0'].shape} is not correct"
#             assert d['matching_matrix_1'].shape[2] == 2, f"matching_matrix_1 shape {d['matching_matrix_1'].shape} is not correct"

#         collated_batch = {
#             key: torch.stack([d[key] for d in batch])
#             if key in ['image0', 'image1', 'matching_matrix_0', 'matching_matrix_1']
#             else [d[key] for d in batch] for key in batch[0]
#         }
#         return collated_batch


import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_lightning import LightningDataModule
import torch.nn.functional as F
from PIL import Image

class ImagePairDataset(Dataset):
    def __init__(self, pkl_file, image_dir, dataset_name, transform=None):
        self.image_dir = image_dir
        self.data = pd.read_pickle(pkl_file)
        self.dataset_name = dataset_name
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img1_name = row['img1_name']
        img2_name = row['img2_name']
        matchings = row['matchings']

        img1_path = os.path.join(self.image_dir, img1_name)
        img2_path = os.path.join(self.image_dir, img2_name)

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        else:
            img1 = transforms.ToTensor()(img1)
            img2 = transforms.ToTensor()(img2)

        h, w = img1.shape[1], img1.shape[2]

        matching_matrix = torch.zeros(h, w, 2)
        for (x0, y0), (x1, y1) in matchings.items():
            if 0 <= y0 < h and 0 <= x0 < w:
                matching_matrix[int(y0), int(x0), :] = torch.tensor([x1, y1])

        return {
            'image0': img1,
            'image1': img2,
            'matching_matrix': matching_matrix,
            'dataset_name': self.dataset_name
        }

class ImagePairDataModule(LightningDataModule):
    def __init__(self, args, config):
        super().__init__()
        self.train_pkl_file = config.DATASET.TRAIN_PKL_FILE
        self.val_pkl_file = config.DATASET.VAL_PKL_FILE
        self.test_pkl_file = config.DATASET.TEST_PKL_FILE
        self.image_dir = config.DATASET.TRAIN_DATA_ROOT
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = ImagePairDataset(self.train_pkl_file, self.image_dir, dataset_name='train', transform=self.transform)
            self.val_dataset = ImagePairDataset(self.val_pkl_file, self.image_dir, dataset_name='val', transform=self.transform)

        if stage == 'test' or stage is None:
            self.test_dataset = ImagePairDataset(self.test_pkl_file, self.image_dir, dataset_name='test', transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory, collate_fn=self.custom_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory, collate_fn=self.custom_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory, collate_fn=self.custom_collate_fn)

    @staticmethod
    def pad_tensor(tensor, pad_height, pad_width):
        c, h, w = tensor.shape
        padding = (0, pad_width - w, 0, pad_height - h)
        return F.pad(tensor, padding, "constant", 0)

    @staticmethod
    def pad_matching_matrix(tensor, pad_height, pad_width):
        h, w, d = tensor.shape
        padding = (0, 0, 0, pad_width - w, 0, pad_height - h)
        return F.pad(tensor, padding, "constant", 0)

    @staticmethod
    def custom_collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))

        max_channels = max(d['image0'].shape[0] for d in batch)
        max_height = max(max(d['image0'].shape[1], d['image1'].shape[1]) for d in batch)
        max_width = max(max(d['image0'].shape[2], d['image1'].shape[2]) for d in batch)

        for d in batch:
            d['image0'] = ImagePairDataModule.pad_tensor(d['image0'], max_height, max_width)
            d['image1'] = ImagePairDataModule.pad_tensor(d['image1'], max_height, max_width)
            d['matching_matrix'] = ImagePairDataModule.pad_matching_matrix(d['matching_matrix'], max_height, max_width)

        collated_batch = {
            key: torch.stack([d[key] for d in batch])
            if key in ['image0', 'image1', 'matching_matrix']
            else [d[key] for d in batch] for key in batch[0]
        }
        return collated_batch
