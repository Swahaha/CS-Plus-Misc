from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from torch.utils.data import Dataset 
from datasets import VOCSegmentation, Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from utils.visualizer import Visualizer
import torch.nn.functional as F


from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


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
            image, mask = self.transforms(image, mask)
            mask = np.array(mask)
            mask = self.map_mask_values(mask)
            mask = torch.tensor(mask, dtype=torch.long)

        return image, mask
    
    def map_mask_values(self, mask):
        mask[mask == 100] = 1
        mask[mask == 200] = 2
        return mask

def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes', 'map'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet50',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=512)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss', 'dice'], help="loss type (default: cross_entropy)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=2000,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser

# def get_dataset(opts):
#     """ Dataset And Augmentation
#     """
#     if opts.dataset == 'voc':
#         train_transform = et.ExtCompose([
#             et.ExtRandomScale((0.5, 2.0)),
#             et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
#             et.ExtRandomHorizontalFlip(),
#             et.ExtToTensor(),
#             et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225]),
#         ])
#         if opts.crop_val:
#             val_transform = et.ExtCompose([
#                 et.ExtResize(opts.crop_size),
#                 et.ExtCenterCrop(opts.crop_size),
#                 et.ExtToTensor(),
#                 et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225]),
#             ])
#         else:
#             val_transform = et.ExtCompose([
#                 et.ExtToTensor(),
#                 et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225]),
#             ])
#         train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
#                                     image_set='train', download=opts.download, transform=train_transform)
#         val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
#                                   image_set='val', download=False, transform=val_transform)

#     if opts.dataset == 'cityscapes':
#         train_transform = et.ExtCompose([
#             et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
#             et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
#             et.ExtRandomHorizontalFlip(),
#             et.ExtToTensor(),
#             et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225]),
#         ])

#         val_transform = et.ExtCompose([
#             et.ExtToTensor(),
#             et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225]),
#         ])

#         train_dst = Cityscapes(root=opts.data_root,
#                                split='train', transform=train_transform)
#         val_dst = Cityscapes(root=opts.data_root,
#                              split='val', transform=val_transform)

#     if opts.dataset == 'map':
#         train_transform = et.ExtCompose([
#             et.ExtRandomScale((0.5, 2.0)),
#             et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
#             et.ExtRandomHorizontalFlip(),
#             et.ExtToTensor(),
#             et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225]),
#         ])
#         val_transform = et.ExtCompose([
#             et.ExtToTensor(),
#             et.ExtNormalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225]),
#         ])
        
#         train_dst = MapSegmentationDataset(root=opts.data_root, transforms=train_transform)
#         val_dst = MapSegmentationDataset(root=opts.data_root, transforms=val_transform)
    
#     return train_dst, val_dst

def calculate_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=2)
    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return mean, std

def get_dataset(opts):
    """ Dataset And Augmentation """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    elif opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)

    elif opts.dataset == 'map':
        # Initial transform to convert images to tensors for mean and std calculation
        initial_transform = et.ExtCompose([
            et.ExtToTensor()
        ])
        
        train_dst = MapSegmentationDataset(root=opts.data_root, split='train', transforms=initial_transform)
        
        # Calculate mean and std from the dataset
        mean, std = calculate_mean_std(train_dst)
        print('Calculated Mean:', mean)
        print('Calculated Std:', std)

        train_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=mean.tolist(), std=std.tolist()),
        ])
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=mean.tolist(), std=std.tolist()),
        ])
        
        # Update the transforms with the calculated mean and std
        train_dst.transforms = train_transform
        val_dst = MapSegmentationDataset(root=opts.data_root, split='val', transforms=val_transform)
    
    return train_dst, val_dst



def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples

# def main():
#     opts = get_argparser().parse_args()
#     if opts.dataset.lower() == 'voc':
#         opts.num_classes = 21
#     elif opts.dataset.lower() == 'cityscapes':
#         opts.num_classes = 19
#     elif opts.dataset.lower() == 'map':
#         opts.num_classes = 3

#     os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print("Device: %s" % device)

#     # Setup random seed
#     torch.manual_seed(opts.random_seed)
#     np.random.seed(opts.random_seed)
#     random.seed(opts.random_seed)

#     metrics = StreamSegMetrics(opts.num_classes)

#     # Setup dataloader
#     if opts.dataset == 'voc' and not opts.crop_val:
#         opts.val_batch_size = 1

#     train_dst, val_dst = get_dataset(opts)
#     train_loader = data.DataLoader(
#         train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
#         drop_last=True)
#     val_loader = data.DataLoader(
#         val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
#     print("Dataset: %s, Train set: %d, Val set: %d" %
#           (opts.dataset, len(train_dst), len(val_dst)))

#     # Ensure you use the calculated mean and std in your normalization step
#     mean, std = calculate_mean_std(train_dst)
#     print('Mean:', mean)
#     print('Std:', std)

#     # Set up model (all models are 'constructed at network.modeling)
#     model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
#     if opts.separable_conv and 'plus' in opts.model:
#         network.convert_to_separable_conv(model.classifier)
#     utils.set_bn_momentum(model.backbone, momentum=0.01)

#     # Initialize the model without pretrained weights
#     model = nn.DataParallel(model)
#     model.to(device)
#     print("[!] Training from scratch")

#     # Set up optimizer
#     optimizer = torch.optim.SGD(params=[
#         {'params': model.module.backbone.parameters(), 'lr': 0.1 * opts.lr},
#         {'params': model.module.classifier.parameters(), 'lr': opts.lr},
#     ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    
#     if opts.lr_policy == 'poly':
#         scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
#     elif opts.lr_policy == 'step':
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

#     # Set up criterion
#     if opts.loss_type == 'focal_loss':
#         criterion = utils.FocalLoss(ignore_index=255, size_average=True)
#     elif opts.loss_type == 'cross_entropy':
#         # weights = torch.tensor([0.02, 0.20, 0.78], dtype=torch.float32).to(device)
#         # criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=255, reduction='mean')
#         criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')


#     def save_ckpt(path):
#         """ save current model """
#         torch.save({
#             "cur_itrs": cur_itrs,
#             "model_state": model.module.state_dict(),
#             "optimizer_state": optimizer.state_dict(),
#             "scheduler_state": scheduler.state_dict(),
#             "best_score": best_score,
#         }, path)
#         print("Model saved as %s" % path)

#     utils.mkdir('checkpoints')
#     # Initialize variables for training
#     best_score = 0.0
#     cur_itrs = 0
#     cur_epochs = 0

#     if opts.test_only:
#         model.eval()
#         val_score, ret_samples = validate(
#             opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)
#         print(metrics.to_str(val_score))
#         return

#     interval_loss = 0
#     while True:  # cur_itrs < opts.total_itrs:
#         model.train()
#         cur_epochs += 1
#         for (images, labels) in train_loader:
#             cur_itrs += 1

#             images = images.to(device, dtype=torch.float32)
#             labels = labels.to(device, dtype=torch.long)

#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             np_loss = loss.detach().cpu().numpy()
#             interval_loss += np_loss

#             if (cur_itrs) % 10 == 0:
#                 interval_loss = interval_loss / 10
#                 print("Epoch %d, Itrs %d/%d, Loss=%f" %
#                       (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
#                 interval_loss = 0.0

#             if (cur_itrs) % opts.val_interval == 0:
#                 save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
#                           (opts.model, opts.dataset, opts.output_stride))
#                 print("validation...")
#                 model.eval()
#                 val_score, ret_samples = validate(
#                     opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)
#                 print(metrics.to_str(val_score))
#                 if val_score['Mean IoU'] > best_score:  # save best model
#                     best_score = val_score['Mean IoU']
#                     save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
#                               (opts.model, opts.dataset, opts.output_stride))
#                 model.train()
#             scheduler.step()

#             if cur_itrs >= opts.total_itrs:
#                 return

# if __name__ == '__main__':
#     main()

# class Dice(nn.Module):
#     def __init__(self, weights):
#         super(Dice, self).__init__()
#         self.weights = weights

#     def forward(self, inputs, targets, smooth=1):
#         # Flatten inputs and targets to calculate intersection and union
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
        
#         intersection = (inputs * targets).sum()
#         dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
#         weighted_dice_loss = dice_loss * self.weights[targets.long()].mean()
#         return weighted_dice_loss

# class Dice(nn.Module):
#     def __init__(self, weights):
#         super(Dice, self).__init__()
#         self.weights = weights

#     def forward(self, inputs, targets, smooth=1):
#         # Flatten inputs and targets to calculate intersection and union
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)

#         # Create one-hot encoding of targets
#         targets_one_hot = torch.eye(self.weights.size(0), device=targets.device)[targets].permute(0, 3, 1, 2)

#         intersection = (inputs * targets_one_hot).sum()
#         dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

#         weighted_dice_loss = dice_loss * self.weights[targets.long()].mean()
#         return weighted_dice_loss


def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
    elif opts.dataset.lower() == 'map':
        opts.num_classes = 3

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    metrics = StreamSegMetrics(opts.num_classes)

    # Setup dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        drop_last=True)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Initialize the model without pretrained weights
    model = nn.DataParallel(model)
    model.to(device)
    print("[!] Training from scratch")

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.module.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.module.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        weights = torch.tensor([0.1, 0.5, 0.5], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=255, reduction='mean')
        # criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    # elif opts.loss_type == 'dice':
    #     class_weights = torch.tensor([1.0, 5.0, 5.0], dtype=torch.float32).to(device)
    #     criterion = Dice(weights=class_weights)

    def save_ckpt(path):
        """ save current model """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Initialize variables for training
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)
        print(metrics.to_str(val_score))
        return

    interval_loss = 0
    while True:  # cur_itrs < opts.total_itrs:
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
                          (opts.model, opts.dataset, opts.output_stride))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                              (opts.model, opts.dataset, opts.output_stride))
                model.train()
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                return

if __name__ == '__main__':
    main()
