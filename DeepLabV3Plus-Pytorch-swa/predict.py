import torch
import torch.nn as nn

from tqdm import tqdm
import network
import utils
import os
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes
from torchvision import transforms as T
from metrics import StreamSegMetrics

from PIL import Image
import matplotlib.pyplot as plt
from glob import glob

def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--input_dir", type=str, default='FullData/test/images',
                        help="path to image directory")
    parser.add_argument("--dataset", type=str, default='map',
                        choices=['voc', 'cityscapes', 'map'], help='Name of training set')

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
    parser.add_argument("--save_val_results_to", default=None,
                        help="save segmentation results to the specified dir")

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser

def decode_map_target(predictions):
    decoded = np.zeros_like(predictions, dtype=np.uint8)
    decoded[predictions == 0] = 0  # Background
    decoded[predictions == 1] = 100  # Node
    decoded[predictions == 2] = 200  # Edge
    return decoded

def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        decode_fn = Cityscapes.decode_target
    elif opts.dataset.lower() == 'map':
        opts.num_classes = 3
        decode_fn = decode_map_target

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    image_files = sorted(glob(os.path.join(opts.input_dir, '*.*')))[:20]

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # Normalization values specific to your dataset
    mean = [0.7715, 0.8988, 0.9856]
    std = [0.2449, 0.1588, 0.0835]

    if opts.crop_val:
        transform = T.Compose([
                T.Resize(opts.crop_size),
                T.CenterCrop(opts.crop_size),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])
    else:
        transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])
    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)
    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files):
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext)-1]
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0) # To tensor of NCHW
            img = img.to(device)
            
            pred = model(img).max(1)[1].cpu().numpy()[0] # HW
            colorized_preds = decode_fn(pred).astype('uint8')
            colorized_preds = Image.fromarray(colorized_preds)
            if opts.save_val_results_to:
                colorized_preds.save(os.path.join(opts.save_val_results_to, img_name + '.png'))

if __name__ == '__main__':
    main()
