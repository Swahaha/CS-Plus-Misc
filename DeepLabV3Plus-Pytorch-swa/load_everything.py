import cv2
import numpy as np
from skimage.measure import label
from PIL import Image
import os

def load_image(image_path):
    return Image.open(image_path).convert('RGB')

def load_mask(mask_path):
    return np.array(Image.open(mask_path))

def separate_edges(mask):
    edge_mask = (mask == 200).astype(np.uint8)  # Assuming '200' is the class for edges
    labeled_edges = label(edge_mask)
    edge_layers = []

    for i in range(1, labeled_edges.max() + 1):
        layer = (labeled_edges == i).astype(np.uint8) * 255  # Create a binary mask for each edge
        edge_layers.append(layer)

    return edge_layers

def save_edge_layers(edge_layers, output_dir, base_name):
    os.makedirs(output_dir, exist_ok=True)
    for i, layer in enumerate(edge_layers):
        layer_image = Image.fromarray(layer)
        layer_image.save(os.path.join(output_dir, f'{base_name}_edge_{i}.png'))

def main(image_path, mask_path, output_dir):
    image = load_image(image_path)
    mask = load_mask(mask_path)

    edge_layers = separate_edges(mask)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_edge_layers(edge_layers, output_dir, base_name)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Separate edges from mask and save as different layers.')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--mask', type=str, required=True, help='Path to the corresponding mask')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the separated edge layers')

    args = parser.parse_args()

    main(args.image, args.mask, args.output_dir)
