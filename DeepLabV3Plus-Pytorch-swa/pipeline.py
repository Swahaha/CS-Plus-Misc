import cv2
import numpy as np
from scipy.ndimage import label, center_of_mass
import torch
import torch.nn as nn
from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import argparse
from torchvision import transforms as T
from PIL import Image
from glob import glob

BACKGROUND = 0
EDGE = 100
NODE = 200

def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--input", type=str, required=True,
                        help="path to a single image")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes', 'map'], help='Name of training set')

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Save Options
    parser.add_argument("--save_val_results_to", default=None,
                        help="save segmentation results to the specified dir")
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser

def decode_map_target(predictions):
    decoded = np.zeros_like(predictions, dtype=np.uint8)
    decoded[predictions == 0] = BACKGROUND  # Background
    decoded[predictions == 1] = EDGE  # Edge
    decoded[predictions == 2] = NODE  # Node
    return decoded

def find_nodes(mask_image):
    labeled_mask, num_labels = label(mask_image == NODE)
    node_centroids = center_of_mass(mask_image == NODE, labeled_mask, range(1, num_labels + 1))
    node_centroids = [(int(x), int(y)) for x, y in node_centroids]
    return node_centroids

def find_edges_around_nodes(mask_image, nodes, radius=10):
    edges_around_nodes = []

    for node in nodes:
        node_edges = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x = node[0] + dx
                y = node[1] + dy
                if 0 <= x < mask_image.shape[0] and 0 <= y < mask_image.shape[1]:
                    if mask_image[x, y] == EDGE:
                        node_edges.append((x, y))
        edges_around_nodes.append(node_edges)
    return edges_around_nodes

def find_closest_node(point, nodes):
    return min(nodes, key=lambda node: np.linalg.norm(np.array(point) - np.array(node)))

def get_angle(p1, p2):
    return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

def angle_difference(angle1, angle2):
    return min(abs(angle1 - angle2), 2 * np.pi - abs(angle1 - angle2))

def compute_weighted_direction(path):
    if len(path) < 2:
        return None
    dx = path[-1][0] - path[0][0]
    dy = path[-1][1] - path[0][1]
    return get_angle((0, 0), (dx, dy))

def depth_first_search(mask_image, start, visited, start_node, nodes, min_path_length=10, initial_dir=None):
    stack = [(start, [start], initial_dir)] 
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]

    while stack:
        (x, y), path, prev_dir = stack.pop()
        if (x, y) in visited:
            continue
        visited.add((x, y))
        path.append((x, y))

        overall_dir = compute_weighted_direction(path[-10:]) if len(path) > 10 else None

        if prev_dir is None:
            sorted_directions = directions
        else:
            sorted_directions = sorted(directions, key=lambda d: angle_difference(prev_dir, get_angle((0, 0), d)))

        for dx, dy in sorted_directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < mask_image.shape[0] and 0 <= ny < mask_image.shape[1]:
                if mask_image[nx, ny] == NODE:
                    end_node = find_closest_node((nx, ny), nodes)
                    if end_node != start_node and len(path) >= min_path_length:
                        path.append((nx, ny))
                        return path, end_node
                if mask_image[nx, ny] == EDGE and (nx, ny) not in visited:
                    angle_to_next = get_angle((x, y), (nx, ny))
                    stack.append(((nx, ny), path.copy(), angle_to_next))

    return (path, find_closest_node(path[-1], nodes)) if len(path) >= min_path_length else ([], None)

def find_paths_from_edges(mask_image, edges_around_nodes, nodes, min_path_length=10):
    all_paths = []
    visited = set()

    for node_edges, start_node in zip(edges_around_nodes, nodes):
        for edge in node_edges:
            if edge not in visited:
                initial_dir = get_angle(start_node, edge)
                path, end_node = depth_first_search(mask_image, edge, visited, start_node, nodes, min_path_length, initial_dir=initial_dir)
                if path and end_node:
                    all_paths.append((path, start_node, end_node))

    return all_paths

def filter_paths(paths, length_tolerance=10):
    filtered_paths = []
    seen_paths = {}

    for path, start_node, end_node in paths:
        if len(path) >= 10:
            path_key = (start_node, end_node)
            if path_key not in seen_paths:
                seen_paths[path_key] = (path, len(path))
                filtered_paths.append((path, start_node, end_node))
            else:
                existing_path, existing_length = seen_paths[path_key]
                if abs(existing_length - len(path)) > length_tolerance:
                    filtered_paths.append((path, start_node, end_node))
                    seen_paths[path_key] = (path, len(path))

    return filtered_paths

def save_paths_to_file(nodes, paths, output_file):
    with open(output_file, 'w') as f:
        f.write("Node coordinates:\n")
        for node in nodes:
            f.write(f"{node}\n")

        f.write("\nPaths found:\n")
        for i, (path, start_node, end_node) in enumerate(paths):
            f.write(f"Path {i} with {len(path)} points. Start Node: {start_node}, End Node: {end_node}\n")

        f.write("\nAdjacency List:\n")
        adjacency_list = {}
        for path, start_node, end_node in paths:
            if start_node not in adjacency_list:
                adjacency_list[start_node] = []
            if end_node not in adjacency_list:
                adjacency_list[end_node] = []
            adjacency_list[start_node].append(end_node)
            adjacency_list[end_node].append(start_node)
        
        for node, neighbors in adjacency_list.items():
            neighbors_str = ', '.join(map(str, neighbors))
            f.write(f"{node}: {neighbors_str}\n")

def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'map':
        opts.num_classes = 3
        decode_fn = decode_map_target

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    image_file = opts.input

    # Set up model (all models are constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Loaded model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Checkpoint not found or not specified, please provide a valid checkpoint path")
        return

    # Normalization values specific to the dataset
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
    
    with torch.no_grad():
        model.eval() 
        img_path = image_file
        ext = os.path.basename(img_path).split('.')[-1]
        img_name = os.path.basename(img_path)[:-len(ext)-1]
        img = Image.open(img_path).convert('RGB')
        img = transform(img).unsqueeze(0) 
        img = img.to(device)
        
        pred = model(img).max(1)[1].cpu().numpy()[0] # HW
        colorized_preds = decode_fn(pred).astype('uint8')
        
        if opts.save_val_results_to:
            os.makedirs(opts.save_val_results_to, exist_ok=True)
            # Save the original input image
            original_img_path = os.path.join(opts.save_val_results_to, img_name + '_original.png')
            Image.open(img_path).save(original_img_path)

            # Save the mask image
            colorized_preds_img = Image.fromarray(colorized_preds)
            mask_img_path = os.path.join(opts.save_val_results_to, img_name + '_mask.png')
            colorized_preds_img.save(mask_img_path)

        # Load the saved mask image for node and path finding
        mask_image = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
        
        # Process the predictions to find nodes and paths
        nodes = find_nodes(mask_image)
        edges_around_nodes = find_edges_around_nodes(mask_image, nodes)
        paths = find_paths_from_edges(mask_image, edges_around_nodes, nodes, min_path_length=10)
        filtered_paths = filter_paths(paths, length_tolerance=10)
        
        # Save the nodes and paths to a text file
        output_file = os.path.join(opts.save_val_results_to, img_name + '_paths.txt')
        save_paths_to_file(nodes, filtered_paths, output_file)

if __name__ == '__main__':
    main()
