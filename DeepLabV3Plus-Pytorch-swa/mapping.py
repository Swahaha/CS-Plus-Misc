import torch
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

def read_and_flip_coordinates(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    coordinates = []
    for line in lines:
        if line.startswith('Paths found:'):
            break
        if line.startswith('(') and line.endswith(')\n'):
            coord = tuple(map(int, line.strip('()\n').split(', ')))
            coordinates.append((coord[1], coord[0]))  # Flip the coordinates

    return coordinates

def find_correspondences(src_points, mkpts0, mkpts1):
    src_points = np.array(src_points)
    dst_points = []
    for pt in src_points:
        distances = np.linalg.norm(mkpts0 - pt, axis=1)
        nearest_index = np.argmin(distances)
        dst_points.append(mkpts1[nearest_index])
    return dst_points

def save_keypoints_image(img, keypoints, output_path):
    img_with_kp = img.copy()
    for kp in keypoints:
        cv2.circle(img_with_kp, (int(kp[0]), int(kp[1])), 5, (255, 0, 0), -1)
    plt.figure(figsize=(10, 10))
    plt.imshow(img_with_kp)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_coordinates(file_path, coordinates):
    with open(file_path, 'w') as file:
        file.write("Node coordinates:\n")
        for coord in coordinates:
            file.write(f"({coord[0]}, {coord[1]})\n")

# Set paths
text_file_path = None
image_folder = 'graphs'  # Folder containing the images
image_1_filename = None

# Find the text file
for filename in os.listdir(image_folder):
    if filename.endswith(".txt"):
        text_file_path = os.path.join(image_folder, filename)
        break

# Find the image file that starts with "image_"
for filename in os.listdir(image_folder):
    if filename.endswith("original.png"):
        image_1_filename = os.path.join(image_folder, filename)
        break

image_2_filename = 'image_3710.png'  # Path to the second image

# Read and flip coordinates from the text file
specific_points_img1 = read_and_flip_coordinates(text_file_path)

# Load images
img1 = cv2.imread(image_1_filename, cv2.IMREAD_COLOR)
img2 = cv2.imread(image_2_filename, cv2.IMREAD_COLOR)

# Convert to RGB
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Convert images to torch tensors
img1_t = K.image_to_tensor(img1, False).float() / 255.
img2_t = K.image_to_tensor(img2, False).float() / 255.

# Ensure images are in (batch, channel, height, width) format
img1_t = img1_t.unsqueeze(0)
img2_t = img2_t.unsqueeze(0)

# Convert to grayscale
img1_gray = K.color.rgb_to_grayscale(img1_t)
img2_gray = K.color.rgb_to_grayscale(img2_t)

# Remove any extra dimensions
img1_gray = img1_gray.squeeze(2)
img2_gray = img2_gray.squeeze(2)

# Load LoFTR model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loftr = KF.LoFTR(pretrained='indoor').to(device)
loftr.eval()

# Prepare input dict
input_dict = {"image0": img1_gray.to(device), 
              "image1": img2_gray.to(device)}

# Match features
with torch.no_grad():
    correspondences = loftr(input_dict)

# Extract matched keypoints
mkpts0 = correspondences['keypoints0'].cpu().numpy()
mkpts1 = correspondences['keypoints1'].cpu().numpy()

# Find corresponding points on the second image
specific_points_img2 = find_correspondences(specific_points_img1, mkpts0, mkpts1)

# Print corresponding points
print("Corresponding points on the second image:", specific_points_img2)

# Save the corresponding points on the second image
output_file_path = os.path.join(image_folder, 'corresponding_points.txt')
save_coordinates(output_file_path, specific_points_img2)

# Save the corresponding points image
output_image_path = os.path.join(image_folder, 'corresponding_points_image.png')
save_keypoints_image(img2, specific_points_img2, output_image_path)
