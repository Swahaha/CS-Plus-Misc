import os
import shutil
from sklearn.model_selection import train_test_split

# Define the paths
base_dir = 'FullData'
images_dir = os.path.join(base_dir, 'images')
masks_dir = os.path.join(base_dir, 'masks')

# Create directories for train, val, test sets
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(base_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, split, 'masks'), exist_ok=True)

# Get list of image and mask files
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')])

# Check that the number of image and mask files match
assert len(image_files) == len(mask_files), "The number of images and masks should be the same."

# Split the dataset into train (70%), val (15%), and test (15%)
train_images, temp_images, train_masks, temp_masks = train_test_split(image_files, mask_files, test_size=0.3, random_state=42)
val_images, test_images, val_masks, test_masks = train_test_split(temp_images, temp_masks, test_size=0.5, random_state=42)

# Function to move files
def move_files(file_list, source_dir, dest_dir):
    for file_name in file_list:
        shutil.move(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))

# Move files to respective directories
move_files(train_images, images_dir, os.path.join(base_dir, 'train', 'images'))
move_files(train_masks, masks_dir, os.path.join(base_dir, 'train', 'masks'))

move_files(val_images, images_dir, os.path.join(base_dir, 'val', 'images'))
move_files(val_masks, masks_dir, os.path.join(base_dir, 'val', 'masks'))

move_files(test_images, images_dir, os.path.join(base_dir, 'test', 'images'))
move_files(test_masks, masks_dir, os.path.join(base_dir, 'test', 'masks'))

print('Dataset split into train, val, and test sets successfully!')
