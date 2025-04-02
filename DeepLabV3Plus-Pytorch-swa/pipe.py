import os
import shutil

# Define the paths
source_dir = 'syndata_curved_w_data_square'  # Update this to your source directory
images_dir = os.path.join(source_dir, 'images')
masks_dir = os.path.join(source_dir, 'masks')

# Create directories if they don't exist
os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

# Iterate over files in the source directory
for filename in os.listdir(source_dir):
    if filename.endswith('.png') and '-seg' not in filename:
        # This is an image file
        shutil.move(os.path.join(source_dir, filename), os.path.join(images_dir, filename))
    elif filename.endswith('-seg.png'):
        # This is a mask file
        shutil.move(os.path.join(source_dir, filename), os.path.join(masks_dir, filename))

print('Dataset organized successfully!')
