import os

# Define the paths
images_dir = 'FullData/images'
masks_dir = 'FullData/masks'

# Get lists of files in each directory
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')])

# Check that the number of image and mask files are the same
assert len(image_files) == len(mask_files), "The number of images and masks should be the same."

# Rename the files in a consistent manner
for idx, (image_file, mask_file) in enumerate(zip(image_files, mask_files), start=1):
    # New filenames
    new_image_name = f'image_{idx:04d}.png'
    new_mask_name = f'mask_{idx:04d}.png'
    
    # New file paths
    new_image_path = os.path.join(images_dir, new_image_name)
    new_mask_path = os.path.join(masks_dir, new_mask_name)
    
    # Old file paths
    old_image_path = os.path.join(images_dir, image_file)
    old_mask_path = os.path.join(masks_dir, mask_file)
    
    # Rename files
    os.rename(old_image_path, new_image_path)
    os.rename(old_mask_path, new_mask_path)

print('Files renamed successfully!')
