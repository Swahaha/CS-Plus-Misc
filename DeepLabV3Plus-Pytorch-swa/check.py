from PIL import Image
import numpy as np

def count_pixel_values(image_path):
    # Open the image using Pillow
    img = Image.open(image_path)
    # Convert the image to grayscale (if not already)
    img = img.convert('L')
    # Convert the image data to a numpy array
    img_array = np.array(img)
    
    # Count the number of pixels with values 0, 1, and 2
    count_0 = np.sum(img_array == 0)
    count_1 = np.sum(img_array == 100)
    count_2 = np.sum(img_array == 200)
    
    return count_0, count_1, count_2

# Example usage
image_path = 'results/image_0012.png'
count_0, count_1, count_2 = count_pixel_values(image_path)

print(f"Number of pixels with background: {count_0}")
print(f"Number of pixels with edge: {count_1}")
print(f"Number of pixels with node: {count_2}")
