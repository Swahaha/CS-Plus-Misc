import os
import numpy as np
from PIL import Image, ImageDraw

# Create directories
os.makedirs("dataa/images", exist_ok=True)
os.makedirs("dataa/correspondences", exist_ok=True)

# Function to create dummy images
def create_dummy_image(path, size=(512, 512)):
    color = tuple(np.random.choice(range(256), size=3).tolist())
    image = Image.new('RGB', size, color=color)
    draw = ImageDraw.Draw(image)
    for i in range(10):
        x, y = np.random.randint(0, size[0]), np.random.randint(0, size[1])
        draw.ellipse((x-5, y-5, x+5, y+5), fill=(255, 0, 0), outline=(0, 255, 0))
    image.save(path)

# Function to create dummy correspondences
def create_dummy_correspondences(path, num_points=10, img_size=(512, 512)):
    with open(path, 'w') as f:
        for _ in range(num_points):
            x1, y1 = np.random.randint(0, img_size[0]), np.random.randint(0, img_size[1])
            x2, y2 = np.random.randint(0, img_size[0]), np.random.randint(0, img_size[1])
            f.write(f"{x1} {y1} {x2} {y2}\n")

# Generate dummy data
for i in range(1, 101):  # Generate 5 pairs
    create_dummy_image(f"dataa/images/pair{i}_1.png")
    create_dummy_image(f"dataa/images/pair{i}_2.png")
    create_dummy_correspondences(f"dataa/correspondences/pair{i}.txt")

# Check the generated files
os.listdir("dataa/images"), os.listdir("dataa/correspondences")
