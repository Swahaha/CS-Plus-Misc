import torch
from PIL import Image
from torchvision import transforms
from src.loftr import LoFTR, default_cfg
import cv2
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define a transformation to convert images to tensor and to grayscale
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# Load images
img0 = Image.open('input_images/img_1.png').convert('RGB')
img1 = Image.open('input_images/img_2.png').convert('RGB')
img0 = transform(img0).unsqueeze(0).to(device) 
img1 = transform(img1).unsqueeze(0).to(device) 

# Initialize LoFTR
matcher = LoFTR(config=default_cfg)
checkpoint = torch.load('indoor_ot.ckpt', map_location=device)

state_dict = checkpoint['state_dict']
model_state_dict = matcher.state_dict()
filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}

matcher.load_state_dict(filtered_state_dict)
matcher = matcher.eval().to(device)

batch = {'image0': img0, 'image1': img1}

with torch.no_grad():
    matcher(batch)
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()

print("Matched keypoints in image 0:", mkpts0)
print("Matched keypoints in image 1:", mkpts1)

img0_rgb = Image.open('input_images/img_1.png').convert('RGB')
img1_rgb = Image.open('input_images/img_2.png').convert('RGB')
img0_np = np.array(img0_rgb)
img1_np = np.array(img1_rgb)

for pt in mkpts0:
    cv2.circle(img0_np, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
for pt in mkpts1:
    cv2.circle(img1_np, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)

cv2.imwrite('output_images/matched_img1.png', cv2.cvtColor(img0_np, cv2.COLOR_RGB2BGR))
cv2.imwrite('output_images/matched_img2.png', cv2.cvtColor(img1_np, cv2.COLOR_RGB2BGR))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img0_np)
plt.title('Image 0 with keypoints')

plt.subplot(1, 2, 2)
plt.imshow(img1_np)
plt.title('Image 1 with keypoints')

plt.show()
