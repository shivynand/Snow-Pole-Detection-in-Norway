import os
import numpy as np
import matplotlib.pyplot as plt
import cv2  

base_path = "/datasets/tdt4265/ad/open/Poles"
lidar_img_dir = f"{base_path}/lidar/combined_color/train"
rgb_img_dir   = f"{base_path}/rgb/images/train"
lidar_lbl_dir = f"{base_path}/lidar/labels/train"
rgb_lbl_dir   = f"{base_path}/rgb/labels/train"

# Step 1 - Get LiDAR image ids that exist in RGB for complete analysis
lidar_files = set(f for f in os.listdir(lidar_img_dir) if f.endswith('.png'))
lidar_nums  = set(int(f.split('_')[1].split('.')[0]) for f in lidar_files)

rgb_files = set(f for f in os.listdir(rgb_img_dir) if f.endswith('.PNG'))
rgb_nums = set(int(f.split('_')[1].split('.')[0]) for f in rgb_files)

common_nums = sorted(list(lidar_nums.intersection(rgb_nums)))


# Step 2 - Correlation matrix for lidar and rgb spectral channels
lidar_correlations = np.zeros((3, 3))
rgb_correlations = np.zeros((3, 3))
count_lidar = 0
count_rgb = 0


for num in common_nums:
    lidar_path = os.path.join(lidar_img_dir, f"image_{num}.png")
    rgb_path = os.path.join(rgb_img_dir, f"frame_{num:06d}.PNG")

    lidar_imgs = cv2.imread(lidar_path)
    rgb_imgs = cv2.imread(rgb_path)

    if lidar_imgs is not None:
        h,w,_ = lidar_imgs.shape
        with open(os.path.join(lidar_lbl_dir, f"image_{num}.txt"), 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5: continue
                _, x, y, nw, nh = map(float, parts)
                x1, y1 = int((x - nw/2) * w), int((y - nh/2) * h)
                x2, y2 = int((x + nw/2) * w), int((y + nh/2) * h)
                pole_region = lidar_imgs[y1:y2, x1:x2]

                if pole_region.size > 50:
                    b = pole_region[:, :, 0].flatten()
                    g = pole_region[:, :, 1].flatten()
                    r = pole_region[:, :, 2].flatten()

                    matrix = np.corrcoef([r, g, b])

                    if not np.isnan(matrix).any():
                        lidar_correlations += matrix
                        count_lidar += 1

    if rgb_imgs is not None:
        h,w,_ = rgb_imgs.shape
        with open(os.path.join(rgb_lbl_dir, f"frame_{num:06d}.txt"), 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5: continue
                _, x, y, nw, nh = map(float, parts)
                x1, y1 = int((x - nw/2) * w), int((y - nh/2) * h)
                x2, y2 = int((x + nw/2) * w), int((y + nh/2) * h)
                pole_region = rgb_imgs[y1:y2, x1:x2]

                if pole_region.size > 50:
                    b = pole_region[:, :, 0].flatten()
                    g = pole_region[:, :, 1].flatten()
                    r = pole_region[:, :, 2].flatten()

                    matrix = np.corrcoef([r, g, b])

                    if not np.isnan(matrix).any():
                        rgb_correlations += matrix
                        count_rgb += 1

# Average the correlation matrices
avg_lidar_corr = lidar_correlations / count_lidar 
avg_rgb_corr = rgb_correlations / count_rgb

print("Average LiDAR Spectral Correlation Matrix:")
print(avg_lidar_corr)   
print("Average RGB Spectral Correlation Matrix:")
print(avg_rgb_corr)   
print(count_lidar, count_rgb)

# Pick a sample image from your common_nums
sample_num = common_nums[0] 
lidar_path = os.path.join(lidar_img_dir, f"image_{sample_num}.png")
img = cv2.imread(lidar_path)

if img is not None:
    # Convert BGR (OpenCV default) to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Split the channels
    # Note: Depending on the dataset, these often map to Height, Intensity, and Density
    c1, c2, c3 = cv2.split(img_rgb)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title("Combined LiDAR Image")
    
    axes[1].imshow(c1, cmap='gray')
    axes[1].set_title("Channel 1 (Red)")
    
    axes[2].imshow(c2, cmap='gray')
    axes[2].set_title("Channel 2 (Green)")
    
    axes[3].imshow(c3, cmap='gray')
    axes[3].set_title("Channel 3 (Blue)")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
else:
    print("Image not found. Check your paths!")