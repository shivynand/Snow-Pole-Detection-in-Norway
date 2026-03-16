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


# Step 2 - Isolate signal and noise
lidar_signals_mean = []
rgb_signals_mean = []

lidar_noise_mean = []
rgb_noise_mean = []

lidar_noise_std = []
rgb_noise_std = []


for num in common_nums:
    lidar_path = os.path.join(lidar_img_dir, f"image_{num}.png")
    rgb_path = os.path.join(rgb_img_dir, f"frame_{num:06d}.PNG")

    lidar_imgs = cv2.imread(lidar_path)
    rgb_imgs = cv2.imread(rgb_path)

    if lidar_imgs is not None:
        lidar_gray = cv2.cvtColor(lidar_imgs, cv2.COLOR_BGR2GRAY)
        hl,wl= lidar_gray.shape
        with open(os.path.join(lidar_lbl_dir, f"image_{num}.txt"), 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5: continue
                _, x, y, nw, nh = map(float, parts)
                x1, y1 = int((x - nw/2) * wl), int((y - nh/2) * hl)
                x2, y2 = int((x + nw/2) * wl), int((y + nh/2) * hl)
                pole_region = lidar_gray[y1:y2, x1:x2]

                signal_strength = np.mean(pole_region)
                lidar_signals_mean.append(signal_strength)

                # Noise for LiDAR from background (with padding)
                nx1, ny1 = max(0, x1-15), y1
                nx2, ny2 = min(wl, x1 - 5), min(hl, y1 + 10)

                bg_lidar = lidar_gray[ny1:ny2, nx1:nx2]

                lidar_noise_mean.append(np.mean(bg_lidar))
                lidar_noise_std.append(np.std(bg_lidar))

                if num < common_nums[5]: # Only save for the first 5 images
                    cv2.imwrite(f"debug_lidar_{num}.png", pole_region)
                    cv2.imwrite(f"debug_lidar_bg_{num}.png", bg_lidar)

    if rgb_imgs is not None:
        rgb_gray = cv2.cvtColor(rgb_imgs, cv2.COLOR_BGR2GRAY)
        hr, wr = rgb_gray.shape
        with open(os.path.join(rgb_lbl_dir, f"frame_{num:06d}.txt"), 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5: continue
                _, x, y, nw, nh = map(float, parts)
                x1, y1 = int((x - nw/2) * wr), int((y - nh/2) * hr)
                x2, y2 = int((x + nw/2) * wr), int((y + nh/2) * hr)
                pole_region = rgb_gray[y1:y2, x1:x2]
                signal_strength = np.mean(pole_region)
                rgb_signals_mean.append(signal_strength)

                # Noise for RGB from background (with padding)
                nx1, ny1 = max(0, x1-15), y1
                nx2, ny2 = min(wr, x1 - 5), min(hr, y1 + 10)

                bg_rgb = rgb_gray[ny1:ny2, nx1:nx2]
                rgb_noise_mean.append(np.mean(bg_rgb))
                rgb_noise_std.append(np.std(bg_rgb))

                if num < common_nums[5]: # Only save for the first 5 images
                    cv2.imwrite(f"debug_rgb_{num}.png", pole_region)
                    cv2.imwrite(f"debug_rgb_bg_{num}.png", bg_rgb)

# Step 3 - Calculate SNR and Contrast and Contrast to Noise Ratio (CNR)

# SNR = Mean Signal / Std Dev Noise
lidar_snr = np.nanmean(lidar_signals_mean) / np.nanmean(lidar_noise_std)
rgb_snr = np.nanmean(rgb_signals_mean) / np.nanmean(rgb_noise_std)

# Contrast = Mean Signal / Mean Noise
lidar_contrast = np.nanmean(lidar_signals_mean) / np.nanmean(lidar_noise_mean)
rgb_contrast = np.nanmean(rgb_signals_mean) / np.nanmean(rgb_noise_mean)

# Contrast to Noise Ratio (CNR) = (Mean Signal - Mean Noise) / Std Dev Noise
lidar_cnr = (np.nanmean(lidar_signals_mean) - np.nanmean(lidar_noise_mean)) / np.nanmean(lidar_noise_std)
rgb_cnr = (np.nanmean(rgb_signals_mean) - np.nanmean(rgb_noise_mean)) / np.nanmean(rgb_noise_std)   

# Print all metrics
print(f"LiDAR SNR: {lidar_snr:.2f}")
print(f"RGB SNR: {rgb_snr:.2f}")

print(f"LiDAR Contrast: {lidar_contrast:.2f}")
print(f"RGB Contrast: {rgb_contrast:.2f}")

print(f"LiDAR CNR: {lidar_cnr:.2f}")
print(f"RGB CNR: {rgb_cnr:.2f}")

#Plotting signal and noise distributions
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(lidar_signals_mean, bins=20, alpha=0.7, label='LiDAR Signal', color='blue')
plt.hist(lidar_noise_mean, bins=20, alpha=0.7, label='LiDAR Noise', color='cyan')
plt.title('LiDAR Signal vs Noise Distribution')
plt.xlabel('Pixel Intensity')  
plt.ylabel('Frequency')
plt.legend()    

plt.subplot(1, 2, 2)
plt.hist(rgb_signals_mean, bins=20, alpha=0.7, label='RGB Signal', color='green')
plt.hist(rgb_noise_mean, bins=20, alpha=0.7, label='RGB Noise', color='lightgreen')
plt.title('RGB Signal vs Noise Distribution')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig("snr_contrast_distribution.png")

