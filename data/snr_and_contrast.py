import os
import numpy as np
import matplotlib.pyplot as plt
import cv2  

# Use the paths you established in aspect_ratio.py

base_path = "/datasets/tdt4265/Poles2025"

V1_IMG_DIR = f"{base_path}/roadpoles_v1/train/images"
V1_LBL_DIR = f"{base_path}/roadpoles_v1/train/labels"

IPHONE_IMG_DIR = f"{base_path}/Road_poles_iPhone/images/Train/train"
IPHONE_LBL_DIR = f"{base_path}/Road_poles_iPhone/labels/Train/train"

def calculate_metrics_for_dataset(img_dir, lbl_dir, max_samples=500):
    signals_mean = []
    noise_mean = []
    noise_std = []
    
    
    label_files = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')][:max_samples]
    
    for lbl_file in label_files:
        img_name = lbl_file.replace('.txt', '.jpg') 
        img_path = os.path.join(img_dir, img_name)
        
       
        if not os.path.exists(img_path):
            img_path = img_path.replace('.jpg', '.PNG')
            
        img = cv2.imread(img_path)
        if img is None: continue
            
        # Extract the RED channel instead of Grayscale (Red is index 2 in OpenCV's BGR)
        # Snow poles are red, snow is white. This gives much better contrast data
        channel = img[:, :, 2] 
        h, w = channel.shape
        
        with open(os.path.join(lbl_dir, lbl_file), 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5: continue
                _, x, y, nw, nh = map(float, parts)
                
                # Bounding box coordinates
                x1, y1 = int((x - nw/2) * w), int((y - nh/2) * h)
                x2, y2 = int((x + nw/2) * w), int((y + nh/2) * h)
                
                # Ensure coordinates are within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                pole_region = channel[y1:y2, x1:x2]
                if pole_region.size == 0: continue
                    
                signals_mean.append(np.mean(pole_region))

                # Noise from background (padding left and right of the pole)
                nx1, ny1 = max(0, x1-15), y1
                nx2, ny2 = min(w, x1-5), min(h, y1+10)
                bg_region = channel[ny1:ny2, nx1:nx2]
                
                if bg_region.size > 0:
                    noise_mean.append(np.mean(bg_region))
                    noise_std.append(np.std(bg_region))

    return signals_mean, noise_mean, noise_std

print("Processing roadpoles_v1...")
v1_sig, v1_noise, v1_std = calculate_metrics_for_dataset(V1_IMG_DIR, V1_LBL_DIR)

print("Processing iPhone...")
ip_sig, ip_noise, ip_std = calculate_metrics_for_dataset(IPHONE_IMG_DIR, IPHONE_LBL_DIR)

# Calculations
v1_snr = np.nanmean(v1_sig) / np.nanmean(v1_std)
ip_snr = np.nanmean(ip_sig) / np.nanmean(ip_std)

v1_contrast = np.nanmean(v1_sig) / np.nanmean(v1_noise)
ip_contrast = np.nanmean(ip_sig) / np.nanmean(ip_noise)

print(f"roadpoles_v1 - SNR: {v1_snr:.2f} | Contrast: {v1_contrast:.2f}")
print(f"iPhone       - SNR: {ip_snr:.2f} | Contrast: {ip_contrast:.2f}")

# Plotting
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(v1_sig, bins=20, alpha=0.7, label='v1 Signal (Pole)', color='red')
plt.hist(v1_noise, bins=20, alpha=0.7, label='v1 Noise (Bg)', color='gray')
plt.title('roadpoles_v1: Signal vs Noise (Red Channel)')
plt.xlabel('Pixel Intensity')  
plt.ylabel('Frequency')
plt.legend()    

plt.subplot(1, 2, 2)
plt.hist(ip_sig, bins=20, alpha=0.7, label='iPhone Signal (Pole)', color='blue')
plt.hist(ip_noise, bins=20, alpha=0.7, label='iPhone Noise (Bg)', color='lightblue')
plt.title('iPhone: Signal vs Noise (Red Channel)')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.savefig("rgb_datasets_snr_comparison.png")
print("Saved plot to rgb_datasets_snr_comparison.png")