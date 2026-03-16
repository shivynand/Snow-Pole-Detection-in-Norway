import os
import cv2
import random
import matplotlib.pyplot as plt

# --- 1. FINAL CORRECTED PATHS ---
base_path = "/datasets/tdt4265/ad/open/Poles"
lidar_img_dir = f"{base_path}/lidar/combined_color/train"
rgb_img_dir   = f"{base_path}/rgb/images/train"
lidar_lbl_dir = f"{base_path}/lidar/labels/train"
rgb_lbl_dir   = f"{base_path}/rgb/labels/train"

def draw_boxes(img, label_path, color=(0, 0, 0)): # Default to Black as you requested
    if img is None: return None
    h, w, _ = img.shape
    if not os.path.exists(label_path):
        return img
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5: continue
            _, x, y, nw, nh = map(float, parts)
            x1 = int((x - nw/2) * w)
            y1 = int((y - nh/2) * h)
            x2 = int((x + nw/2) * w)
            y2 = int((y + nh/2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1) # Thin line for small objects
    return img

# Get list from LiDAR folder
all_lidar_files = [f for f in os.listdir(lidar_img_dir) if f.endswith('.png')]
samples = random.sample(all_lidar_files, 3)

fig, axes = plt.subplots(3, 2, figsize=(15, 10))

for i, img_name in enumerate(samples):
    # --- LIDAR LOADING ---
    path_l = os.path.join(lidar_img_dir, img_name)
    img_l = cv2.imread(path_l)
    lbl_l = os.path.join(lidar_lbl_dir, img_name.replace('.png', '.txt'))

    # --- RGB MAPPING (image_822.png -> frame_000822.PNG) ---
    img_num = img_name.split('_')[1].split('.')[0]
    rgb_name = f"frame_{int(img_num):06d}.PNG" # Formats to 6 digits
    path_r = os.path.join(rgb_img_dir, rgb_name)
    lbl_r = os.path.join(rgb_lbl_dir, rgb_name.replace('.PNG', '.txt'))
    
    img_r = cv2.imread(path_r)

    # --- DRAWING ---
    res_l = draw_boxes(img_l, lbl_l, color=(0, 0, 0)) # Black boxes
    res_r = draw_boxes(img_r, lbl_r, color=(0, 255, 0)) # Green for RGB contrast

    # --- DISPLAY ---
    if res_r is not None:
        axes[i, 0].imshow(cv2.cvtColor(res_r, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"RGB: {rgb_name}")
    else:
        axes[i, 0].set_title(f"RGB MISSING: {rgb_name}")
        
    if res_l is not None:
        axes[i, 1].imshow(cv2.cvtColor(res_l, cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(f"LiDAR: {img_name}")

plt.tight_layout()
plt.savefig("final_audit.png")
print("Check final_audit.png for the paired comparison!")