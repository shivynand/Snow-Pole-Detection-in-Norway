import os
import numpy as np
import matplotlib.pyplot as plt

ROADPOLES_LABEL_DIR = "/datasets/tdt4265/Poles2025/roadpoles_v1/train/labels"
IPHONE_LABEL_DIR = "/datasets/tdt4265/Poles2025/Road_poles_iPhone/labels/Train/train"

widths_roadpoles = []
heights_roadpoles = []

widths_iphone = []
heights_iphone = []

for lbl in os.listdir(ROADPOLES_LABEL_DIR):
    with open(os.path.join(ROADPOLES_LABEL_DIR, lbl), 'r') as f:
        for line in f:
            # YOLO: class x y width height
            _, _, _, w, h = map(float, line.split())
            widths_roadpoles.append(w)
            heights_roadpoles.append(h)

for lbl in os.listdir(IPHONE_LABEL_DIR):
    with open(os.path.join(IPHONE_LABEL_DIR, lbl), 'r') as f:
        for line in f:
            # YOLO: class x y width height
            _, _, _, w, h = map(float, line.split())
            widths_iphone.append(w)
            heights_iphone.append(h)

avg_wr = np.mean(widths_roadpoles)
avg_hr = np.mean(heights_roadpoles)

avg_wi = np.mean(widths_iphone)
avg_hi = np.mean(heights_iphone)

aspect_ratio_rp = avg_hr / avg_wr
aspect_ratio_ip = avg_hi / avg_wi

print(f"Total Poles Analyzed: {len(widths_roadpoles)}")
print(f"Average Normalized Width: {avg_wr:.4f}")
print(f"Average Aspect Ratio (H/W): {aspect_ratio_rp:.2f}")

print(f"Average Normalized Width iPhone: {avg_wi:.4f}")
print(f"Average Aspect Ratio iPhone (H/W): {aspect_ratio_ip:.2f}")

# Seeing average trend for how wide poles are
if avg_wr < 0.01:
    print("WARNING: RoadPoles_v1 poles are extremely thin (<1% of image width). Consider high-res training!")

if avg_wi < 0.01:
    print("WARNING: iPhone poles are extremely thin (<1% of image width). Consider high-res training!")

# Visualing resolution against pole width trade off
resolutions = [320, 416, 640, 1024, 1280]
pixel_widths_roadpoles = [res * avg_wr for res in resolutions]
pixel_widths_iphone = [res * avg_wi for res in resolutions]

plt.figure(figsize=(10, 6))
plt.plot(resolutions, pixel_widths_roadpoles,
         marker='o', linestyle='--', color='red',
         label='roadpoles_v1')
plt.plot(resolutions, pixel_widths_iphone,
         marker='s', linestyle='-', color='blue',
         label='iPhone')
plt.axhline(y=1.0, color='black', linestyle='-.', label='1 px')
plt.title("Pole width (pixels) vs YOLO input resolution")
plt.xlabel("Input resolution (px)")
plt.ylabel("Pole width (px)")
plt.legend()
plt.grid(True)
plt.savefig("resolution_tradeoff_both.png", dpi=150)
plt.show()