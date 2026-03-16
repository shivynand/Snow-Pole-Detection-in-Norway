import os
import numpy as np
import matplotlib.pyplot as plt

LABEL_DIR = "/datasets/tdt4265/ad/open/Poles/lidar/labels/train"

widths = []
heights = []

for lbl in os.listdir(LABEL_DIR):
    with open(os.path.join(LABEL_DIR, lbl), 'r') as f:
        for line in f:
            # YOLO: class x y width height
            _, _, _, w, h = map(float, line.split())
            widths.append(w)
            heights.append(h)

avg_w = np.mean(widths)
avg_h = np.mean(heights)
aspect_ratio = avg_h / avg_w

print(f"Total Poles Analyzed: {len(widths)}")
print(f"Average Normalized Width: {avg_w:.4f}")
print(f"Average Aspect Ratio (H/W): {aspect_ratio:.2f}")

# Seeing average trend for how wide poles are
if avg_w < 0.01:
    print("WARNING: Poles are extremely thin (<1% of image width). Consider high-res training!")

# Visualing resolution against pole width trade off
avg_normalized_width = 0.0072
resolutions = [320, 416, 640, 1024, 1280]
pixel_widths = [res * avg_normalized_width for res in resolutions]

plt.figure(figsize=(10, 6))
plt.plot(resolutions, pixel_widths, marker='o', linestyle='--', color='red')
plt.axhline(y=1.0, color='black', linestyle='-') 
plt.title("Expected Pole Width vs. Model Input Resolution")
plt.xlabel("YOLO Input Resolution (Pixels)")
plt.ylabel("Pole Width in Pixels")
plt.grid(True)
plt.savefig("resolution_tradeoff.png")