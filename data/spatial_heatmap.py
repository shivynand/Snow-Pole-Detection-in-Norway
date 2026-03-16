import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import seaborn as sns

base_path = "/datasets/tdt4265/ad/open/Poles"

lidar_lbl_dir = f"{base_path}/lidar/labels/train"
rgb_lbl_dir   = f"{base_path}/rgb/labels/train"

def spatial_heatmap(label, title):
    x = []
    y = []
    count = 0 

    for label_file in os.listdir(label):
        if not label_file.endswith('.txt'): continue

        count += 1

        with open(os.path.join(label, label_file), 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5: continue

                x.append(float(parts[1]))
                y.append(float(parts[2]))

    plt.figure(figsize=(12, 4))

    hb = plt.hexbin(x, y, gridsize=30, cmap='inferno', mincnt=1)

    plt.gca().invert_yaxis()

    plt.colorbar(hb, label='Frequency of Poles')

    plt.title(f'Spatial Distribution of Poles Heatmap: {title}')
    plt.xlabel('Normalized X (Width)')
    plt.ylabel('Normalized Y (Height)')
    
    plt.show()

spatial_heatmap(lidar_lbl_dir, 'LiDAR Poles')
spatial_heatmap(rgb_lbl_dir, 'RGB Poles')