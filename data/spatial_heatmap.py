import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

base_path = "/datasets/tdt4265/Poles2025"

v1_lbl_dir = f"{base_path}/roadpoles_v1/train/labels"
iphone_lbl_dir = f"{base_path}/Road_poles_iPhone/labels/Train/train"

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
    
    filename = f'spatial_heatmap_{title.replace(" ", "_").lower()}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

spatial_heatmap(v1_lbl_dir, 'Dashcam Poles')
spatial_heatmap(iphone_lbl_dir, 'iPhone Poles')

