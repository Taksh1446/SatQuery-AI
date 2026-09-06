import json
from PIL import Image

DATA_DIR = "data/vrsbench"

with open(f"{DATA_DIR}/VRSBench_EVAL_referring.json") as f:
    referring = json.load(f)

entry = referring[0]
print("Ground truth:", entry["ground_truth"])
print("Image id:", entry["image_id"])

img_path = f"{DATA_DIR}/Images_val/Images_val/{entry['image_id']}"
img = Image.open(img_path)
print("Actual image size:", img.size)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re

# Parse the <x1><y1><x2><y2> string into numbers
coords = [int(x) for x in re.findall(r'\d+', entry["ground_truth"])]
x1, y1, x2, y2 = coords
print("Parsed coords (0-100 scale):", x1, y1, x2, y2)

# Convert to pixels
w, h = img.size
px1, py1, px2, py2 = x1/100*w, y1/100*h, x2/100*w, y2/100*h

fig, ax = plt.subplots()
ax.imshow(img)
rect = patches.Rectangle((px1, py1), px2-px1, py2-py1, linewidth=2, edgecolor='red', facecolor='none')
ax.add_patch(rect)
ax.set_title(entry["question"])
plt.show()