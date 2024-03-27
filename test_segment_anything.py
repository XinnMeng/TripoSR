import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything import SamPredictor, sam_model_registry

device = "cuda"

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    i = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    for ann in sorted_anns:
        box = ann['bbox']
        show_box(box, ax)
        i += 1
        if i > 10:
            break

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    



sam = sam_model_registry["vit_h"](checkpoint="/home/xin/lib/sam_vit_h_4b8939.pth")
sam.to(device=device)
mask_generator =  SamAutomaticMaskGenerator(
    model=sam,
    # points_per_side=32,
    # pred_iou_thresh=0.86,
    # stability_score_thresh=0.92,
    # crop_n_layers=1,
    # crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)
predictor = SamPredictor(sam)
image = cv2.imread("/home/xin/Dropbox/Reconstruction/TripoSR/laptop/1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(20,20))
# plt.imshow(image)
# plt.axis('off')
# plt.show()

start_time = time.time()
masks = mask_generator.generate(image)
# predictor.set_image(image)
# masks, _, _ = predictor.predict("bottle")

segment_time = time.time() - start_time
print("segment time: ", segment_time)
print(len(masks))
print(masks[0].keys())

# Pick top 10

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()

import ipdb; ipdb.set_trace()