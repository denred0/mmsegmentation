import os
import json
import cv2
from PIL import Image
import os.path as osp

import numpy as np

from tqdm import tqdm

from src.classes import LASER_CLASSES

data_folder = "data/data_new/images/"
dataset_root = "data/data_new/output/"
img_ext = 'png'

# class_list = [
#     "blue",
#     "gray",
# ]

files_list = os.listdir(data_folder)
json_list = [x for x in files_list if ".json" in x]

for j in tqdm(json_list):

    orig_image_path = data_folder + j.split(".")[0] + '.' + img_ext
    orig_image = cv2.imread(orig_image_path)
    orig_image_height, orig_image_width = orig_image.shape[:2]

    mask_image = np.zeros((orig_image_height, orig_image_width, 3), dtype=np.uint8)

    with open("{}{}".format(data_folder, j)) as label_json:
        labels = json.load(label_json)["shapes"]

    for l in labels:
        class_index = LASER_CLASSES.index(l["label"])
        points = l["points"]

        if l["shape_type"] == "polygon" or l["shape_type"] == "linestrip":
            contour = [np.array(points, dtype=np.int32)]
            cv2.drawContours(mask_image, [contour[0]], 0, (class_index, class_index, class_index), -1)
        elif l["shape_type"] == "rectangle":
            cv2.rectangle(mask_image, (int(points[0][0]), int(points[0][1])), (int(points[1][0]), int(points[1][1])),
                          (class_index, class_index, class_index), -1)

    out_mask_path = "{}masks/{}.png".format(dataset_root, j.split(".")[0])
    out_image_path = "{}imgs/{}.png".format(dataset_root, j.split(".")[0])

    cv2.imwrite(out_mask_path, mask_image)
    cv2.imwrite(out_image_path, orig_image)

    classes = tuple(LASER_CLASSES)
    palette = [[255, 255, 255], [0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0]]



    seg_img = Image.fromarray(mask_image[:,:,0]).convert('P')
    seg_img.putpalette(np.array(palette, dtype=np.uint8))
    out_mask_vis_path = "{}masks_rgb/{}.png".format(dataset_root, j.split(".")[0])
    seg_img.save(out_mask_vis_path)




