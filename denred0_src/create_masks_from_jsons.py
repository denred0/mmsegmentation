import os
import json
import cv2
from pathlib import Path
from PIL import Image
import os.path as osp

import numpy as np
import shutil

from tqdm import tqdm

from denred0_src.classes import LASER_CLASSES, PALETTE
from utils import get_all_files_in_folder


def convert_json_to_masks(input_dir, output_dir, img_ext):
    # create folders
    dirpath = output_dir
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    dirpath = output_dir.joinpath('images')
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    dirpath = output_dir.joinpath('masks')
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    dirpath = output_dir.joinpath('masks_rgb')
    Path(dirpath).mkdir(parents=True, exist_ok=True)

    files_json = get_all_files_in_folder(input_dir, ['*.json'])

    for f_json in tqdm(files_json):

        orig_image_path = input_dir.joinpath(f_json.stem + img_ext)
        orig_image = cv2.imread(str(orig_image_path), cv2.IMREAD_COLOR)
        orig_image_height, orig_image_width = orig_image.shape[:2]

        mask_image = np.zeros((orig_image_height, orig_image_width, 3), dtype=np.uint8)

        with open(str(f_json)) as label_json:
            labels = json.load(label_json)["shapes"]

        for l in labels:
            class_index = LASER_CLASSES.index(l["label"])
            points = l["points"]

            if l["shape_type"] == "polygon" or l["shape_type"] == "linestrip":
                contour = [np.array(points, dtype=np.int32)]
                cv2.drawContours(mask_image, [contour[0]], 0, (class_index, class_index, class_index), -1)
            elif l["shape_type"] == "rectangle":
                cv2.rectangle(mask_image, (int(points[0][0]), int(points[0][1])),
                              (int(points[1][0]), int(points[1][1])),
                              (class_index, class_index, class_index), -1)

        # out_mask_path = "{}masks/{}.png".format(dataset_root, j.split(".")[0])
        # out_image_path = "{}imgs/{}.png".format(dataset_root, j.split(".")[0])
        cv2.imwrite(str(output_dir.joinpath('images').joinpath(f_json.stem + img_ext)), orig_image)
        cv2.imwrite(str(output_dir.joinpath('masks').joinpath(f_json.stem + img_ext)), mask_image)

        # classes = tuple(LASER_CLASSES)

        seg_img = Image.fromarray(mask_image[:, :, 0]).convert('P')
        seg_img.putpalette(np.array(PALETTE, dtype=np.uint8))
        # out_mask_vis_path = "{}masks_rgb/{}.png".format(dataset_root, j.split(".")[0])
        seg_img.save(output_dir.joinpath('masks_rgb').joinpath(f_json.stem + img_ext))


if __name__ == '__main__':
    input_dir = Path("denred0_data/create_masks_from_json/images_and_json")
    output_dir = Path("denred0_data/create_masks_from_json/output")
    img_ext = '.png'

    convert_json_to_masks(input_dir=input_dir, output_dir=output_dir, img_ext=img_ext)
