import numpy as np
from pathlib import Path
import shutil
import os
from PIL import Image
from tqdm import tqdm

from denred0_src.classes import LASER_CLASSES, PALETTE
from utils import get_all_files_in_folder

import cv2


def sort_images_per_classes(data_dir, output_dir, images_ext):
    dirpath = output_dir
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(dirpath).mkdir(parents=True, exist_ok=True)

    all_images = get_all_files_in_folder(data_dir.joinpath('images'), images_ext)
    all_masks = get_all_files_in_folder(data_dir.joinpath('masks'), images_ext)

    assert len(all_images) == len(all_masks), f"Masks an images count " \
                                              f"should be equal: {len(all_images)} and {len(all_masks)}"
    assert len(all_images) != 0, "No images found! Check your paths"

    print('Total images {}'.format(len(all_images)))

    classes_count = {}
    without_defects = []

    for cl in LASER_CLASSES:
        Path(output_dir).joinpath(cl).mkdir(parents=True, exist_ok=True)
        Path(output_dir).joinpath(cl).joinpath('images').mkdir(parents=True, exist_ok=True)
        Path(output_dir).joinpath(cl).joinpath('masks').mkdir(parents=True, exist_ok=True)
        Path(output_dir).joinpath(cl).joinpath('masks_rgb').mkdir(parents=True, exist_ok=True)

    for index, value in enumerate(LASER_CLASSES):
        classes_count[index] = 0

    # palette = [[255, 255, 255], [0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0]]

    for mask_path in tqdm(all_masks):
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gt_class = list(np.unique(gt_mask))

        if len(gt_class) == 1 and gt_class[0] == 0:
            without_defects.append(str(mask_path))

        gt_mask = np.stack((gt_mask,) * 3, axis=-1)

        for cl in gt_class:
            shutil.copy(data_dir.joinpath('images').joinpath(mask_path.name),
                        output_dir.joinpath(LASER_CLASSES[cl]).joinpath('images'))
            shutil.copy(data_dir.joinpath('masks').joinpath(mask_path.name),
                        output_dir.joinpath(LASER_CLASSES[cl]).joinpath('masks'))

            seg_img = Image.fromarray(gt_mask[:, :, 0]).convert('P')
            seg_img.putpalette(np.array(PALETTE, dtype=np.uint8))
            seg_img.save(output_dir.joinpath(LASER_CLASSES[cl]).joinpath('masks_rgb').joinpath(mask_path.name))

            classes_count[cl] += 1

    print()
    print('Images per labels:')
    for key, cl_count in classes_count.items():
        print('Label ' + str(key) + ' (' + LASER_CLASSES[key] + '):   ' + str(cl_count))

    print()
    print('Images without defects', len(without_defects))


if __name__ == '__main__':
    data_dir = Path('denred0_data/dataset_analyze/input')
    output_dir = Path('denred0_data/dataset_analyze/output')
    images_ext = ['*.png']

    sort_images_per_classes(data_dir=data_dir,
                            output_dir=output_dir,
                            images_ext=images_ext)
