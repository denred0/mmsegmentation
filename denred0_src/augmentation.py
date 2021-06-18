import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
from PIL import Image

from denred0_src.classes import LASER_CLASSES, PALETTE
from utils import get_all_files_in_folder

import albumentations as A
import shutil
import cv2


def create_augmented_imgs_for_class(data_dir, label, images_ext, aug_count=1):
    # clear folder
    dirpath = Path(data_dir).joinpath('aug')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(data_dir).joinpath('aug').mkdir(parents=True, exist_ok=True)
    Path(data_dir).joinpath('aug').joinpath('images').mkdir(parents=True, exist_ok=True)
    Path(data_dir).joinpath('aug').joinpath('masks').mkdir(parents=True, exist_ok=True)
    Path(data_dir).joinpath('aug').joinpath('masks_rgb').mkdir(parents=True, exist_ok=True)

    all_images = get_all_files_in_folder(data_dir.joinpath('images'), images_ext)
    all_masks = get_all_files_in_folder(data_dir.joinpath('masks'), images_ext)

    assert len(all_images) == len(all_masks), f"Masks an images count " \
                                              f"should be equal: {len(all_images)} and {len(all_masks)}"
    assert len(all_images) != 0, "No images found! Check your paths"

    print('\nClass {}. Total images {}'.format(label, len(all_images)))
    print(f'Augmentation {label}...')

    transform = A.Compose([
        A.Rotate(limit=35, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
    ], p=1)

    for i, (img_path, mask_path) in tqdm(enumerate(zip(all_images, all_masks)), total=len(all_images)):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        for idx in range(aug_count):
            transformed = transform(image=img, mask=mask)
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']

            cv2.imwrite(
                str(Path(data_dir).joinpath('aug').joinpath('images').joinpath(
                    img_path.stem + '_aug_' + str(idx) + '.png')),
                transformed_image)

            cv2.imwrite(
                str(Path(data_dir).joinpath('aug').joinpath('masks').joinpath(
                    img_path.stem + '_aug_' + str(idx) + '.png')),
                transformed_mask)

            transformed_mask = np.stack((transformed_mask,) * 3, axis=-1)
            seg_img = Image.fromarray(transformed_mask[:, :, 0]).convert('P')
            seg_img.putpalette(np.array(PALETTE, dtype=np.uint8))
            seg_img.save(
                str(Path(data_dir).joinpath('aug').joinpath('masks_rgb').joinpath(
                    img_path.stem + '_aug_' + str(idx) + '.png')))


def create_augmented_imgs_for_dataset(data_dir, images_ext, aug_count=1):
    # clear folder
    dirpath = Path(data_dir).joinpath('aug')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(data_dir).joinpath('aug').mkdir(parents=True, exist_ok=True)
    Path(data_dir).joinpath('aug').joinpath('images').mkdir(parents=True, exist_ok=True)
    Path(data_dir).joinpath('aug').joinpath('masks').mkdir(parents=True, exist_ok=True)
    Path(data_dir).joinpath('aug').joinpath('masks_rgb').mkdir(parents=True, exist_ok=True)

    all_images = get_all_files_in_folder(data_dir.joinpath('images'), images_ext)
    all_masks = get_all_files_in_folder(data_dir.joinpath('masks'), images_ext)

    assert len(all_images) == len(all_masks), f"Masks an images count " \
                                              f"should be equal: {len(all_images)} and {len(all_masks)}"
    assert len(all_images) != 0, "No images found! Check your paths"

    print('\nTotal images {}'.format(len(all_images)))
    print('Augmentation...')

    transform = A.Compose(
        [A.OneOf([
            # A.Rotate(limit=35, p=1),
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
            A.Transpose(p=1)], p=1)], p=1)

    for i, (img_path, mask_path) in tqdm(enumerate(zip(all_images, all_masks)), total=len(all_images)):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        for idx in range(aug_count):
            transformed = transform(image=img, mask=mask)
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']

            cv2.imwrite(
                str(Path(data_dir).joinpath('aug').joinpath('images').joinpath(
                    img_path.name + '_aug_' + str(idx) + '.png')),
                transformed_image)

            cv2.imwrite(
                str(Path(data_dir).joinpath('aug').joinpath('masks').joinpath(
                    img_path.name + '_aug_' + str(idx) + '.png')),
                transformed_mask)

            transformed_mask = np.stack((transformed_mask,) * 3, axis=-1)
            seg_img = Image.fromarray(transformed_mask[:, :, 0]).convert('P')
            seg_img.putpalette(np.array(PALETTE, dtype=np.uint8))
            seg_img.save(
                str(Path(data_dir).joinpath('aug').joinpath('masks_rgb').joinpath(
                    img_path.name + '_aug_' + str(idx) + '.png')))


if __name__ == '__main__':
    aug_count = 2
    data_dir = Path('denred0_data/augmentation/images_per_classes')
    images_ext = ['*.png']
    for cl in LASER_CLASSES:
        class_data_dir = data_dir.joinpath(cl)
        create_augmented_imgs_for_class(data_dir=class_data_dir,
                                        label=cl,
                                        images_ext=images_ext,
                                        aug_count=aug_count)

    # data_dir_dataset = Path('denred0_data/augmentation/whole_dataset')
    # images_ext = ['*.png']
    # create_augmented_imgs_for_dataset(data_dir=data_dir_dataset,
    #                                   images_ext=images_ext,
    #                                   aug_count=aug_count)
