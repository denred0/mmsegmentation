import cv2
import numpy as np
import pandas as pd
import shutil

from tqdm import tqdm
from pathlib import Path
from utils import get_all_files_in_folder
from sklearn.model_selection import train_test_split


def create_splits_files(root_dir, test_split):
    train_dir = Path('denred0_data/train_test_split/train')
    if train_dir.exists() and train_dir.is_dir():
        shutil.rmtree(train_dir)
    Path(train_dir).mkdir(parents=True, exist_ok=True)

    train_images_dir = Path('denred0_data/train_test_split/train/images')
    if train_images_dir.exists() and train_images_dir.is_dir():
        shutil.rmtree(train_images_dir)
    Path(train_images_dir).mkdir(parents=True, exist_ok=True)

    train_masks_dir = Path('denred0_data/train_test_split/train/masks')
    if train_masks_dir.exists() and train_masks_dir.is_dir():
        shutil.rmtree(train_masks_dir)
    Path(train_masks_dir).mkdir(parents=True, exist_ok=True)

    train_masks_rgb_dir = Path('denred0_data/train_test_split/train/masks_rgb')
    if train_masks_rgb_dir.exists() and train_masks_rgb_dir.is_dir():
        shutil.rmtree(train_masks_rgb_dir)
    Path(train_masks_rgb_dir).mkdir(parents=True, exist_ok=True)

    test_dir = Path('denred0_data/train_test_split/test')
    if test_dir.exists() and test_dir.is_dir():
        shutil.rmtree(test_dir)
    Path(test_dir).mkdir(parents=True, exist_ok=True)

    test_images_dir = Path('denred0_data/train_test_split/test/images')
    if test_images_dir.exists() and test_images_dir.is_dir():
        shutil.rmtree(test_images_dir)
    Path(test_images_dir).mkdir(parents=True, exist_ok=True)

    test_masks_dir = Path('denred0_data/train_test_split/test/masks')
    if test_masks_dir.exists() and test_masks_dir.is_dir():
        shutil.rmtree(test_masks_dir)
    Path(test_masks_dir).mkdir(parents=True, exist_ok=True)

    test_masks_rgb_dir = Path('denred0_data/train_test_split/test/masks_rgb')
    if test_masks_rgb_dir.exists() and test_masks_rgb_dir.is_dir():
        shutil.rmtree(test_masks_rgb_dir)
    Path(test_masks_rgb_dir).mkdir(parents=True, exist_ok=True)

    all_masks = get_all_files_in_folder(root_dir.joinpath('masks'), ['*.png'])

    labels = []
    images_list = []
    for msk in tqdm(all_masks):
        mask = cv2.imread(str(msk), cv2.IMREAD_GRAYSCALE)

        classes = np.unique(mask)
        for cl in classes:
            labels.append(cl)
            images_list.append(msk.stem)

    # classes + counts
    labels_dict = pd.DataFrame(labels, columns=["x"]).groupby('x').size().to_dict()
    all_labels = sum(labels_dict.values())

    labels_parts = []
    for key, value in labels_dict.items():
        labels_parts.append(value / all_labels)

    straify = False
    min_part = 0.2
    if np.min(labels_parts) < min_part:
        straify = True

    # val_part = 0.2

    type = 2

    if straify:

        if type == 1:
            # add 0.05 for accuracy stratification
            test_split += 0.2

            # collect all classes

            # stratify
            X_train, X_test, y_train, y_test = train_test_split(images_list, labels, test_size=test_split,
                                                                random_state=42,
                                                                stratify=labels, shuffle=True)
            # remove dublicates
            X_train = np.unique(X_train).tolist()
            X_test = np.unique(X_test).tolist()

            # get images that exist in train and test
            dublicates = []
            for xtr in tqdm(X_train):
                for xtt in X_test:
                    if xtr == xtt:
                        dublicates.append(xtr)

            # delete such images from train and test
            for dubl in dublicates:
                X_train.remove(dubl)
                X_test.remove(dubl)

            # add dubl images in train and test with stratify
            for i, dubl in tqdm(enumerate(dublicates)):
                if i % int((10 - (test_split) * 10)) == 0:
                    X_test.append(dubl)
                else:
                    X_train.append(dubl)

            # copy images and txts
            for name in tqdm(X_train):
                shutil.copy(root_dir.joinpath('images').joinpath(name + '.png'), train_images_dir)
                shutil.copy(root_dir.joinpath('masks').joinpath(name + '.png'), train_masks_dir)
                shutil.copy(root_dir.joinpath('masks_rgb').joinpath(name + '.png'), train_masks_rgb_dir)

            for name in tqdm(X_test):
                shutil.copy(root_dir.joinpath('images').joinpath(name + '.png'), test_images_dir)
                shutil.copy(root_dir.joinpath('masks').joinpath(name + '.png'), test_masks_dir)
                shutil.copy(root_dir.joinpath('masks_rgb').joinpath(name + '.png'), test_masks_rgb_dir)

            # check stratification
            all_train_masks = get_all_files_in_folder(train_masks_dir, ['*.png'])

            # collect train classes and compare with all classes
            labels_train = []
            for msk in tqdm(all_train_masks):
                mask = cv2.imread(str(msk), cv2.IMREAD_GRAYSCALE)

                classes = np.unique(mask)
                for cl in classes:
                    labels_train.append(cl)

            labels_train_dict = pd.DataFrame(labels_train, columns=["x"]).groupby('x').size().to_dict()

            st = []
            for key, value in labels_dict.items():
                val = labels_train_dict[key] / value
                st.append(val)

                print(f'Class {key} | counts {value} | train_part {val}')

            print('Train part:', np.mean(st))
        else:
            labels_dict[-1] = 99999999

            # assign to image one class - rarest class
            x_all = []
            labels_all = []
            for msk in tqdm(all_masks):
                mask = cv2.imread(str(msk), cv2.IMREAD_GRAYSCALE)

                classes = np.unique(mask)
                # for cl in classes:
                #     labels_train.append(cl)
                #
                #
                # lines = loadtxt(str(txt), delimiter=' ', unpack=False).tolist()

                # lab = []
                # for line in lines:
                #     lab.append(line[0])

                best_cat = -1
                x_all.append(msk.stem)
                for l in classes:
                    if labels_dict[l] < labels_dict[best_cat]:
                        best_cat = l
                labels_all.append(best_cat)

            # stratify
            X_train, X_test, y_train, y_test = train_test_split(x_all, labels_all, test_size=test_split,
                                                                random_state=42,
                                                                stratify=labels_all, shuffle=True)

            # copy images and txts
            for name in tqdm(X_train):
                shutil.copy(root_dir.joinpath('images').joinpath(name + '.png'), train_images_dir)
                shutil.copy(root_dir.joinpath('masks').joinpath(name + '.png'), train_masks_dir)
                shutil.copy(root_dir.joinpath('masks_rgb').joinpath(name + '.png'), train_masks_rgb_dir)

            for name in tqdm(X_test):
                shutil.copy(root_dir.joinpath('images').joinpath(name + '.png'), test_images_dir)
                shutil.copy(root_dir.joinpath('masks').joinpath(name + '.png'), test_masks_dir)
                shutil.copy(root_dir.joinpath('masks_rgb').joinpath(name + '.png'), test_masks_rgb_dir)

            # check stratification
            all_train_masks = get_all_files_in_folder(train_masks_dir, ['*.png'])

            # collect train classes and compare with all classes
            labels_train = []
            for msk in tqdm(all_train_masks):
                mask = cv2.imread(str(msk), cv2.IMREAD_GRAYSCALE)

                classes = np.unique(mask)
                for cl in classes:
                    labels_train.append(cl)

            labels_train_dict = pd.DataFrame(labels_train, columns=["x"]).groupby('x').size().to_dict()

            st = []
            labels_dict.pop(-1)
            for key, value in labels_dict.items():
                val = labels_train_dict[key] / value
                st.append(val)

                print(f'Class {key} | counts {value} | test_part {val}')

            print('Train part:', np.mean(st))

    else:

        np.random.shuffle(all_masks)
        train_FileNames, val_FileNames = np.split(np.array(all_masks), [int(len(all_masks) * (1 - test_split))])

        for name in tqdm(train_FileNames):
            shutil.copy(root_dir.joinpath('images').joinpath(name + '.png'), train_images_dir)
            shutil.copy(root_dir.joinpath('masks').joinpath(name + '.png'), train_masks_dir)
            shutil.copy(root_dir.joinpath('masks_rgb').joinpath(name + '.png'), train_masks_rgb_dir)

        for name in tqdm(val_FileNames):
            shutil.copy(root_dir.joinpath('images').joinpath(name + '.png'), test_images_dir)
            shutil.copy(root_dir.joinpath('masks').joinpath(name + '.png'), test_masks_dir)
            shutil.copy(root_dir.joinpath('masks_rgb').joinpath(name + '.png'), test_masks_rgb_dir)

    split_dir = root_dir.joinpath('splits')
    if split_dir.exists() and split_dir.is_dir():
        shutil.rmtree(split_dir)
    Path(split_dir).mkdir(parents=True, exist_ok=True)

    all_masks_train = get_all_files_in_folder(train_masks_dir, ['*.png'])
    all_masks_test = get_all_files_in_folder(test_masks_dir, ['*.png'])

    with open(split_dir.joinpath('train.txt'), 'w') as f:
        for m in all_masks_train:
            f.write(m.stem + '\n')

    with open(split_dir.joinpath('val.txt'), 'w') as f:
        for m in all_masks_test:
            f.write(m.stem + '\n')


if __name__ == '__main__':
    create_splits_files(root_dir=Path('denred0_data/data_train_augmentation'), test_split=0.2)
