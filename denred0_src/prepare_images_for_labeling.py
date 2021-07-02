import cv2
import shutil
import os
import numpy as np

from pathlib import Path
from tqdm import tqdm

from utils import get_all_files_in_folder


def create_images_for_labeling():
    dirpath = Path('denred0_data/prepare_images_for_labeling/result')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(dirpath).mkdir(parents=True, exist_ok=True)

    dirpath = Path('denred0_data/prepare_images_for_labeling/not_labeled')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(dirpath).mkdir(parents=True, exist_ok=True)

    images_visual = get_all_files_in_folder(Path('denred0_data/prepare_images_for_labeling/visualization'), ['*.png'])

    root_directory = 'denred0_data/prepare_images_for_labeling/all_data'
    for subdir, dirs, files in os.walk(root_directory):
        for folder in dirs:

            images = get_all_files_in_folder(Path(root_directory).joinpath(folder), ['*.png'])

            for im in tqdm(images):
                result_image = np.zeros((1024, 1024, 3), dtype=int)

                image_dequs = cv2.imread(str(im), cv2.IMREAD_COLOR)

                orig_image = image_dequs[:512, :512, :]
                image_mark_dequs = image_dequs[512:1024, :512, :]
                cv2.putText(image_mark_dequs, folder, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                            cv2.LINE_AA)

                image_mark = np.zeros((512, 512, 3), dtype=int)
                image_mark_model = np.zeros((512, 512, 3), dtype=int)
                for imvisual in images_visual:
                    if imvisual.stem == im.stem:
                        image_vis = cv2.imread(str(imvisual), cv2.IMREAD_COLOR)

                        image_mark = image_vis[:, 512:1024, :]
                        image_mark_model = image_vis[:, 1024:, :]

                result_image[:512, :512, :] = orig_image
                result_image[:512, 512:, :] = image_mark_dequs
                result_image[512:, :512, :] = image_mark
                result_image[512:, 512:, :] = image_mark_model
                cv2.imwrite('denred0_data/prepare_images_for_labeling/result/' + im.name, result_image)

                if np.sum(image_mark) == 0:
                    cv2.imwrite('denred0_data/prepare_images_for_labeling/not_labeled/' + im.name, orig_image)


if __name__ == '__main__':
    create_images_for_labeling()
