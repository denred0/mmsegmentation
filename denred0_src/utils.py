import cv2
from pathlib import Path
from skimage import exposure

import numpy as np
import shutil
from matplotlib import pyplot as plt


def get_all_files_in_folder(folder, types):
    files_grabbed = []
    for t in types:
        files_grabbed.extend(folder.rglob(t))
    files_grabbed = sorted(files_grabbed, key=lambda x: x)
    return files_grabbed


def to_txt():
    dir = Path('data/labels')
    # dir = Path('mmsegmentation/iccv09Data/labels')

    all_images = sorted(list(dir.glob('*.png')))
    print()

    for image_path in all_images:
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        gray = gray.astype(int)

        np.savetxt('data/labels_txt/' + image_path.stem + '.txt', gray, fmt='%d')  # use exponential notation

    #  img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    #
    # # print(np.unique(img_rgb))
    #  #print(img_rgb)
    #
    #  cv2.imwrite('data/labels_rgb/' + image_path.name, img_rgb)


def create_border(im, bordersize=2, color=[255, 0, 255]):
    border_image = cv2.copyMakeBorder(
        im,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=color
    )

    return border_image


def change_exposure(input_dir, images_ext, output_dir, concat_dir):
    dirpath = output_dir
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    dirpath = concat_dir
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(dirpath).mkdir(parents=True, exist_ok=True)

    files = get_all_files_in_folder(input_dir, images_ext)

    mask_path = Path('denred0_data/change_exposure/masks_rgb')

    for file in files:
        image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)

        mask= cv2.imread(str(mask_path.joinpath(file.name)), cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        image_eq = cv2.equalizeHist(image)
        cv2.imwrite(str(output_dir.joinpath(file.name)), image_eq)

        # create borders
        image_eq = create_border(image_eq, bordersize=1)
        image = create_border(image, bordersize=1)

        # concatenate images
        vis = np.concatenate((image, image_eq), axis=1)

        cv2.imwrite(str(concat_dir.joinpath(file.name)), vis)
        #
        # ret, th1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        # th2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
        #                            cv2.THRESH_BINARY, 11, 1.2)
        # th3 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
        #                            cv2.THRESH_BINARY, 11, 2)
        # titles = ['Original Image', 'Global Thresholding (v = 127)',
        #           'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
        # images = [image, mask, th2, th3]
        # for i in range(4):
        #     plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        #     plt.title(titles[i])
        #     plt.xticks([]), plt.yticks([])
        # plt.show()




if __name__ == '__main__':
    input_dir = Path('denred0_data/change_exposure/input')
    output_dir = Path('denred0_data/change_exposure/output')
    images_ext = ['*.png']
    concat_dir = Path('denred0_data/change_exposure/concatenate')

    change_exposure(input_dir=input_dir,
                    images_ext=images_ext,
                    output_dir=output_dir,
                    concat_dir=concat_dir)
