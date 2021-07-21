import cv2
import shutil
import numpy as np

from tqdm import tqdm
from pathlib import Path
from PIL import ImageFont, ImageDraw, Image
from utils import get_all_files_in_folder

images_result = get_all_files_in_folder(Path('denred0_test_results'), ['*.png'])

images_source_path = 'denred0_data/data_train_augmentation'
img_size = 512

dirpath = Path('denred0_data/visualization')
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)
Path(dirpath).mkdir(parents=True, exist_ok=True)

for im_path in tqdm(images_result):
    if "aug" not in im_path.name:
        img_orig = cv2.imread(images_source_path + '/images/' + im_path.name, cv2.IMREAD_COLOR)
        img_mask = cv2.imread(images_source_path + '/masks_rgb/' + im_path.name, cv2.IMREAD_COLOR)

        img_pred = cv2.imread(str(im_path), cv2.IMREAD_COLOR)
        img_text = np.zeros([img_size, img_size, 3], dtype=np.uint8)
        img_text[:] = 255

        cv2.rectangle(img_text, (50, 20), (100, 70), (0, 0, 0), 2)
        cv2.rectangle(img_text, (50, 120), (100, 170), (255, 0, 0), -1)
        cv2.rectangle(img_text, (50, 220), (100, 270), (0, 0, 255), -1)
        cv2.rectangle(img_text, (50, 320), (100, 370), (0, 255, 0), -1)
        cv2.rectangle(img_text, (50, 420), (100, 470), (0, 255, 255), -1)

        cv2.putText(img_text, 'No defect', (120, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img_text, 'Risunok', (120, 155), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img_text, 'Morshiny', (120, 255), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img_text, 'Nadav', (120, 355), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img_text, 'Izlom', (120, 455), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # b, g, r, a = 0, 255, 0, 0
        # fontpath = "CYRIL1.TTF"
        # font = ImageFont.truetype(fontpath, 48)
        # img_pil = Image.fromarray(img_text)
        # draw = ImageDraw.Draw(img_pil)
        # draw.text((110, 80), "парпарап", font=font, fill=(b, g, r, a))
        # img_text = np.array(img_pil)

        img_one_line = np.concatenate((img_orig, img_mask), axis=1)

        img_top = np.concatenate((img_orig, img_text), axis=1)
        img_bottom = np.concatenate((img_mask, img_pred), axis=1)
        img_result = np.concatenate((img_top, img_bottom), axis=0)

        cv2.imwrite(str(dirpath.joinpath(im_path.name)), img_one_line)
