import cv2
import numpy as np
import mmcv
import shutil

from PIL import Image
from pathlib import Path
from tqdm import tqdm
from mmseg.apis import inference_segmentor, init_segmentor

from denred0_configs.deeplabv3plus.deeplabv3plus_r101_d8_512x512_40k_voc12aug import create_config
# from denred0_configs.pspnet.pspnet_r101_d8_512x512_40k_voc12aug import create_config
from classes import LASER_CLASSES, PALETTE
from utils import get_all_files_in_folder, create_border


def inference(exp_name, images_dir, checkpoint, images_ext, data_root_cfg, output_folder, create_visualization,
              device='cuda:0'):
    # create folders
    dirpath = output_folder.joinpath(exp_name)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    dirpath = output_folder.joinpath(exp_name).joinpath('visualization')
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    dirpath = output_folder.joinpath(exp_name).joinpath('images_result')
    Path(dirpath).mkdir(parents=True, exist_ok=True)

    # get config
    _, datasets, cfg = create_config(data_root=data_root_cfg, exp_name=exp_name)

    cfg.test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(512, 512),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **cfg.img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]
    cfg.data.test.pipeline = cfg.test_pipeline

    model = init_segmentor(cfg, checkpoint, device=device)

    # get all images
    images = get_all_files_in_folder(images_dir, images_ext)

    for image_path in tqdm(images):
        image = mmcv.imread(image_path)

        result = inference_segmentor(model, image)
        img_res = result[0].astype('uint8')
        seg_img = Image.fromarray(img_res).convert('P')
        seg_img.putpalette(np.array(PALETTE, dtype=np.uint8))
        seg_img.save(output_folder.joinpath(exp_name).joinpath('images_result').joinpath(image_path.name))

        if create_visualization:
            image_orig = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask_orig = cv2.imread('denred0_data/data_train_augmentation/masks_rgb/' + image_path.name, cv2.IMREAD_COLOR)
            mask_res = cv2.imread(
                str(output_folder.joinpath(exp_name).joinpath('images_result').joinpath(image_path.name)),
                cv2.IMREAD_COLOR)

            # create borders
            # image_orig = create_border(image_orig)
            # mask_orig = create_border(mask_orig)
            # mask_res = create_border(mask_res)

            # concatenate images
            vis = np.concatenate((image_orig, mask_orig, mask_res), axis=1)

            cv2.imwrite(str(output_folder.joinpath(exp_name).joinpath('visualization').joinpath(image_path.name)), vis)


if __name__ == '__main__':
    checkpoint = 'denred0_work_dirs/deeplabv3plus_r101_d8_512x512_40k_voc12aug_aug_dataset/iter_10000.pth'
    images_dir = Path('denred0_data/data_train/dataset/images')
    images_ext = ['*.png']
    data_root_cfg_train = 'denred0_data/data_train_augmentation/'
    exp_name = 'deeplabv3plus_r101_d8_512x512_40k_voc12aug_aug_dataset'
    device = 'cuda:0'
    output_folder = Path('denred0_data/inference')
    create_visualization = True

    inference(exp_name=exp_name,
              images_dir=images_dir,
              checkpoint=checkpoint,
              images_ext=images_ext,
              data_root_cfg=data_root_cfg_train,
              output_folder=output_folder,
              create_visualization=create_visualization,
              device=device)
