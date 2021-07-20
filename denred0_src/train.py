from mmseg.apis import train_segmentor
from denred0_configs.deeplabv3plus.deeplabv3plus_r101_d8_512x512_40k_voc12aug import create_config

# from denred0_configs.hrnet.fcn_hr18_512x512_160k_ade20k import create_config
# from denred0_configs.deeplabv3plus.deeplabv3plus_r101_d8_512x512_40k_voc12aug_visual_transformer import create_config
# from denred0_configs.unet.deeplabv3_unet_s5_d16_64x64_40k_drive import create_config
# from denred0_configs.pspnet.pspnet_r101_d8_512x512_40k_voc12aug import create_config

data_root = 'denred0_data/data_train_augmentation'
exp_name = 'deeplabv3plus_r101_d8_512x512_40k_voc12aug_aug_dataset'
model, datasets, cfg = create_config(data_root=data_root, exp_name=exp_name, do_test_split=True)
train_segmentor(model, datasets, cfg, distributed=False, validate=True, meta=dict())
