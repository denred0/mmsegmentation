from mmseg.apis import train_segmentor
from denred0_configs.deeplabv3plus.deeplabv3plus_r101_d8_512x512_40k_voc12aug import create_config

data_root = 'denred0_data/data_train_augmentation/'
exp_name = 'deeplabv3plus_r101-d8_512x512_40k_voc12aug_aug_dataset'
model, datasets, cfg = create_config(data_root=data_root,
                                     exp_name=exp_name)
train_segmentor(model, datasets, cfg, distributed=False, validate=True, meta=dict())
