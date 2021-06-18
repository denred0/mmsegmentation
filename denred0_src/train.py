from mmseg.apis import train_segmentor
from denred0_configs.deeplabv3plus.deeplabv3plus_r101_d8_512x512_40k_voc12aug import create_config

model, datasets, cfg = create_config()
train_segmentor(model, datasets, cfg, distributed=False, validate=True, meta=dict())
