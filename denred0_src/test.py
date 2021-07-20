import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

from denred0_configs.deeplabv3plus.deeplabv3plus_r101_d8_512x512_40k_voc12aug import create_config

data_root = 'denred0_data/data_train_augmentation'
exp_name = 'deeplabv3plus_r101_d8_512x512_40k_voc12aug_aug_dataset'
model, datasets, cfg = create_config(data_root=data_root, exp_name=exp_name)
print()

# cfg = mmcv.Config.fromfile('denred0_configs/deeplabv3plus/deeplabv3plus_r101_d8_512x512_40k_voc12aug_visual_transformer.py')
# # if args.options is not None:
# #     cfg.merge_from_dict(args.options)
# set cudnn_benchmark
if cfg.get('cudnn_benchmark', False):
    torch.backends.cudnn.benchmark = True

aug_test = False
if aug_test:
    # hard code index
    cfg.data.test.pipeline[1].img_ratios = [
        0.5, 0.75, 1.0, 1.25, 1.5, 1.75
    ]
    cfg.data.test.pipeline[1].flip = True
cfg.model.pretrained = None
cfg.data.test.test_mode = True

# init distributed env first, since logger depends on the dist info.
launcher = 'none'
# if launcher == 'none':
distributed = False

# build the dataloader
# TODO: support multiple images per gpu (only minor changes are needed)
dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=distributed,
    shuffle=False)

# build the model and load checkpoint
cfg.model.train_cfg = None
checkpoint = 'denred0_work_dirs/deeplabv3plus_r101_d8_512x512_40k_voc12aug_aug_dataset/iter_40000.pth'

model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
fp16_cfg = cfg.get('fp16', None)
if fp16_cfg is not None:
    wrap_fp16_model(model)
checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
if 'CLASSES' in checkpoint.get('meta', {}):
    model.CLASSES = checkpoint['meta']['CLASSES']
else:
    print('"CLASSES" not found in meta, use dataset.CLASSES instead')
    model.CLASSES = dataset.CLASSES
if 'PALETTE' in checkpoint.get('meta', {}):
    model.PALETTE = checkpoint['meta']['PALETTE']
else:
    print('"PALETTE" not found in meta, use dataset.PALETTE instead')
    model.PALETTE = dataset.PALETTE

efficient_test = False
# if args.eval_options is not None:
#     efficient_test = args.eval_options.get('efficient_test', False)

show = True
opacity = 0.5
show_dir = 'denred0_test_results'

# if not distributed:
model = MMDataParallel(model, device_ids=[0])
outputs = single_gpu_test(model, data_loader, show, show_dir,
                          efficient_test, opacity)
# else:
#     model = MMDistributedDataParallel(
#         model.cuda(),
#         device_ids=[torch.cuda.current_device()],
#         broadcast_buffers=False)
#     outputs = multi_gpu_test(model, data_loader, args.tmpdir,
#                              args.gpu_collect, efficient_test)
# eval = 'mAP'
eval = ['mIoU', 'mDice', 'mFscore']
rank, _ = get_dist_info()
if rank == 0:
    # if args.out:
    #     print(f'\nwriting results to {args.out}')
    #     mmcv.dump(outputs, args.out)
    kwargs = {} #if args.eval_options is None else args.eval_options
    # if args.format_only:
    #     dataset.format_results(outputs, **kwargs)
    if eval:
        dataset.evaluate(outputs, eval, **kwargs)
