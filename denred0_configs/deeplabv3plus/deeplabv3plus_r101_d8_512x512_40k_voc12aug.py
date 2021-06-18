import os.path as osp
import mmcv

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

from mmcv import Config
from mmseg.apis import set_random_seed

from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor, inference_segmentor

from denred0_src.classes import LASER_CLASSES, PALETTE


def create_config():
    # convert dataset annotation to semantic segmentation map
    data_root = 'denred0_data/data/'
    img_dir = 'imgs'
    ann_dir = 'masks_rgb'
    split_dir = 'splits'
    test_split = 0.2
    model_name = 'deeplabv3plus_r101-d8_512x512_40k_voc12aug'

    # define class and palette for better visualization
    classes = tuple(LASER_CLASSES)  # ('background', 'picture', 'pushed', 'wrinkle', 'break')
    # palette = [[255, 255, 255], [0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0]]
    # palette = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]

    # for file in mmcv.scandir(osp.join(data_root, ann_dir), suffix='.txt'):
    #     seg_map = np.loadtxt(osp.join(data_root, ann_dir, file)).astype(np.uint8)
    #     seg_img = Image.fromarray(seg_map).convert('P')
    #     seg_img.putpalette(np.array(palette, dtype=np.uint8))
    #     seg_img.save(osp.join(data_root, ann_dir, file.replace('.txt', '.png')))

    # for file in mmcv.scandir(osp.join(data_root, 'masks'), suffix='.png'):
    #     img = cv2.imread(osp.join(data_root, 'masks', file), cv2.IMREAD_GRAYSCALE)
    #     seg_img = Image.fromarray(img).convert('P')
    #     seg_img.putpalette(np.array(palette, dtype=np.uint8))
    #     seg_img.save(osp.join(data_root, ann_dir, file))

    # split train/val set randomly

    mmcv.mkdir_or_exist(osp.join(data_root, split_dir))
    filename_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
        osp.join(data_root, ann_dir), suffix='.png')]
    with open(osp.join(data_root, split_dir, 'train.txt'), 'w') as f:
        # select first 4/5 as train set
        train_length = int(len(filename_list) * (1 - test_split))
        f.writelines(line + '\n' for line in filename_list[:train_length])
    with open(osp.join(data_root, split_dir, 'val.txt'), 'w') as f:
        # select last 1/5 as train set
        f.writelines(line + '\n' for line in filename_list[train_length:])

    @DATASETS.register_module()
    class RTT_defects(CustomDataset):
        CLASSES = classes
        PALETTE = PALETTE

        def __init__(self, split, **kwargs):
            super().__init__(img_suffix='.png', seg_map_suffix='.png', split=split, **kwargs)
            assert osp.exists(self.img_dir) and self.split is not None

    cfg = Config.fromfile('configs/deeplabv3plus/deeplabv3plus_r101-d8_512x512_40k_voc12aug.py')

    # Since we use ony one GPU, BN is used instead of SyncBN
    cfg.norm_cfg = dict(type='BN', requires_grad=True)
    cfg.model.backbone.norm_cfg = cfg.norm_cfg
    cfg.model.decode_head.norm_cfg = cfg.norm_cfg
    cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg

    # modify num classes of the model in decode/auxiliary head
    cfg.model.decode_head.num_classes = 5
    cfg.model.auxiliary_head.num_classes = 5

    # Modify dataset type and path
    cfg.dataset_type = 'RTT_defects'
    cfg.data_root = data_root

    cfg.data.samples_per_gpu = 10
    cfg.data.workers_per_gpu = 10

    cfg.img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    cfg.crop_size = (256, 256)
    cfg.train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        # dict(type='Resize', img_scale=(320, 240), ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **cfg.img_norm_cfg),
        dict(type='Pad', size=cfg.crop_size, pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ]

    cfg.test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(512, 512),
            # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            flip=False,
            transforms=[
                # dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **cfg.img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]

    cfg.data.train.type = cfg.dataset_type
    cfg.data.train.data_root = cfg.data_root
    cfg.data.train.img_dir = img_dir
    cfg.data.train.ann_dir = ann_dir
    cfg.data.train.pipeline = cfg.train_pipeline
    cfg.data.train.split = 'splits/train.txt'

    cfg.data.val.type = cfg.dataset_type
    cfg.data.val.data_root = cfg.data_root
    cfg.data.val.img_dir = img_dir
    cfg.data.val.ann_dir = ann_dir
    cfg.data.val.pipeline = cfg.test_pipeline
    cfg.data.val.split = 'splits/val.txt'

    cfg.data.test.type = cfg.dataset_type
    cfg.data.test.data_root = cfg.data_root
    cfg.data.test.img_dir = img_dir
    cfg.data.test.ann_dir = ann_dir
    cfg.data.test.pipeline = cfg.test_pipeline
    cfg.data.test.split = 'splits/val.txt'

    # We can still use the pre-trained Mask RCNN model though we do not need to
    # use the mask branch
    cfg.load_from = 'denred0_checkpoints/deeplabv3plus_r101-d8_512x512_40k_voc12aug_20200613_205333-faf03387.pth'
    # cfg.init_cfg = dict(type='Pretrained', checkpoint='denred0_checkpoints/deeplabv3plus_r101-d8_512x512_40k_voc12aug_20200613_205333-faf03387.pth')

    # Set up working dir to save files and logs.
    cfg.work_dir = './denred0_work_dirs/' + model_name

    cfg.runner.max_iters = 42000
    cfg.log_config.interval = 100
    cfg.evaluation.interval = 1000
    cfg.checkpoint_config.interval = 1000

    cfg.checkpoint_config.meta = dict(
        CLASSES=classes,
        PALETTE=PALETTE)

    # Set seed to facitate reproducing the result
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    # Let's have a look at the final config used for training
    # print(f'Config:\n{cfg.pretty_text}')

    # Build the dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_segmentor(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    model.PALETTE = datasets[0].PALETTE

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    return model, datasets, cfg

# train_segmentor(model, datasets, cfg, distributed=False, validate=True, meta=dict())

# img = mmcv.imread('data/images/20201221_49a15f3d-1761-4bc5-a838-f079766d5cca_035017_1_001_laser.png')
#
# model.cfg = cfg
# result = inference_segmentor(model, img)
# img_res = result[0].astype('uint8')
# seg_img = Image.fromarray(img_res).convert('P')
# seg_img.putpalette(np.array(palette, dtype=np.uint8))
# seg_img.save(osp.join(data_root, 'result', '1.png'))

# opencvImage = cv2.cvtColor(np.array(seg_img), cv2.COLOR_RGB2BGR)
#
#
# cv2.imwrite('data/result/1.png', opencvImage)
