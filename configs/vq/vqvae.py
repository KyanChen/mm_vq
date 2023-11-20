# custom_imports = dict(imports=['mmseg.vq'], allow_failed_imports=False)
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, by_epoch=True, save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=20),
)

work_dir = './work_dirs/mapgpt/vqvae-commit-256'

vis_backends = [
    dict(type='LocalVisBackend'),
    # dict(type='WandbVisBackend', init_kwargs=dict(project='mapgpt', group='vqvae', name='vqvae-commit'))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False

crop_size = (256, 256)

# model settings
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size_divisor=32,
    pad_val=255,
    seg_pad_val=255
)

model = dict(
    type='VQVAEEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='VQVAEModel',
        num_classes=7,
        ignore_index=255,
        quantizer=dict(
            type='VectorQuantizer',
            n_e=64,
            # n_e=128,
            vq_embed_dim=768,
            legacy=False,
            beta=1.,
            with_codebook_reset=True,
            mu=0.99,
        ),
        down_block_types=["DownEncoderBlock2D"]*5,
        up_block_types=["UpDecoderBlock2D"]*5,
        block_out_channels=(32, 64, 128, 256, 512),
        layers_per_block=2,
        act_fn="silu",
        latent_channels=256,
        vq_embed_dim=768,
    ),
    decode_head=dict(
        type='PseudoDecodeHead',
        threshold=0.5,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        ignore_index=255,
        align_corners=False
    )
)


dataset_type = 'SemanticVQVAEDataset'
data_root = r'data_samples'

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=crop_size,
        ratio_range=(0.1, 2.0),
        keep_ratio=True,
        interpolation='nearest'
    ),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged'),
    dict(type='Resize', scale=crop_size, keep_ratio=True, interpolation='nearest'),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

batch_size_per_gpu = 4
num_workers = 8
persistent_workers = True
# indices = list(range(0, 8))
indices = None

train_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root+'/train',
        reduce_zero_label=True,
        indices=indices,
        img_suffix='.png',
        seg_map_suffix='.png',
        data_prefix=dict(
            img_path='', seg_map_path=''),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=batch_size_per_gpu*2,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root+'/val',
        reduce_zero_label=True,
        indices=indices,
        img_suffix='.png',
        seg_map_suffix='.png',
        data_prefix=dict(
            img_path='', seg_map_path=''),
        pipeline=test_pipeline,
        test_mode=True
    )
)

test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

base_lr = 1e-4
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.01),
)

max_epochs = 200
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=100),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=0,
        end=max_epochs,
        by_epoch=True,
    )
]


train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


