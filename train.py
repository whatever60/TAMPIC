from datetime import datetime
import math

import torch
import torch.nn.functional as F
import lightning as L
from torch.optim import AdamW
from schedulers import WarmupScheduler
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from ema import EMA
from models import TAMPICResNetLightningModule
from datasets import TAMPICDataModule

L.seed_everything(42)


base_dir = "/mnt/c/aws_data/data/camii"
base_dir = "/home/ubuntu/data/camii"

# hparams
lr = 1e-3
crop_size_final = 128
warmup_steps = 1000
max_epochs = 2_000
num_devices = 1
batch_size = 8
grad_accum = 4
num_batches_per_epoch = 50
check_val_every_n_epoch = 10

amplicon_type, taxon_level = "16s", "genus"

train_config = {
    "pretrained-no_empty-weight_density": {
        "weight_by_label": True,
        "weight_by_plate": False,
        "weight_by_density": True,
        "weight_density_kernel_size": 100,
        "p_num_igs": {1: 2, 2: 2, 3: 1},
        "p_igs": {"rgb-red": 2, "rgb-white": 3, "hsi": 1},
        "p_last_time_point": 0.6,
        "p_hsi_channels": 0.2,
        #
        "p_num_igs_val": {1: 0, 2: 0, 3: 1},
        "p_igs_val": {"rgb-red": 1, "rgb-white": 1, "hsi": 1},
        "p_last_time_point_val": 1,
        "p_hsi_channels_val": 1,
        #
        "keep_empty": False,
        "keep_others": True,
        "pretrained": True,
    },
    "pretrained-rgb_only": {  # hsi is always dropped
        "weight_by_label": True,
        "weight_by_plate": False,
        "weight_by_density": False,
        "weight_density_kernel_size": None,
        "p_num_igs": {1: 0.3, 2: 0.7, 3: 0},
        "p_igs": {"rgb-red": 1, "rgb-white": 2, "hsi": 0},
        "p_last_time_point": 0.6,
        "p_hsi_channels": 0.2,
        #
        "p_num_igs_val": {1: 0, 2: 1, 3: 0},
        "p_igs_val": {"rgb-red": 1, "rgb-white": 1, "hsi": 0},
        "p_last_time_point_val": 1,
        "p_hsi_channels_val": 0,
        #
        "keep_empty": True,
        "keep_others": True,
        "pretrained": True,
    },
    "pretrained-rgb_only-weight_density": {
        "weight_by_label": True,
        "weight_by_plate": False,
        "weight_by_density": True,
        "weight_density_kernel_size": 100,
        "p_num_igs": {1: 0.3, 2: 0.7, 3: 0},
        "p_igs": {"rgb-red": 1, "rgb-white": 2, "hsi": 0},
        "p_last_time_point": 0.6,
        "p_hsi_channels": 0.2,
        #
        "p_num_igs_val": {1: 0, 2: 1, 3: 0},  # for val always provide everything
        "p_igs_val": {"rgb-red": 1, "rgb-white": 1, "hsi": 0},
        "p_last_time_point_val": 1,
        "p_hsi_channels_val": 0,
        #
        "keep_empty": True,
        "keep_others": True,
        "pretrained": True,
    },
    "pretrained-rgb_only-no_empty-weight_density": {  # hsi is always dropped
        "weight_by_label": True,
        "weight_by_plate": False,
        "weight_by_density": True,
        "weight_density_kernel_size": 100,
        "p_num_igs": {1: 0.3, 2: 0.7, 3: 0},
        "p_igs": {"rgb-red": 1, "rgb-white": 2, "hsi": 0},
        "p_last_time_point": 0.6,
        "p_hsi_channels": 0.2,
        #
        "p_num_igs_val": {1: 0, 2: 1, 3: 0},  # for val always provide everything
        "p_igs_val": {"rgb-red": 1, "rgb-white": 1, "hsi": 0},
        "p_last_time_point_val": 1,
        "p_hsi_channels_val": 0,
        #
        "keep_empty": False,
        "keep_others": True,
        "pretrained": True,
    },
    "pretrained-hsi_only-no_empty": {
        "weight_by_label": True,
        "weight_by_plate": False,
        "weight_by_density": False,
        "weight_density_kernel_size": None,
        #
        "p_num_igs": {1: 1, 2: 0, 3: 0},
        "p_igs": {"rgb-red": 0, "rgb-white": 0, "hsi": 1},
        "p_last_time_point": 0.6,
        "p_hsi_channels": 0.2,
        #
        "p_num_igs_val": {1: 1, 2: 0, 3: 0},
        "p_igs_val": {"rgb-red": 0, "rgb-white": 0, "hsi": 1},
        "p_last_time_point_val": 1,
        "p_hsi_channels_val": 1,
        #
        "keep_empty": False,
        "keep_others": True,
        "pretrained": True,
    },
    "pretrained-hsi_only-no_empty-pretrained_hsi_base": {
        "weight_by_label": True,
        "weight_by_plate": False,
        "weight_by_density": False,
        "weight_density_kernel_size": None,
        #
        "p_num_igs": {1: 1, 2: 0, 3: 0},
        "p_igs": {"rgb-red": 0, "rgb-white": 0, "hsi": 1},
        "p_last_time_point": 0.6,
        "p_hsi_channels": 0.2,
        #
        "p_num_igs_val": {1: 1, 2: 0, 3: 0},
        "p_igs_val": {"rgb-red": 0, "rgb-white": 0, "hsi": 1},
        "p_last_time_point_val": 1,
        "p_hsi_channels_val": 1,
        #
        "keep_empty": False,
        "keep_others": True,
        "pretrained": True,
        "_pretrained_hsi_base": True,
        "_norm_and_sum": True,
    },
}
mode = "pretrained-hsi_only-no_empty-pretrained_hsi_base"

# Fit the model
dm = TAMPICDataModule(
    metadata_train_path=f"{base_dir}/all_0531.json",
    weight_by_label=train_config[mode]["weight_by_label"],
    weight_by_density=train_config[mode]["weight_by_density"],
    weight_density_kernel_size=train_config[mode]["weight_density_kernel_size"],
    weight_by_plate=train_config[mode]["weight_by_plate"],
    #
    p_num_igs=train_config[mode]["p_num_igs"],
    p_igs=train_config[mode]["p_igs"],
    p_last_time_point=train_config[mode]["p_last_time_point"],
    p_hsi_channels=train_config[mode]["p_hsi_channels"],
    #
    p_num_igs_val=train_config[mode]["p_num_igs_val"],
    p_igs_val=train_config[mode]["p_igs_val"],
    p_last_time_point_val=train_config[mode]["p_last_time_point_val"],
    p_hsi_channels_val=train_config[mode]["p_hsi_channels_val"],
    #
    amplicon_type=amplicon_type,
    taxon_level=taxon_level,
    min_counts=10,
    min_counts_dominate=0,
    min_ratio=2,
    min_num_isolates=30,
    keep_empty=train_config[mode]["keep_empty"],
    keep_others=train_config[mode]["keep_others"],
    crop_size_init={"rgb-red": 224, "rgb-white": 224, "hsi": 128},
    crop_size_final=crop_size_final,
    target_mask_kernel_size=5,
    num_devices=num_devices,
    batch_size=batch_size,
    num_workers=16,
    num_batches_per_epoch=num_batches_per_epoch,
    # _hsi_group_k=3,
    _hsi_crop_size=196,
)
dm.setup()
# Create the model
model = TAMPICResNetLightningModule(
    model_type="resnet18",
    num_classes=len(dm.label_clean2idx),
    lr=lr,
    pretrained=train_config[mode]["pretrained"],
    warmup_steps=warmup_steps,
    total_steps=math.ceil(max_epochs * dm._num_batches_per_epoch / grad_accum),
    batch_size=batch_size,
    wavelengths=dm.hsi_wavelengths,
    _pretrained_hsi_base=train_config[mode].get("_pretrained_hsi_base", False),
    _norm_and_sum=train_config[mode].get("_norm_and_sum", False),
)
today = datetime.today()
logger = TensorBoardLogger(
    "tb_logs",
    name=f"{today.strftime('%Y%m%d')}_TAMPIC_{amplicon_type}_{taxon_level}_{mode}",
)
lr_monitor = LearningRateMonitor(logging_interval="step")
ckpt_callback = ModelCheckpoint(
    monitor="val_easy_acc/dataloader_idx_0",
    filename="best_model",
    save_top_k=3,
    save_last=1,
    mode="max",
    every_n_epochs=check_val_every_n_epoch * 5,
)
ema_callback = EMA(decay=0.999)
model_summary_callback = ModelSummary(max_depth=3)
# Initialize the Trainer
trainer = L.Trainer(
    logger=logger,
    callbacks=[lr_monitor, ckpt_callback, ema_callback, model_summary_callback],
    max_epochs=max_epochs,
    accumulate_grad_batches=grad_accum,
    reload_dataloaders_every_n_epochs=1,
    check_val_every_n_epoch=check_val_every_n_epoch,
    # gpus=0,  # if using GPU
)
trainer.fit(
    model,
    datamodule=dm,
    # ckpt_path="/home/ubuntu/dev/TAMPIC/tb_logs/20240610_TAMPIC_16s_genus_pretrained-no_empty-weight_density/version_1/checkpoints/epoch=1259-step=16380.ckpt",
)
