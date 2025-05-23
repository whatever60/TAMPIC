from datetime import datetime
import math
import argparse
import os
import json

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)

from tampic.ema import EMA
from tampic.models.models import TAMPICResNetLightningModule
from tampic.datasets import TAMPICDataModule


L.seed_everything(42)

base_dir = "/mnt/c/aws_data/data/camii"
base_dir = "/home/ubuntu/data/camii"

# hparams
lr = 2e-3
max_epochs = 30
num_batches_per_epoch = 1000
num_devices = 1
batch_size = 64  # per device batch size before gradient accumulation
grad_accum = 2
num_samples_per_epoch = num_batches_per_epoch * batch_size * num_devices
total_steps = math.ceil(max_epochs * num_batches_per_epoch / grad_accum)
warmup_steps = 2000
check_val_every_n_epoch = 1
checkpoint_every_n_epoch = 5
# amplicon_type, taxon_level = "16s", "genus"

# the ratio of rgb crop size and hsi crop size should be roughly 1296 : 926 = 7 : 5 = 1.4
crop_size_init = {
    "224_old": {"rgb-red": 224, "rgb-white": 224, "hsi": 128},
    "192": {"rgb-red": 192, "rgb-white": 192, "hsi": 128},
    "128": {"rgb-red": 128, "rgb-white": 128, "hsi": 96},
}
crop_size_final = 128

# Function to load config
def load_config(config_arg) -> dict:
    pre_config_path = os.path.join(os.path.dirname(__file__), "pre_configs.json")

    # Load predefined configurations
    with open(pre_config_path, "r") as f:
        predefined_configs = json.load(f)

    # Check if the argument is a JSON file
    if os.path.isfile(config_arg) and config_arg.endswith(".json"):
        with open(config_arg, "r") as f:
            config = json.load(f)
        print(f"Loaded configuration from {config_arg}")
    elif config_arg in predefined_configs:
        print(f"Using predefined configuration: {config_arg}")
        config = predefined_configs[config_arg]
    else:
        raise ValueError(
            f"Configuration '{config_arg}' not found. Must be a JSON file or a predefined config."
        )
    # modify a few arguments to use int as keys since JSON does not support int as keys
    config["p_num_igs"] = {int(k): v for k, v in config["p_num_igs"].items()}
    config["p_num_igs_val"] = {int(k): v for k, v in config["p_num_igs_val"].items()}
    return config


parser = argparse.ArgumentParser(description="Train a model")
parser.add_argument("--data", type=str, default="all_0531.json")
parser.add_argument(
    "--amplicon_type",
    type=str,
    default="16s",
    choices=["16s", "its"],
)
parser.add_argument(
    "--taxon_level",
    type=str,
    default="genus",
    choices=["domain", "phylum", "class", "order", "family", "genus", "species"],
)
parser.add_argument(
    "--config",
    type=str,
    help="Path to the configuration file or a predefined config name",
)
parser.add_argument("--name", type=str, default="unnamed", help="Name of the run")
args = parser.parse_args()

train_config_mode = load_config(args.config)

# mode = "pretrained-no_empty-weight_density-192"
# print(f"\n======== Training model under {mode} mode. ========\n")

# Fit the model
dm = TAMPICDataModule(
    metadata_train_path=f"{base_dir}/{args.data}",
    # metadata_train_path=f"{base_dir}/all_20250416.json",
    weight_by_label=train_config_mode["weight_by_label"],
    weight_by_density=train_config_mode["weight_by_density"],
    weight_density_kernel_size=train_config_mode["weight_density_kernel_size"],
    weight_by_plate=train_config_mode["weight_by_plate"],
    #
    p_num_igs=train_config_mode["p_num_igs"],
    p_igs=train_config_mode["p_igs"],
    p_last_time_point=train_config_mode["p_last_time_point"],
    p_hsi_channels=train_config_mode["p_hsi_channels"],
    #
    p_num_igs_val=train_config_mode["p_num_igs_val"],
    p_igs_val=train_config_mode["p_igs_val"],
    p_last_time_point_val=train_config_mode["p_last_time_point_val"],
    p_hsi_channels_val=train_config_mode["p_hsi_channels_val"],
    #
    amplicon_type=args.amplicon_type,
    taxon_level=args.taxon_level,
    min_counts=10,
    min_counts_dominate=0,
    min_ratio=2,
    min_num_isolates=42,
    keep_empty=train_config_mode["keep_empty"],
    keep_others=train_config_mode["keep_others"],
    crop_size_init=train_config_mode.get("crop_size_init", crop_size_init["224_old"]),
    crop_size_final=train_config_mode.get("crop_size_final", crop_size_final),
    target_mask_kernel_size=5,
    num_devices=num_devices,
    batch_size=batch_size,
    num_batches_per_epoch=num_batches_per_epoch,
    num_workers=16,  # this affects memory consumption linearly
    prefetch_factor=4,
    # _hsi_group_k=3,
    _hsi_crop_size=196,
    _hsi_norm=True,
    # _hsi_avg_dim=train_config_mode.get("_hsi_avg_dim", None),
    _couple_rgb=True,
    _hsi_wavelengths_overwrite=train_config_mode.get("_hsi_wavelengths_overwrite"),
)
dm.setup()

# setup callbacks
today = datetime.today()
logger = TensorBoardLogger(
    "tb_logs",
    # name=f"{today.strftime('%Y%m%d')}_TAMPIC_{os.path.splitext(args.data)[0]}_"
    # f"{amplicon_type}_{taxon_level}_{args.name}",
    name="-".join(
        [
            today.strftime("%Y%m%d"),
            "TAMPIC",
            os.path.splitext(args.data)[0],
            args.amplicon_type,
            args.taxon_level,
            args.name,
        ]
    ),
)
lr_monitor = LearningRateMonitor(logging_interval="step")
ckpt_callback = ModelCheckpoint(
    # monitor="val_easy_acc/dataloader_idx_0",
    monitor="val-easy_top1_acc",
    filename="{epoch}-{step}-{val-easy_top1_acc:.2f}-{val-easy_top3_acc:.2f}",
    mode="max",
    save_top_k=-1,
    save_last=True,
    every_n_epochs=checkpoint_every_n_epoch,
)
ema_callback = EMA(decay=0.999)
model_summary_callback = ModelSummary(max_depth=2)

# Create the model
model = TAMPICResNetLightningModule(
    model_type="resnet18",
    # num_classes=len(dm.label_clean2idx),
    lr=lr,
    pretrained=train_config_mode["pretrained"],
    warmup_steps=warmup_steps,
    total_steps=total_steps,
    batch_size=batch_size,
    hsi_avg_dim=train_config_mode.get("hsi_avg_dim", None),
    wavelengths=dm.hsi_wavelengths,
    _pretrained_hsi_base=train_config_mode.get("_pretrained_hsi_base", False),
    _norm_and_sum=train_config_mode.get("_norm_and_sum", False),
    _prediction_log_dir=os.path.join(logger.log_dir, "predictions"),
    _multi_level_aux_loss_weight=train_config_mode.get(
        "_multi_level_aux_loss_weight", 0.0
    ),
    _multi_level_label_clean2idx=dm.label_clean2idx_all,
    label_clean2idx=dm.label_clean2idx,
)

# Initialize the Trainer
trainer = L.Trainer(
    logger=logger,
    callbacks=[lr_monitor, ckpt_callback, ema_callback, model_summary_callback],
    max_epochs=max_epochs,
    accumulate_grad_batches=grad_accum,
    reload_dataloaders_every_n_epochs=1,
    check_val_every_n_epoch=check_val_every_n_epoch,
    num_sanity_val_steps=0,
    devices=num_devices,  # if using GPU
)
trainer.fit(
    model,
    datamodule=dm,
    # ckpt_path="/home/ubuntu/dev/tampic/tb_logs/20250425_TAMPIC_16s_genus_pretrained-large_data-rgb_only-no_empty-no_others-weight_density-128/version_1/checkpoints/last.ckpt",
)
