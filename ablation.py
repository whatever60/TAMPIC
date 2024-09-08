import subprocess
import json
import os
import argparse
from itertools import product


def modify_and_save_config(crop_size_init, hsi_avg_dim, output_dir):
    # Modify the base configuration
    config = base_config.copy()
    config["crop_size_init"] = crop_size_init
    config["_hsi_avg_dim"] = hsi_avg_dim

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the modified config as a temporary JSON file
    filename = f'config_crop_{crop_size_init["hsi"]}_hsi_avg_{hsi_avg_dim}.json'
    temp_file = os.path.join(output_dir, filename)

    with open(temp_file, "w") as outfile:
        json.dump(config, outfile, indent=4)

    return temp_file


# Hard-coded base JSON configuration
base_config = {
    "weight_by_label": True,
    "weight_by_plate": False,
    "weight_by_density": True,
    "weight_density_kernel_size": 100,
    "p_num_igs": {"1": 2, "2": 2, "3": 1},
    "p_igs": {"rgb-red": 2, "rgb-white": 3, "hsi": 1},
    "p_last_time_point": 0.6,
    "p_hsi_channels": 0.2,
    "p_num_igs_val": {"1": 0, "2": 0, "3": 1},
    "p_igs_val": {"rgb-red": 1, "rgb-white": 1, "hsi": 1},
    "p_last_time_point_val": 1,
    "p_hsi_channels_val": 1,
    "crop_size_init": {"rgb-red": 64, "rgb-white": 64, "hsi": 46},
    "crop_size_final": 128,  # Fixed to 128 as requested
    "keep_empty": False,
    "keep_others": True,
    "pretrained": True,
    "_pretrained_hsi_base": True,
    "_norm_and_sum": True,
}

# Possible values for crop_size_init and _hsi_avg_dim
crop_size_init_values = [
    {"rgb-red": 64, "rgb-white": 64, "hsi": 46},
    {"rgb-red": 96, "rgb-white": 96, "hsi": 69},
    {"rgb-red": 128, "rgb-white": 128, "hsi": 91},
    {"rgb-red": 192, "rgb-white": 192, "hsi": 137},
    {"rgb-red": 384, "rgb-white": 384, "hsi": 274},
    {"rgb-red": 512, "rgb-white": 512, "hsi": 366},
]

hsi_avg_dim_values = [100, 200, 300, 400, 462]

output_dir = "ablation_configs/default"
# Generate all combinations of crop_size_init and _hsi_avg_dim
for crop_size_init, hsi_avg_dim in product(
    crop_size_init_values, hsi_avg_dim_values
):
    temp_json = modify_and_save_config(crop_size_init, hsi_avg_dim, output_dir)
    args = [
        "python",
        "train.py",
        "--config",
        temp_json,
        "--name",
        f"ablation_crop-size_{crop_size_init['rgb-red']}_hsi-avg-dim_{hsi_avg_dim}",
    ]
    print("Running command:", " ".join(args))
    subprocess.run(args)