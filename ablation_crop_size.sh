#!/bin/bash

# Path to the directory where the config files are located
config_dir="ablation_configs/crop_size"

# List of RGB crop sizes
# declare -a rgb_sizes=("64" "96" "128" "192" "256" "384" "512")
declare -a rgb_sizes=("192" "256" "384" "512")

# Loop through each config and execute train.py
for size in "${rgb_sizes[@]}"
do
    config_file="${config_dir}/${size}.json"
    echo "Running training with config ${config_file}..."
    python train.py --config ${config_file} --name "ablation_crop-size_${size}"
    break
done
