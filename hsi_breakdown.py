import sys
import os
import yaml

import numpy as np
import imageio
import cv2 as cv
from joblib import Parallel, delayed
from tqdm.auto import tqdm, trange

from hsi_crop import parse_yolo_annotation

sys.path.append("/home/ubuntu/dev/CAMII_dev")

from data_transform import _crop_array


def save_grayscale_pngs_from_npz(
    input_npz_path: str,
    output_dir: str,
    plate_annot: str = None,
    k: int = 0,
) -> None:
    """
    Save 16-bit grayscale PNG files from a 3D tensor stored in a .npz file.

    Args:
        input_npz_path (str): Path to the input .npz file.
        output_dir (str): Directory where the output PNG files will be saved.
    """
    # Load the .npz file
    data = np.load(input_npz_path)
    tensor = data["data"]  # Assuming the data is stored under the key "data"

    if plate_annot is not None:
        x0, y0, x1, y1 = parse_yolo_annotation(plate_annot)
        tensor = _crop_array(tensor, [x0, y0, x1, y1])

    # Ensure tensor is in uint16 format
    tensor = tensor.astype(np.uint16)

    # Load the corresponding YAML file
    input_npz_base = os.path.splitext(os.path.basename(input_npz_path))[0]
    yaml_path = f"{os.path.splitext(input_npz_path)[0]}.yaml"
    with open(yaml_path, "r") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)

    wavelengths = yaml_data["wavelength"]

    # Ensure output directory exists
    specific_output_dir = os.path.join(output_dir, input_npz_base)
    if not os.path.exists(specific_output_dir):
        os.makedirs(specific_output_dir)

    # Save each channel as a 16-bit grayscale PNG
    for i in trange(0, len(wavelengths), max(1, k)):
        if k > 0:
            output_path = os.path.join(
                    specific_output_dir, f"{'_'.join(map(str, wavelengths[i:i+k]))}.tif"
                )
            cv.imwrite(output_path, tensor[:, :, i : i + k])
        else:
            output_path = os.path.join(specific_output_dir, f"{wavelengths[i]}.png")
            imageio.imwrite(output_path, tensor[:, :, i], format="png")


def process_directory(
    input_npz_dir: str,
    output_dir: str,
    plate_annot_dir: str = None,
    k: int = 0,
    n_jobs: int = 16,
) -> None:
    """
    Process each .npz file in the given directory and save 16-bit grayscale PNG files.

    Args:
        input_npz_dir (str): Directory containing .npz files.
        output_dir (str): Directory where the output PNG files will be saved.
        n_jobs (int): Number of jobs to run in parallel (default: 4).
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find all .npz files in the directory
    npz_files = [
        os.path.join(input_npz_dir, file_name)
        for file_name in os.listdir(input_npz_dir)
        if file_name.endswith(".npz")
    ]
    if plate_annot_dir is not None:
        plate_annot_files = [
            os.path.join(
                plate_annot_dir,
                os.path.splitext(os.path.basename(file_name))[0] + "_rgb.txt",
            )
            for file_name in npz_files
        ]
        if not all(os.path.isfile(f) for f in plate_annot_files):
            raise FileNotFoundError(
                "All plate annotation files must be present in the plate_annot_dir."
            )
    else:
        plate_annot_files = [None] * len(npz_files)

    # Process each .npz file in parallel with a progress bar
    Parallel(n_jobs=n_jobs)(
        delayed(save_grayscale_pngs_from_npz)(
            input_npz_path,
            output_dir,
            plate_annot=plate_annot_file,
            k=k,
        )
        for input_npz_path, plate_annot_file in zip(npz_files, plate_annot_files)
    )


# Example usage:
# save_grayscale_pngs_from_npz("path/to/input/file.npz", "path/to/output/directory")
# process_directory("path/to/input/directory", "path/to/output/directory")
