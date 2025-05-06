"""Preprocess hsi by iterating through train dataload and val dataloader and crop at the coordinate of each sample.

Crop to a fixed size, and following training will be capped by this size.

File name is <output_dir>/<plate_name>/<x_coord>_<y_coord>_<size>.npz
"""

import json
import os

import numpy as np
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from tampic.utils.affine_transform import get_query2target_func
from tampic.utils.hsi_breakdown import read_npz_file_and_crop
from tampic.utils.crop_image import crop_image

base_dir = "/home/ubuntu/data/camii"
hsi_dir = "hyperspectral"
hsi_plate_dec_dir = "hyperspectral_plate_detection"
metadata_path = "all_0531.json"
size = 366

with open(f"{base_dir}/{metadata_path}") as f:
    metadata = json.load(f)

dfs = []
for proj, meta_proj in metadata["data"].items():
    if meta_proj["status"] != "valid":
        continue
    with open(f"{base_dir}/{proj}/{meta_proj['amplicon_info']}") as f:
        amplicon_info = json.load(f)
    for plate, meta_plate in tqdm(meta_proj["plates"].items(), desc=proj):
        rows = []
        if meta_plate["status"] != "valid":
            continue
        transform_info = meta_plate["isolate_transform"]["to_rgb"]
        func_transform_iso2rgb = get_query2target_func(
            **transform_info["params"],
            **transform_info["stats"],
            flip=transform_info["flip"],
        )
        if "hsi" in meta_plate["images"]:
            time_points_hsi = {
                int(k[1:]): {
                    "dir": os.path.join(base_dir, proj, v["dir"]),
                    "transform": v["transform"]["from_rgb"],
                }
                for k, v in meta_plate["images"]["hsi"].items()
                if k != "default" and v["status"] == "valid"
            }
        else:
            continue
        for time_point_info in time_points_hsi.values():
            dir_ = time_point_info["dir"]
            os.makedirs(dir_, exist_ok=True)
            transform_params = time_point_info["transform"]
            hsi_npz_path = os.path.abspath(
                os.path.join(dir_, "..", "..", hsi_dir, f"{os.path.basename(dir_)}.npz")
            )
            hsi_plate_annot_path = os.path.abspath(
                os.path.join(
                    dir_,
                    "..",
                    "..",
                    hsi_plate_dec_dir,
                    f"{os.path.basename(dir_)}_rgb.txt",
                )
            )
            # hsi_npz = np.load(hsi_npz_path)["data"]
            hsi_npz = read_npz_file_and_crop(
                input_npz_path=hsi_npz_path, plate_annot=hsi_plate_annot_path
            )
            func_transform_rgb2hsi = get_query2target_func(
                **transform_params["params"],
                **transform_params["stats"],
                flip=transform_params["flip"],
            )

            def process_isolate(meta_isolate, hsi_npz, dir_, size):
                coords_iso = np.array([meta_isolate["coord"]])
                coords_hsi = func_transform_rgb2hsi(func_transform_iso2rgb(coords_iso))
                coords_hsi = coords_hsi[0]
                img_iso = crop_image(hsi_npz, center=coords_hsi, crop_size=size)
                output_path = os.path.join(
                    dir_, f"{coords_hsi[0]:.3f}_{coords_hsi[1]:.3f}_{size}.npz"
                )
                # save back as npz
                np.savez_compressed(output_path, data=img_iso)

            # Parallelize the process
            Parallel(n_jobs=16)(
                delayed(process_isolate)(meta_isolate, hsi_npz, dir_, size)
                for isolate, meta_isolate in tqdm(
                    amplicon_info[plate].items(), desc=plate, leave=False
                )
            )
