# take npz and yolo annotation of the plate position, crop according to the yolo
# annotation, and save the cropped images as rgb and do pca on the cropped HSI array.

import sys
import os
import glob
import subprocess

from joblib import Parallel, delayed
from tqdm.auto import tqdm


def process_file(file_: str, output_dir: str):
    yaml_file = file_.replace(".npz", ".yaml")
    # print this location of python executable
    command = [
        f"{script_dir}/data_transform.py",
        "npy2png",
        "-i",
        file_,
        "-m",
        yaml_file,
        "-o",
        output_dir,
        "-qr",
        "0.999",
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    data_dir = "/home/ubuntu/data/camii/202401_darpa_arcadia_arl_boneyard"
    files = glob.glob(os.path.join(data_dir, "hyperspectral", "*.npz"))
    script_dir = os.path.expanduser("~/dev/CAMII_dev")
    Parallel(n_jobs=16)(
        delayed(process_file)(
            file_, output_dir=os.path.join(data_dir, "hyperspectral_rgb")
        )
        for file_ in tqdm(files)
    )
