# take npz and yolo annotation of the plate position, crop according to the yolo
# annotation, and save the cropped images as rgb and do pca on the cropped HSI array.
import os
import glob
import subprocess
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import pandas as pd


def parse_yolo_annotation(annotation_file: str):
    df = pd.read_csv(annotation_file, sep=" ", header=None)
    x_c, y_c, w, h = df.iloc[0, 1:].astype(float)
    w *= 0.9
    h *= 0.9
    x0 = x_c - w / 2
    y0 = y_c - h / 2
    x1 = x_c + w / 2
    y1 = y_c + h / 2
    return x0, y0, x1, y1


def process_file_with_pca(file_: str, output_dir: str, pca_output_dir: str):
    basedir = os.path.dirname(file_)
    parent_dir = os.path.dirname(basedir)
    basename = os.path.splitext(os.path.basename(file_))[0]
    yaml_file = os.path.join(basedir, f"{basename}.yaml")
    annotation_file = os.path.join(
        parent_dir, "hyperspectral_plate_detection", f"{basename}_rgb.txt"
    )

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
        "--pca",
        "-po",
        pca_output_dir,
        "-qp",
        "0.005",
    ]
    if os.path.isfile(annotation_file):
        command.extend(
            ["-c"] + list(map(str, parse_yolo_annotation(annotation_file)))
        )
    subprocess.run(command, check=True)


if __name__ == "__main__":
    data_dir = "/home/ubuntu/data/camii/202304_darpa_arcadia_soil"
    files = sorted(glob.glob(os.path.join(data_dir, "hyperspectral", "*.npz")))
    script_dir = os.path.expanduser("~/dev/CAMII_dev")
    output_dir = os.path.join(data_dir, "hyperspectral_rgb_cropped")
    pca_output_dir = os.path.join(data_dir, "hyperspectral_pca")

    Parallel(n_jobs=4)(
        delayed(process_file_with_pca)(file_, output_dir, pca_output_dir)
        for file_ in tqdm(files)
    )
