import json
import os
import warnings
from typing import Optional
from functools import lru_cache

import numpy as np
from scipy.spatial import KDTree
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import lightning as L
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 as cv
from PIL import Image
from tqdm.auto import tqdm
from rich import print as rprint
from tabulate import tabulate
from joblib import Parallel, delayed

from ..utils.affine_transform import get_query2target_func
from ..utils.crop_image import crop_image


class _HSINormalize(A.Normalize):
    def __init__(self, mean: list[float], std: list[float]):
        super().__init__(mean, std)
        # self._to_tensor = T.ToTensor()

    def __call__(self, **kwargs):
        img = kwargs["image"]
        # if not isinstance(img, np.ndarray) and img.dtype == np.uint16:
        #     raise ValueError(
        #         f"Image must be a numpy array of type uint16, getting {type(img)}"
        #     )

        dropout = kwargs["hsi_channel_dropout"]
        mean = np.array(self.mean)
        std = np.array(self.std)
        img = (
            torch.from_numpy((img - mean[dropout]) / std[dropout])
            .float()
            .permute(2, 0, 1)
        )
        return {"image": img}


def get_geom_transforms(
    crop_size_rgb_white: int,
    crop_size_rgb_red: int,
    crop_size_hsi: int,
    couple: bool = False,
) -> dict:
    if not couple:
        geom_transforms = {
            "rgb-red": A.Compose(
                [
                    A.RandomResizedCrop(
                        size=(crop_size_rgb_red, crop_size_rgb_red), scale=(0.4, 1)
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=20, p=0.5),
                ],
                additional_targets={"target_mask": "image"},
            ),
            "rgb-white": A.Compose(
                [
                    A.RandomResizedCrop(
                        size=(crop_size_rgb_white, crop_size_rgb_white), scale=(0.4, 1)
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=20, p=0.5),
                ],
                additional_targets={"target_mask": "image"},
            ),
            "hsi": A.Compose(
                [
                    A.RandomResizedCrop(
                        size=(crop_size_hsi, crop_size_hsi), scale=(0.4, 1)
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=20, p=0.5),
                ],
                additional_targets={"target_mask": "image"},
            ),
        }
    else:
        assert crop_size_rgb_white == crop_size_rgb_red, "Crop size must be the same."
        geom_transforms = {
            "rgb": A.Compose(
                [
                    A.RandomResizedCrop(
                        size=(crop_size_rgb_white, crop_size_rgb_white), scale=(0.4, 1)
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=20, p=0.5),
                ],
                additional_targets={
                    "target_mask": "image",
                    "rgb_2": "image",
                    "target_mask_2": "image",
                },
            ),
            "hsi": A.Compose(
                [
                    A.RandomResizedCrop(
                        size=(crop_size_hsi, crop_size_hsi), scale=(0.4, 1)
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=20, p=0.5),
                ],
                additional_targets={"target_mask": "image"},
            ),
        }

    return geom_transforms


class GroupTransform:
    def __init__(
        self,
        channel_stats: dict[str, dict[str, float]],
        crop_size: int | dict[str, int],
        split: str = "train",
        _calc_stats_mode: bool = False,
        _no_norm: bool = False,
        _couple_rgb: bool = False,
    ):
        self._couple_rgb = _couple_rgb

        if isinstance(crop_size, int):
            crop_size = {"rgb-red": crop_size, "rgb-white": crop_size, "hsi": crop_size}
        self.crop_size = crop_size

        def _norm_for_no_norm(**kwargs):
            kwargs["image"] = torch.from_numpy(kwargs["image"].transpose(2, 0, 1))
            return kwargs

        if not _calc_stats_mode:
            # Geometric transformations applied per modality
            self.geom_transforms = get_geom_transforms(
                crop_size_rgb_white=crop_size["rgb-white"],
                crop_size_rgb_red=crop_size["rgb-red"],
                crop_size_hsi=crop_size["hsi"],
                couple=_couple_rgb,
            )

            # Color transformations are specific to each modality
            if split == "train":
                # ignore user warnings from pydantic in color jitter initialization
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.color_transforms = {
                        "rgb-red": A.Compose(
                            [
                                A.ColorJitter(
                                    brightness=0.4,
                                    contrast=0.4,
                                    saturation=0.4,
                                    hue=0.1,
                                    p=0.8,
                                ),
                                A.ToGray(p=0.1),
                            ]
                        ),
                        "rgb-white": A.Compose(
                            [
                                A.ColorJitter(
                                    brightness=0.4,
                                    contrast=0.4,
                                    saturation=0.4,
                                    hue=0.1,
                                    p=0.8,
                                ),
                                A.ToGray(p=0.1),
                            ]
                        ),
                        "hsi": A.Compose(
                            [
                                A.RandomBrightnessContrast(p=0.5),
                                A.OneOf(
                                    [
                                        A.GaussNoise(var_limit=(0.01, 0.05), p=0.5),
                                        A.GaussianBlur(
                                            blur_limit=(3, 5),
                                            p=0.5,
                                            sigma_limit=(0.2, 0.5),
                                        ),
                                    ],
                                    p=0.5,
                                ),
                            ]
                        ),
                    }
            else:
                self.color_transforms = {
                    "rgb-red": lambda **kwargs: kwargs,
                    "rgb-white": lambda **kwargs: kwargs,
                    "hsi": lambda **kwargs: kwargs,
                }
            self.norm = {
                "rgb-red": A.Compose(
                    [
                        A.Normalize(
                            mean=channel_stats["rgb-red"]["mean"],
                            std=channel_stats["rgb-red"]["std"],
                        ),
                        ToTensorV2(),
                    ]
                ),
                "rgb-white": A.Compose(
                    [
                        A.Normalize(
                            mean=channel_stats["rgb-white"]["mean"],
                            std=channel_stats["rgb-white"]["std"],
                        ),
                        ToTensorV2(),
                    ]
                ),
                "hsi": _HSINormalize(
                    mean=channel_stats["hsi"]["mean"],
                    std=channel_stats["hsi"]["std"],
                ),
            }

        else:  # no augmentation
            self.geom_transforms = {
                "rgb-red": lambda **kwargs: kwargs,
                "rgb-white": lambda **kwargs: kwargs,
                "hsi": lambda **kwargs: kwargs,
            }
            self.color_transforms = {
                "rgb-red": lambda **kwargs: kwargs,
                "rgb-white": lambda **kwargs: kwargs,
                "hsi": lambda **kwargs: kwargs,
            }
            self.norm = {
                "rgb-red": A.Compose(
                    [
                        A.Normalize(
                            mean=np.zeros_like(
                                channel_stats["rgb-red"]["mean"]
                            ).tolist(),
                            std=np.ones_like(channel_stats["rgb-red"]["std"]).tolist(),
                        ),
                        ToTensorV2(),
                    ]
                ),
                "rgb-white": A.Compose(
                    [
                        A.Normalize(
                            mean=np.zeros_like(
                                channel_stats["rgb-white"]["mean"]
                            ).tolist(),
                            std=np.ones_like(
                                channel_stats["rgb-white"]["std"]
                            ).tolist(),
                        ),
                        ToTensorV2(),
                    ]
                ),
                "hsi": _HSINormalize(
                    mean=np.zeros_like(channel_stats["hsi"]["mean"]).tolist(),
                    std=np.ones_like(channel_stats["hsi"]["std"]).tolist(),
                ),
            }
        if _no_norm:
            self.norm = {
                "rgb-red": _norm_for_no_norm,
                "rgb-white": _norm_for_no_norm,
                "hsi": _norm_for_no_norm,
            }

    def __call__(
        self,
        # images: dict[str, np.ndarray],
        # target_masks: dict[str, np.ndarray],
        data: dict[str, np.ndarray],
    ) -> dict[str, dict]:
        """
        Apply geometric and color transformations to the images.

        Args:
            images (dict[str, np.ndarray]): Dictionary of images for each modality group.
            target_masks (dict[str, np.ndarray]): Dictionary of target masks (Gaussian images) for each modality group.

        Returns:
            dict[str, np.ndarray]: Transformed images for each modality group.
        """
        # transformed_images = {}
        # transformed_target_masks = {}
        data_aug = {}

        data_augmented = (
            self._geom_transforms_call_coupled(data)
            if self._couple_rgb
            else self._geom_transforms_call(data)
        )
        for modality, data_m_augmented in data_augmented.items():
            try:
                image_transformed = self.color_transforms[modality](
                    image=data_m_augmented["image"]
                )["image"]
            except cv.error as e:
                rprint(
                    modality,
                    data_m_augmented["image"].shape,
                    data_m_augmented["image"].dtype,
                )
                raise e

            # Apply color transformation to the image only
            if modality == "hsi":
                hsi_channel_dropout = data[modality]["hsi_channel_dropout"]
                image_transformed = self.norm[modality](
                    image=image_transformed,
                    hsi_channel_dropout=hsi_channel_dropout,
                )["image"]
                image_full = torch.zeros(
                    (len(hsi_channel_dropout), *image_transformed.shape[1:]),
                    dtype=image_transformed.dtype,
                    device=image_transformed.device,
                )
                image_full[hsi_channel_dropout] += image_transformed
                data_aug[modality] = {
                    "image": image_full,
                    "target_mask": torch.from_numpy(
                        data_m_augmented["target_mask"]
                    ).float(),
                    "hsi_channel_dropout": torch.from_numpy(hsi_channel_dropout),
                }
            else:
                image_transformed = self.norm[modality](image=image_transformed)[
                    "image"
                ]
                data_aug[modality] = {
                    "image": image_transformed,
                    "target_mask": torch.from_numpy(
                        data_m_augmented["target_mask"]
                    ).float(),
                }

        return data_aug

    def _geom_transforms_call_coupled(
        self, data: dict[str, np.ndarray]
    ) -> dict[str, dict[str, np.ndarray]]:
        """When rgb modalities are coupled:
        If both rgb modalities are given in data, pass them to the same transform.
        If only one rgb modality is give, pass it to the transform.
        """
        num_rgb_modalities = sum(
            1 for modality in ["rgb-red", "rgb-white"] if modality in data
        )
        if num_rgb_modalities == 2:
            augmented_rgb = self.geom_transforms["rgb"](
                image=data["rgb-red"]["image"],
                target_mask=data["rgb-red"]["target_mask"],
                rgb_2=data["rgb-white"]["image"],
                target_mask_2=data["rgb-white"]["target_mask"],
            )
            ret = {
                "rgb-red": {
                    "image": augmented_rgb["image"],
                    "target_mask": augmented_rgb["target_mask"],
                },
                "rgb-white": {
                    "image": augmented_rgb["rgb_2"],
                    "target_mask": augmented_rgb["target_mask_2"],
                },
            }
            if "hsi" in data:
                augmented_hsi = self.geom_transforms["hsi"](
                    image=data["hsi"]["image"], target_mask=data["hsi"]["target_mask"]
                )
                ret["hsi"] = {
                    "image": augmented_hsi["image"],
                    "target_mask": augmented_hsi["target_mask"],
                }
        else:
            ret = {}
            for modality, data_m in data.items():
                img = data_m["image"]
                target_mask = data_m["target_mask"]
                ret[modality] = self.geom_transforms[modality.split("-")[0]](
                    image=img, target_mask=target_mask
                )
        return ret

    def _geom_transforms_call(
        self, data: dict[str, np.ndarray]
    ) -> dict[str, dict[str, np.ndarray]]:
        """When rgb modalities are not coupled:
        Pass each modality to the corresponding transform.
        """
        ret = {}
        for modality, data_m in data.items():
            img = data_m["image"]
            target_mask = data_m["target_mask"]
            ret[modality] = self.geom_transforms[modality](
                image=img, target_mask=target_mask
            )
        return ret


# def adaptive_avg_pool(data: np.ndarray, output_size: int) -> np.ndarray:
#     """Take average on the second dimension (channel) of data utilizing pytorch adaptive pooling.
#     """
#     ndim = data.ndim
#     b, c, *s = data.shape
#     data = torch.from_numpy(data)
#     data = data.permute(0, *list(range(2, ndim)), 1).view(b, int(np.prod(s)), c)
#     data = F.adaptive_avg_pool1d(data, output_size)
#     data = data.view(b, *s, output_size).permute(0, ndim - 1, *list(range(1, ndim - 1)))
#     return data.numpy()


def adaptive_avg_pool(data: np.ndarray, output_size: int) -> np.ndarray:
    """Take average on the last dimension of data utilizing pytorch adaptive pooling."""
    dtype = data.dtype
    *s, c = data.shape
    # data = torch.from_numpy(data).float()
    # data = data.view(int(np.prod(s)), c)
    # data = F.adaptive_avg_pool1d(data, output_size)
    # data = data.view(*s, output_size)
    # return data.numpy().astype(dtype)
    return (
        F.adaptive_avg_pool1d(
            torch.from_numpy(data).float().view(int(np.prod(s)), c), output_size
        )
        .view(*s, output_size)
        .numpy()
        .astype(dtype)
    )


@lru_cache(maxsize=None)  # Cache indefinitely based on hsi_path
def get_kdtree(hsi_path: str) -> tuple[KDTree, list[str], list[tuple[float, float]]]:
    """Build and cache the KDTree for a given hsi_path."""
    coord_files = []
    coord_list = []

    # Find all coordinate files in the directory
    for file in os.listdir(hsi_path):
        if file.endswith(".npz"):
            try:
                # Extract the coordinates from the file name
                coord_x, coord_y, _ = file.split("_")
                coord_x, coord_y = float(coord_x), float(coord_y)
                coord_files.append(file)
                coord_list.append((coord_x, coord_y))
            except ValueError:
                continue  # Skip files that do not match the expected format

    if not coord_list:
        raise ValueError("No valid coordinate files found in the directory.")

    # Build KDTree and return it along with the coordinate files
    tree = KDTree(coord_list)
    return tree, coord_files, coord_list


class ImageDataset(Dataset):
    image_groups = ["rgb-red", "rgb-white", "hsi"]

    def __init__(
        self,
        df: pd.DataFrame,
        channel_stats: dict[str, dict[str, float]],
        target_mask_kernel_size: int,
        # size of image patch before augmentation
        crop_size_init: int | dict[str, int],
        # size of image patch after augmentation, i.e, network input
        crop_size_final: int | dict[str, int],
        hsi_wavelengths: np.ndarray,
        hsi_ceil: int = None,
        split: str = "train",
        _hsi_group_k: int = 0,
        _hsi_crop_size: int = 0,
        _hsi_coord_digits: int = 3,
        _calc_stats_mode: bool = False,
        _hsi_avg_dim: int | None = None,
        _couple_rgb: bool = False,
    ):
        """
        Initialize the dataset with the given parameters.

        Args:
            df (pd.DataFrame): DataFrame with precomputed project, plate, and image information.
            target_mask_kernel_size (int): Size of the kernel for Gaussian images.
            crop_size_init (int): Size of the crop for geometric transformations.
            crop_size_final (int): Size of the initial crop centered at the coordinates.
            num_hsi_wavelengths (int): Number of HSI channels.
        """
        self.df = df
        self.target_mask_kernel_size = target_mask_kernel_size
        if isinstance(crop_size_init, int):
            crop_size_init = {k: crop_size_init for k in self.image_groups}
        self.crop_size_init = crop_size_init
        if isinstance(crop_size_final, int):
            crop_size_final = {k: crop_size_final for k in self.image_groups}
        self.crop_size_final = crop_size_final
        self.hsi_wavelengths = hsi_wavelengths
        self.hsi_ceil = hsi_ceil
        self._hsi_group_k = _hsi_group_k
        self._hsi_crop_size = _hsi_crop_size
        self._hsi_coord_digits = _hsi_coord_digits
        self._hsi_avg_dim = _hsi_avg_dim

        # some sanity checks
        if self._hsi_group_k > 0 and self._hsi_crop_size > 0:
            raise ValueError("Cannot specify both _hsi_group_k and _hsi_crop_size.")

        self.transform = GroupTransform(
            channel_stats=channel_stats,
            crop_size=crop_size_final,
            # hsi_ceil=hsi_ceil,
            split=split,
            _calc_stats_mode=_calc_stats_mode,
            _couple_rgb=_couple_rgb,
        )
        self._transform_for_visual = GroupTransform(
            channel_stats=channel_stats,
            crop_size=crop_size_final,
            # hsi_ceil=hsi_ceil,
            split=split,
            _calc_stats_mode=_calc_stats_mode,
            _no_norm=True,
            _couple_rgb=_couple_rgb,
        )

    def __len__(self) -> int:
        """
        Get the total number of images in the dataset.

        Returns:
            int: Total number of images.
        """
        return len(self.df)

    def getitem_no_norm(self, idx: int) -> dict[str, torch.Tensor]:
        _transform, self.transform = self.transform, self._transform_for_visual
        data = self.__getitem__(idx)
        self.transform = _transform
        return data

    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample from the dataset.

        A sample is defined as one picked isolate from one plate of one project. The label
            of the samples are precomputed and stored in the dataframe. The input images
            consists of multiple image groups: rgb-red, rgb-white, and hsi. Each image
            group can have multiple images across many time points.

        An isolate has a coordinate which needs to be transformed into the coord frame of
            each image group. Each image is cropped to a square patch of predetermined
            size (an attribute on the dataset object) and centered at the transformed
            coord. This transformation callable is also precomputed and stored in the df.

        Some randomness is added when choosing images from each image group.
            - First, a image group could be dropped out (replaced with all 0).
            - Second, each image group could have some images dropped out.
            - Thrid, one time point is randomly selected from available time points.

        All the randomness is precomputed and stored in the dataframe.


        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Transformed images and labels.
        """
        row = self.df.iloc[idx]
        # coords = row[["coord_x", "coord_y"]].to_numpy()
        data = {}

        for ig in row["rand_chosen_igs"]:
            # Both groups of rgb-red and rgb-white only have one image and therefore
            # no image dropout is needed.
            crop_size_init = self.crop_size_init[ig]
            if ig == "hsi":
                # 0 means a channel is dropped out, 1 means kept.
                hsi_channel_dropout = row["rand_channel_dropout_hsi"]
                coords_hsi = row[["coord_x_hsi", "coord_y_hsi"]].to_numpy()
                # For efficiency in cropping and augmentation, we stack all the images
                # that are not dropped in HSI group.
                if self._hsi_crop_size:
                    crop_center = np.array([self._hsi_crop_size] * 2) / 2
                else:
                    crop_center = coords_hsi
                img = crop_image(
                    self._read_hsi(
                        row[f"rand_image_{ig}_path"],
                        hsi_channel_dropout,
                        hsi_coord=coords_hsi,
                    ),
                    center=crop_center,
                    crop_size=crop_size_init,
                )
            else:
                coords_rgb = row[["coord_x_rgb", "coord_y_rgb"]].to_numpy()
                img = crop_image(
                    # np.array(Image.open(row[f"rand_image_{ig}_path"]).convert("RGB")),
                    cv.cvtColor(
                        cv.imread(row[f"rand_image_{ig}_path"], cv.IMREAD_COLOR),
                        cv.COLOR_BGR2RGB,
                    ),
                    center=coords_rgb,
                    crop_size=crop_size_init,
                )
            target_mask = gaussian_image(
                positions=(crop_size_init / 2, crop_size_init / 2),
                shape=(crop_size_init, crop_size_init),
                fwhm=self.target_mask_kernel_size,
            )
            data_m = {"image": img, "target_mask": target_mask}
            if ig == "hsi":
                data_m["hsi_channel_dropout"] = hsi_channel_dropout
            data[ig] = data_m

        data_aug = self.transform(data)

        # post processing to give collate_fn a easier time
        for ig in self.image_groups:
            if ig in data_aug:
                data_aug[ig]["dropped"] = False
                data_aug[ig]["available"] = True
                data_aug[ig]["time_point"] = row[f"rand_image_{ig}_tp"]
                data_aug[ig]["time_points"] = ",".join(
                    map(str, row["image_groups_info"][ig].keys())
                )
            else:
                if ig == "hsi":
                    shape = (
                        len(self.hsi_wavelengths),
                        self.crop_size_final[ig],
                        self.crop_size_final[ig],
                    )
                else:
                    shape = (3, self.crop_size_final[ig], self.crop_size_final[ig])
                data_aug[ig] = {
                    "image": torch.zeros(*shape),
                    "target_mask": torch.zeros(*shape[1:]),
                    "time_point": -999,
                    "time_points": "",
                }
                if ig == "hsi":
                    data_aug[ig]["hsi_channel_dropout"] = torch.zeros(
                        len(self.hsi_wavelengths), dtype=torch.bool
                    )
                if ig in row["image_groups_info"]:
                    data_aug[ig]["dropped"] = True
                    data_aug[ig]["available"] = True
                else:
                    data_aug[ig]["dropped"] = False
                    data_aug[ig]["available"] = False

        meta = {
            "project_id": row["project"],
            "plate_id": row["plate"],
            "isolate_id": row["isolate"],
            "index_in_df": self.df.index[idx],
        }

        return {
            "data": data_aug,
            "meta": meta,
            "label": row["label_clean_idx"],
            "label_all_levels": torch.tensor(row["label_all_levels_clean_idx"]),
        }

    def _read_hsi(
        self,
        hsi_path,
        hsi_channel_dropout,
        t: int = 4,
        hsi_coord: tuple[float, float] | None = None,
    ) -> np.ndarray:
        read_img_func = lambda x: cv.imread(
            os.path.join(hsi_path, x), cv.IMREAD_UNCHANGED
        )
        if self._hsi_crop_size > 0:
            tree, coord_files, coord_list = get_kdtree(hsi_path)
            dist, index = tree.query(hsi_coord)

            # Assert the nearest distance is less than 0.1
            if dist >= 0.1:
                nearest_coord = coord_list[index]
                raise ValueError(
                    f"Nearest coordinate {nearest_coord} found with distance {dist:.4f} "
                    f"to expected coordinate {hsi_coord}. Distance must be < 0.1."
                )

            # Use the nearest neighbor's file name
            file_name = coord_files[index]
            img = np.load(os.path.join(hsi_path, file_name))["data"]

            if self._hsi_crop_size > 0:
                if self._hsi_avg_dim:
                    img = adaptive_avg_pool(img, self._hsi_avg_dim)
                img = img[..., hsi_channel_dropout]

        elif self._hsi_group_k > 0:
            # hsi is stored as channel-grouped 16bit tif, for example 3 channels per
            # group, so that they can still be visually inspected. However in this case
            # we should be more cautious about the dropout.
            image_paths = []
            wavelengths_read = []
            for i in range(0, len(self.hsi_wavelengths), self._hsi_group_k):
                if any(hsi_channel_dropout[i : i + self._hsi_group_k]):
                    ws = self.hsi_wavelengths[i : i + self._hsi_group_k]
                    image_paths.append("_".join(map(str, ws)) + ".tif")
                    wavelengths_read.extend(
                        list(
                            range(
                                i, min(i + self._hsi_group_k, len(self.hsi_wavelengths))
                            )
                        )
                    )
            if t > 1:  # use joblib
                img = np.concatenate(
                    Parallel(n_jobs=t, prefer="threads")(
                        delayed(read_img_func)(os.path.join(hsi_path, path))
                        for path in image_paths
                    ),
                    axis=-1,
                )
            else:
                img = np.concatenate(
                    [read_img_func(path) for path in image_paths], axis=-1
                )
            # drop those that are not needed
            img = img[..., np.where(hsi_channel_dropout[wavelengths_read])[0]]
        else:  # hsi is stored as channel-wise gray scale 16bit png.
            if t > 1:  # use joblib
                img = np.dstack(
                    Parallel(n_jobs=t, prefer="threads")(
                        delayed(read_img_func)(f"{wl}.png")
                        for wl in self.hsi_wavelengths
                    )
                )
            else:
                img = np.dstack(
                    [
                        read_img_func(f"{wl}.png")
                        for wl in self.hsi_wavelengths[hsi_channel_dropout]
                    ]
                )
        if self.hsi_ceil is not None:
            # img = img / self.hsi_ceil
            # starting from albumentations 1.4.10, GaussNoise forces float input to be
            # float32 (otherwise opencv will throw error), which I don't really like or
            # understand. I think this should be a bug since other transforms don't
            # have this constraint.
            img = (img / np.quantile(img, 0.99)).clip(0, 1).astype(np.float32)
        return img

    @staticmethod
    def _crop_image(
        img: np.ndarray, center: tuple[int, int], crop_size: int
    ) -> np.ndarray:
        """
        Crop the image centered at the given coordinates.

        Args:
            img (np.ndarray): Image to be cropped.
            center (tuple[int, int]): Center coordinates for the crop.

        Returns:
            np.ndarray: Cropped image.
        """
        x, y = center
        half_crop = crop_size // 2
        img_pil = Image.fromarray(img)
        img_cropped = img_pil.crop(
            (x - half_crop, y - half_crop, x + half_crop, y + half_crop)
        )
        return np.array(img_cropped)


def gaussian_image(
    positions: np.ndarray | tuple[float, float],
    fwhm: float,
    shape: np.ndarray | None = None,
    pois: np.ndarray | None = None,
    trunc: float = 1e-5,
    normalize: bool = True,
) -> np.ndarray:
    """
    Generate a Gaussian distribution centered at multiple positions with specified FWHM,
    clipped at 1e-5 and optionally normalized to sum to 1, on a canvas of given shape.

    Args:
        positions (numpy.ndarray): An array of shape [num_centers, 2] with the (x, y) coordinates for the centers.
        shape (tuple): The shape (height, width) of the background canvas.
        fwhm (float): The full width at half maximum (FWHM) of the Gaussian distribution.
        trunc (float, optional): The truncation threshold for the Gaussian values. Default is 1e-5.
        normalize (bool, optional): Whether to normalize the final output to sum to 1. Default is True.

    Returns:
        numpy.ndarray: A 2D array containing the combined Gaussian distributions on a zero background.
    """
    # Convert FWHM to standard deviation (sigma)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    if shape is not None:  # return a 2d array
        if pois is not None:
            raise ValueError("Cannot specify both shape and pois.")
        height, width = shape
        if not isinstance(positions, np.ndarray):
            positions = np.array([positions])
        elif positions.ndim == 1:
            positions = positions.reshape(1, -1)

        # Validate positions
        if positions.min() < 0:
            raise ValueError("Positions must be non-negative.")
        if positions[:, 0].max() >= width or positions[:, 1].max() >= height:
            raise ValueError("Positions must be within the canvas.")

        # Create a meshgrid for the entire canvas
        x = np.linspace(0, width - 1, width)
        y = np.linspace(0, height - 1, height)
        # x, y = np.meshgrid(x, y)
        # grid = np.dstack((x, y))

        # Initialize the output canvas
        exponent_x = (x[:, None] - positions[:, 0]) ** 2  # [X, P]
        exponent_y = (y[:, None] - positions[:, 1]) ** 2  # [Y, P]
        # [X, Y, P]
        exponent = -(exponent_x[:, None] + exponent_y[None, ...]) / (2 * sigma**2)
        z = 1 / (2 * np.pi * sigma**2) * np.exp(exponent).sum(axis=-1).transpose()
        # Truncate values below truncation threshold
        z[z < trunc] = 0
        z /= z.sum()
        if not normalize:  # Normalize if required
            z *= len(positions)
    else:
        if pois is None:
            raise ValueError("Must specify either shape or pois.")
        if not isinstance(pois, np.ndarray):
            pois = np.array([pois])
        elif pois.ndim == 1:
            pois = pois.reshape(1, -1)
        exponent_x = (pois[:, 0:1] - positions[:, 0]) ** 2  # [Q, P]
        exponent_y = (pois[:, 1:2] - positions[:, 1]) ** 2  # [Q, P]
        # [Q, P]
        exponent = -(exponent_x + exponent_y) / (2 * sigma**2)
        z = 1 / (2 * np.pi * sigma**2) * np.exp(exponent)
        if normalize:  # Normalize if required
            z = z.mean(axis=-1)
        else:
            z = z.sum(axis=-1)

    return z


def test_gaussian_image(plot: bool = True) -> None:
    # Testing the function with a non-integer position and kernel size
    position = np.array([[10.3, 5.9], [20.6, 15], [2, 10]])  # Non-integer position
    shape = (20, 30)  # Canvas size
    fwhm = 3.5  # Non-integer kernel size

    gaussian_canvas = gaussian_image(position, shape=shape, fwhm=fwhm, normalize=False)
    assert np.isclose(gaussian_canvas.sum(), len(position))

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        img = ax.imshow(gaussian_canvas, cmap="gray")
        fig.colorbar(img, ax=ax, location="right", anchor=(0, 0.5), shrink=0.3)


class TAMPICDataModule(L.LightningDataModule):
    # selecting only 1 image group is the most likely.
    default_p_num_igs = {1: 2, 2: 1, 3: 1}
    # selecting rgb-white is the most likely, followed by rgb-red, and hsi.
    default_p_igs = {"rgb-red": 2, "rgb-white": 3, "hsi": 1}

    # for validation set, let's default to selecting as many as possible
    default_p_num_igs_val = {k: 0 for k in default_p_num_igs.keys()}
    default_p_num_igs_val[max(default_p_num_igs.keys())] = 1
    # doesn't matter, everyone will be selected anyways
    default_p_igs_val = {k: 1 for k in default_p_igs.keys()}

    def __init__(
        self,
        *,
        metadata_train_path: str,
        val_easy_ratio: float = 0.3,
        # metadata_val_easy_path: dict,
        # metadata_val_mid_path: dict,
        #
        weight_by_label: bool = True,
        weight_by_density: bool = False,
        weight_density_kernel_size: int = 5,
        weight_by_plate: bool = False,
        #
        p_num_igs: Optional[list[int]] = None,
        p_igs: Optional[list[int]] = None,
        p_last_time_point: Optional[list[int]] = 0.5,
        p_hsi_channels: float = 0.3,
        rng_seed: Optional[int] = 42,
        #
        p_num_igs_val: Optional[list[int]] = None,
        p_igs_val: Optional[list[int]] = None,
        p_last_time_point_val: Optional[list[int]] = 0.5,
        p_hsi_channels_val: float = 1.0,
        rng_seed_val: Optional[int] = 60,  # let's separate the seeds
        #
        amplicon_type: str = None,
        taxon_level: str = "genus",
        min_counts: int = 10,
        min_counts_dominate: int = 0,
        min_purity: float = 0.1,
        min_ratio: int = 2,
        min_num_isolates: int = 30,
        keep_empty: bool = False,
        keep_others: bool = True,
        #
        crop_size_init: int | dict[str, int],
        crop_size_final: int | dict[str, int],
        target_mask_kernel_size: int,
        #
        num_devices: int,
        batch_size: int,
        num_workers: int,
        prefetch_factor: int = 2,
        num_batches_per_epoch: Optional[int] = None,
        #
        _hsi_avg_dim: int | None = None,
        _hsi_group_k: int = 0,
        _hsi_crop_size: int = 0,
        _hsi_norm: int = False,
        _stats_key: str = "0618_16s",
        _k_per_sample: int = 60,
        _calc_stats_mode: bool = False,
        _couple_rgb: bool = False,
        _hsi_wavelengths_overwrite: None | int = None,
        _log: int = 1,
    ):
        """
        Initialize the data module with the given parameters.

        Args:
            metadata_train_path (dict): Metadata for the training set.
            metadata_val_easy_path (dict): Metadata for the first validation set.
            metadata_val_mid_path (dict): Metadata for the second validation set.
            target_mask_kernel_size (int): Size of the kernel for Gaussian images.
            crop_size_init (int): Size of the crop for geometric transformations.
            crop_size_final (int): Size of the initial crop centered at the coordinates.
            num_hsi_channels (int): Number of HSI channels.
            batch_size (int): Batch size.
            num_workers (int): Number of workers for data loading.
            p_num_igs (list[int], optional): Probability weights for number of image groups. Defaults to [2, 1, 1].
            p_igs (list[int], optional): Probability weights for each image group. Defaults to [2, 3, 1].
            sampling_weights (list[float], optional): Weights for sampling each sample. Defaults to None.
            num_batches_per_epoch (int, optional): Length of an epoch. Defaults to None.
        """
        super().__init__()
        self.save_hyperparameters()

        # data metadata
        self.metadata_train_path = metadata_train_path
        self.val_easy_ratio = val_easy_ratio
        # self.metadata_val_easy_path = metadata_val_easy_path
        # self.metadata_val_mid_path = metadata_val_mid_path

        # arguments for calculating weights
        self.weight_by_label = weight_by_label
        self.weight_by_density = weight_by_density
        self.weight_density_kernel_size = weight_density_kernel_size
        self.weight_by_plate = weight_by_plate

        # arguments for choosing images
        self.p_num_igs = p_num_igs if p_num_igs else self.default_p_num_igs
        self.p_igs = p_igs if p_igs else self.default_p_igs
        self.p_last_time_point = p_last_time_point
        self.p_hsi_channels = p_hsi_channels
        self._rng = np.random.default_rng(rng_seed)
        # the above for validation
        self.p_num_igs_val = (
            p_num_igs_val if p_num_igs_val else self.default_p_num_igs_val
        )
        self.p_igs_val = p_igs_val if p_igs_val else self.default_p_igs_val
        self.p_last_time_point_val = p_last_time_point_val
        self.p_hsi_channels_val = p_hsi_channels_val
        self._rng_val = np.random.default_rng(rng_seed_val)

        # arguments for getting labels
        self.amplicon_type = amplicon_type
        self.taxon_level = taxon_level
        self.min_counts = min_counts
        self.min_counts_dominate = min_counts_dominate
        self.min_purity = min_purity
        self.min_ratio = min_ratio
        self.min_num_isolates = min_num_isolates
        self.keep_empty = keep_empty  # `empty` will be one of the target labels
        self.keep_others = keep_others  # `others` will be one of the target labels

        # augmentation
        self.crop_size_init = crop_size_init
        self.crop_size_final = crop_size_final
        self.target_mask_kernel_size = target_mask_kernel_size

        # batching
        self.num_devices = num_devices
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        # others
        self._hsi_avg_dim = _hsi_avg_dim
        self._hsi_group_k = _hsi_group_k
        self._hsi_crop_size = _hsi_crop_size
        self._hsi_norm = _hsi_norm
        self._stats_key = _stats_key
        self._k_per_sample = _k_per_sample
        self._calc_stats_mode = _calc_stats_mode
        self._couple_rgb = _couple_rgb
        self._hsi_wavelengths_overwrite = _hsi_wavelengths_overwrite
        self._log = _log

    @staticmethod
    def _report_df_stats(df: pd.DataFrame) -> None:
        num_isolates = len(df)
        num_projects = len(df["project"].unique())
        num_plates = len(df.groupby(["project", "plate"]))
        num_media = df["medium_type"].nunique()
        rprint(
            f"Data loading completed. There are in total {num_isolates} isolates from "
            f"{num_projects} projects, {num_plates} plates, and {num_media} medium types."
        )

    def _report_label_stats(self, df: pd.DataFrame) -> None:
        # report how many isolates are fitted as what and how many are dropped
        num_labels_raw = df["label"].nunique()
        num_dropped = df.query("label in ['impure', 'ambiguous']").shape[0]
        num_empty = df.query("label == 'empty'").shape[0]
        rprint("Label fitting completed.")
        rprint(f"\tThere are {num_labels_raw} unique labels.")
        rprint(f"\t{num_dropped} are dropped for being 'impure' or 'ambiguous'.")
        rprint(
            f"\t{num_empty} 'empty' isolates are {'kept' if self.keep_empty else 'dropped'}."
        )
        rprint(
            f"\t{len(self.label2idx) - len(self.label_clean2idx) + 1} labels are "
            f"aggregated into 'others' and {'kept' if self.keep_others else 'dropped'}."
        )

    def setup(self, stage: Optional[str] = None):
        """
        Set up the datasets for training and validation.

        Args:
            stage (Optional[str]): Stage for which to set up the data. Defaults to None.
        """
        if stage in (None, "fit"):
            df_all = self._json_to_dataframe(self.metadata_train_path)
            self._fit_labels(df_all.query("split == 'train' | split == 'val_easy'"))
            # self._calc_training_stats(df_train_all)

            df_train_all = df_all.query("split == 'train'").copy()
            df_val_easy = df_all.query("split == 'val_easy'").copy()
            df_val_mid = df_all.query("split == 'val_mid'").copy()
            self.df_train_all = self._add_weight(self._transform_labels(df_train_all))
            self.df_val_easy = self._transform_labels(df_val_easy)
            self.df_val_mid = self._transform_labels(df_val_mid)

            if self._log:
                self._report_df_stats(df_all)
                self._report_label_stats(df_all)

            def _get_label_dist(x):
                return {
                    k: x["label_clean"].value_counts().to_dict().get(k, 0)
                    for k in self.label_clean2idx
                }

            label_dist_df = pd.DataFrame(
                {
                    "train": _get_label_dist(self.df_train_all),
                    "val_easy": _get_label_dist(self.df_val_easy),
                    "val_mid": _get_label_dist(self.df_val_mid),
                },
                index=self.label_clean2idx,
            )
            self.label_dist_df = label_dist_df
            rprint("Final size of datasets:")
            rprint(f"\tTrain: {len(self.df_train_all)}. ")
            rprint(f"\tVal easy: {len(self.df_val_easy)}. ")
            rprint(f"\tVal mid: {len(self.df_val_mid)}. ")
            rprint(
                tabulate(
                    label_dist_df, label_dist_df.columns, tablefmt="rounded_outline"
                )
            )
        else:
            raise NotImplementedError(f"Stage {stage} not supported.")

    def train_dataloader(self):
        """
        Get the training dataloader.

        Returns:
            DataLoader: Training dataloader.
        """
        df_train = self._sample_training_data(self.df_train_all)
        df_train = self._set_randomness(df_train)
        train_dataset = ImageDataset(
            df=df_train,
            channel_stats=self.stats,
            target_mask_kernel_size=self.target_mask_kernel_size,
            crop_size_init=self.crop_size_init,
            crop_size_final=self.crop_size_final,
            hsi_wavelengths=self.hsi_wavelengths,
            hsi_ceil=self.hsi_ceil,
            split="train",
            _hsi_group_k=self._hsi_group_k,
            _hsi_crop_size=self._hsi_crop_size,
            _calc_stats_mode=self._calc_stats_mode,
            _hsi_avg_dim=self._hsi_avg_dim,
            _couple_rgb=self._couple_rgb,
        )

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            # pin_memory=True,
            persistent_workers=True,
            shuffle=False,
        )

    def val_dataloader(self):
        """
        Get the validation dataloaders.

        Returns:
            list[DataLoader]: list of validation dataloaders.
        """
        df_val_easy = self._set_randomness(self.df_val_easy, split="val")
        val_dataset_1 = ImageDataset(
            df=df_val_easy,
            channel_stats=self.stats,
            target_mask_kernel_size=self.target_mask_kernel_size,
            crop_size_init=self.crop_size_init,
            crop_size_final=self.crop_size_final,
            hsi_wavelengths=self.hsi_wavelengths,
            hsi_ceil=self.hsi_ceil,
            split="val",
            _hsi_group_k=self._hsi_group_k,
            _hsi_crop_size=self._hsi_crop_size,
            _hsi_avg_dim=self._hsi_avg_dim,
            _couple_rgb=self._couple_rgb,
        )

        df_val_mid = self._set_randomness(self.df_val_mid, split="val")
        val_dataset_2 = ImageDataset(
            df=df_val_mid,
            channel_stats=self.stats,
            target_mask_kernel_size=self.target_mask_kernel_size,
            crop_size_init=self.crop_size_init,
            crop_size_final=self.crop_size_final,
            hsi_wavelengths=self.hsi_wavelengths,
            hsi_ceil=self.hsi_ceil,
            split="val",
            _hsi_group_k=self._hsi_group_k,
            _hsi_crop_size=self._hsi_crop_size,
            _hsi_avg_dim=self._hsi_avg_dim,
            _couple_rgb=self._couple_rgb,
        )

        return [
            DataLoader(
                val_dataset_1,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                # pin_memory=True,
                persistent_workers=True,
                shuffle=False,
            ),
            DataLoader(
                val_dataset_2,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                # pin_memory=True,
                persistent_workers=True,
                shuffle=False,
            ),
        ]

    def _json_to_dataframe(
        self, metadata_path: str, random_split_seed: int = 2024
    ) -> pd.DataFrame:
        """
        Convert JSON metadata to a Pandas DataFrame and add random components.

        Args:
            metadata (dict): JSON metadata.
            split (str): Split type ('train' or 'val'). This random seed stands alone
                from the rest of the random components, so that the split is controled
                independently.

        Returns:
            pd.DataFrame: DataFrame with precomputed project, plate, and image information.
        """
        metadata_dir = os.path.dirname(metadata_path)
        with open(metadata_path) as f:
            metadata = json.load(f)
        # load global properties
        global_properties = metadata["global_properties"]

        self._num_hsi_channels = global_properties["hsi_num_channels"]
        self._hsi_wavelengths = np.loadtxt(
            os.path.join(metadata_dir, global_properties["hsi_wavelengths"])
        )
        assert self._num_hsi_channels == len(self._hsi_wavelengths)
        if self._hsi_wavelengths_overwrite is None:
            self.hsi_wavelengths = self._hsi_wavelengths
            self.num_hsi_channels = self._num_hsi_channels
        else:
            self.hsi_wavelengths = np.array(self._hsi_wavelengths_overwrite)
            self.num_hsi_channels = len(self.hsi_wavelengths)
        
        self._hsi_ceil = global_properties["hsi_ceil"]
        self.hsi_ceil = self._hsi_ceil if self._hsi_norm else None
        self.taxon_levels = np.array(global_properties["taxonomy_levels"])
        self.taxon_level2idx = {k: i for i, k in enumerate(self.taxon_levels)}
        self.taxon_level_idx = self.taxon_level2idx[self.taxon_level]
        self.stats = global_properties["stats"][self._stats_key]  # for normalization
        if self._hsi_avg_dim is not None:
            if self._hsi_avg_dim >= len(self.hsi_wavelengths):
                self._hsi_avg_dim = None
            else:
                self.hsi_wavelengths = adaptive_avg_pool(
                    self.hsi_wavelengths, self._hsi_avg_dim
                )
                self.num_hsi_channels = self.hsi_wavelengths.shape[0]
                self.stats["hsi"]["mean"] = adaptive_avg_pool(
                    np.array(self.stats["hsi"]["mean"]), self._hsi_avg_dim
                ).tolist()
                self.stats["hsi"]["std"] = adaptive_avg_pool(
                    np.array(self.stats["hsi"]["std"]), self._hsi_avg_dim
                ).tolist()

        # load information of each isolate (project id, plate id isolate id, available
        # modalities, available time point for each modality, path to all of the images
        # in dict, label)
        dfs = []
        rng = np.random.default_rng(random_split_seed)
        for proj, meta_proj in metadata["data"].items():
            df_proj = self._dict_to_dataframe_proj(
                meta_proj,
                base_dir=os.path.join(metadata_dir, proj),
                random_seed=rng.integers(0, 2**32 - 1),
            )
            # if df_proj is not None:
            df_proj["project"] = proj
            dfs.append(df_proj)
        df = pd.concat(dfs).reset_index(drop=True)
        # df["split"] = df["split"].replace("val", "val_mid")
        # rng = np.random.default_rng(random_split_seed)
        # df.loc[
        #     rng.choice(
        #         df.query("split == 'train'").index,
        #         size=int(len(df) * self.val_easy_ratio),
        #         replace=False,
        #     ),
        #     "split",
        # ] = "val_easy"
        return df

    def _dict_to_dataframe_proj(
        self, meta_proj: dict, base_dir: str, random_seed: int
    ) -> pd.DataFrame:
        # if meta_proj["status"] != "valid":
        #     return None
        with open(f"{base_dir}/{meta_proj['amplicon_info']}") as f:
            amplicon_info = json.load(f)
        dfs_proj = []
        for plate, meta_plate in meta_proj["plates"].items():
            rows = []
            if meta_plate["status"] != "valid":
                continue
            sample_type = meta_plate.get("sample_type", "unknown")
            medium_type = meta_plate.get("medium_type", "unknown")
            num_isolates = len(amplicon_info[plate])
            image_groups_info = {}
            if "rgb" in meta_plate["images"]:
                time_points_rgb = {
                    # int(k[1:]): {"red": v["red"], "white": v["white"]}
                    int(k[1:]): {"red": v.get("red"), "white": v.get("white")}
                    for k, v in meta_plate["images"]["rgb"].items()
                    if k != "default" and v["status"] == "valid"
                }
                image_groups_info["rgb-red"] = {
                    k: os.path.join(base_dir, v["red"])
                    for k, v in time_points_rgb.items()
                    if v["red"]
                }
                image_groups_info["rgb-white"] = {
                    k: os.path.join(base_dir, v["white"])
                    for k, v in time_points_rgb.items()
                    if v["white"]
                }
                transform_info = meta_plate["isolate_transform"]["to_rgb"]
                func_transform_iso2rgb = get_query2target_func(
                    **transform_info["params"],
                    **transform_info["stats"],
                    flip=transform_info["flip"],
                )
            else:
                func_transform_iso2rgb = None
            if "hsi" in meta_plate["images"]:
                time_points_hsi = {
                    int(k[1:]): {
                        "dir": os.path.join(base_dir, v["dir"]),
                        "transform": v["transform"]["from_rgb"],
                    }
                    for k, v in meta_plate["images"]["hsi"].items()
                    if k != "default" and v["status"] == "valid"
                }
                image_groups_info["hsi"] = time_points_hsi
            image_groups_available = [k for k, v in image_groups_info.items() if v]

            # for removing plates missing nessesary image groups.
            p = np.array([self.p_igs[ig] for ig in image_groups_available])
            p_val = np.array([self.p_igs_val[ig] for ig in image_groups_available])
            if not (p.max() > 0 and p_val.max() > 0):
                continue

            for isolate, meta_isolate in amplicon_info[plate].items():
                if self.amplicon_type not in meta_isolate:
                    continue
                label, tax_all_levels = get_isolate_label(
                    meta_isolate[self.amplicon_type],
                    min_counts=self.min_counts,
                    min_counts_dominate=self.min_counts_dominate,
                    min_ratio=self.min_ratio,
                    min_purity=self.min_purity,
                    taxon_level_idx=self.taxon_level_idx,
                )
                if tax_all_levels is None:
                    tax_all_levels = [label] * (self.taxon_level_idx + 1)
                rows.append(
                    {
                        "plate": plate,
                        "sample_type": sample_type,
                        "medium_type": medium_type,
                        "num_isolates": num_isolates,
                        # "transform_iso2rgb": meta_plate["transform"][
                        #     "isolate_to_rgb"
                        # ],
                        # "transform_rgb2hsi": meta_plate["transform"]["rgb_to_hsi"],
                        "image_groups_info": image_groups_info,
                        "image_groups_available": image_groups_available,
                        # "available_time_points_rgb-red": available_time_points_rgb-red,
                        # "available_time_points_rgb-white": available_time_points_rgb-white,
                        # "available_time_points_hsi": available_time_points_hsi,
                        "isolate": isolate,
                        "coord_x": meta_isolate["coord"][0],
                        "coord_y": meta_isolate["coord"][1],
                        "label": label,
                        # "label_all_levels": {
                        #     l: t
                        #     for l, t in zip(
                        #         self.taxon_levels[: self.taxon_level_idx],
                        #         tax_all_levels,
                        #     )
                        # },
                        "label_all_levels": tax_all_levels,
                        "split": meta_plate["split"][self.amplicon_type],
                    }
                )
            if not rows:
                continue
            df_plate = pd.DataFrame(rows)
            # transform isolate coords on to the frame of RGB.
            # We assume all samples have RGB images.
            # Coords in HSI frame are calculated later in the step of calculating
            # randomness.
            if func_transform_iso2rgb:
                df_plate[["coord_x_rgb", "coord_y_rgb"]] = func_transform_iso2rgb(
                    df_plate[["coord_x", "coord_y"]].to_numpy()
                )
            else:
                df_plate[["coord_x_rgb", "coord_y_rgb"]] = None
            dfs_proj.append(df_plate)
            # if func_transform_rgb2hsi:
            #     if func_transform_iso2rgb is None:
            #         raise NotImplementedError(
            #             "RGB must be available to transform to HSI"
            #         )
            #     df_plate[["coord_x_hsi", "coord_y_hsi"]] = func_transform_rgb2hsi(
            #         df_plate[["coord_x_rgb", "coord_y_rgb"]].to_numpy()
            #     )
            # else:
            #     df_plate[["coord_x_hsi", "coord_y_hsi"]] = None

        df_proj = pd.concat(dfs_proj).reset_index(drop=True)
        df_proj["split"] = df_proj["split"].replace("val", "val_mid")
        train_split_idx = df_proj.query("split == 'train'").index
        if len(train_split_idx) > 0:
            rng = np.random.default_rng(random_seed)
            df_proj.loc[
                rng.choice(
                    train_split_idx,
                    size=int(len(df_proj) * self.val_easy_ratio),
                    replace=False,
                ),
                "split",
            ] = "val_easy"
        return df_proj

    def _fit_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add another column `_label` to self.df_train which is an integer
            representation of the label.

        Labels with at least self.min_num_isolates are kept and transformed to numerical
            values. Labels with fewer isolates are transformed to "others" if
            self.keep_others is True.

        Therefore, `empty` will only be one of the label if:
            1. self.keep_empty is True
            2. At least self.min_num_isolates are empty.
        """
        label2idx_all: list[dict[str, int]] = []
        label_clean2idx_all: list[dict[str, int]] = []
        idx2label_clean_all: list[dict[int, str]] = []
        # label_counts = df["label"].value_counts().to_dict()

        for idx in range(self.taxon_level_idx + 1):
            label_counts = (
                pd.Series(
                    [
                        label_all_levels[idx]
                        for label_all_levels in df["label_all_levels"]
                    ]
                )
                .value_counts()
                .to_dict()
            )
            label2idx = {}
            label_clean2idx = {}
            idx2label_clean = {}

            iter_label_counts = iter(label_counts.items())
            i = 0
            for label, count in iter_label_counts:
                if label in ["impure", "ambiguous"]:
                    continue
                if count >= self.min_num_isolates:
                    if self.keep_empty or label != "empty":
                        label2idx[label] = i
                        label_clean2idx[label] = i
                        idx2label_clean[i] = label
                        i += 1
                else:
                    break
            if self.keep_others:  # keep iterating
                label_clean2idx["others"] = i
                idx2label_clean[i] = "others"
                for label, count in iter_label_counts:
                    if label in ["impure", "ambiguous"]:
                        continue
                    label2idx[label] = i
                    i += 1

            label2idx_all.append(label2idx)
            label_clean2idx_all.append(label_clean2idx)
            idx2label_clean_all.append(idx2label_clean)

        self.label2idx = label2idx_all[self.taxon_level_idx]
        self.label_clean2idx = label_clean2idx_all[self.taxon_level_idx]
        self.idx2label_clean = idx2label_clean_all[self.taxon_level_idx]

        self.label2idx_all = label2idx_all
        self.label_clean2idx_all = label_clean2idx_all
        self.idx2label_clean_all = idx2label_clean_all

    def _transform_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the labels in the DataFrame to numerical values."""
        # first map all legit taxon name and "others" and "empty" (conditioned on hparams) to idx
        # all other labels are dropped.
        df["label_idx"] = df["label"].map(self.label2idx.get)
        df = df.dropna(subset=["label_idx"]).copy()
        df["label_idx"] = df["label_idx"].astype(int)
        # then we map labels to clean labels, and drop those that are not in clean labels,
        # for example, when not keeping others, all minor labels are dropped here.
        df["label_clean"] = df["label_idx"].map(self.idx2label_clean)
        df = df.dropna(subset=["label_clean"]).copy()
        df["label_clean_idx"] = df["label_clean"].map(self.label_clean2idx).astype(int)

        label_clean_idx_all_levels: list[list[int]] = []
        for idx, (label2idx, label_clean2idx, idx2label_clean) in enumerate(
            zip(self.label2idx_all, self.label_clean2idx_all, self.idx2label_clean_all)
        ):
            label = df["label_all_levels"].apply(lambda x: x[idx])
            label_idx = label.map(label2idx)
            label_clean = label_idx.map(idx2label_clean)
            label_clean_idx = label_clean.map(label_clean2idx)
            label_clean_idx_all_levels.append(label_clean_idx.to_list())
        # [num_levels, num_rows]
        label_clean_idx_all_levels_arr = np.array(label_clean_idx_all_levels).T
        df["label_all_levels_clean_idx"] = pd.Series(
            [i for i in label_clean_idx_all_levels_arr], index=df.index
        )
        return df

    def _add_weight(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a weight column to the DataFrame based on the label.
        Samples can be weighted by:
            1. the inverse of the label counts.
            2. the inverse of density of gaussian image, so that isolates from denser
                regions are not selected as often.
            3. the inverse of number of isolates on plate.
        """
        weights = np.ones(len(df))
        if self.weight_by_label:
            label_clean2count = df["label_clean_idx"].value_counts().to_dict()
            label_clean2weight = {k: 1 / v for k, v in label_clean2count.items()}
            weights *= df["label_clean_idx"].map(label_clean2weight)
        if self.weight_by_density:
            # the weight of each isolate for this part is the inverse of the density at
            # the coordinate of the isolate (round to closest integer)
            _weights = np.ones(len(df))
            for p in df.plate.unique():
                in_plate = (df["plate"] == p).to_numpy()
                df_plate = df[in_plate]
                w = gaussian_image(
                    df_plate[["coord_x_rgb", "coord_y_rgb"]].to_numpy(),
                    pois=df_plate[["coord_x_rgb", "coord_y_rgb"]].to_numpy(),
                    fwhm=self.weight_density_kernel_size,
                    trunc=0,
                    normalize=False,
                )
                _weights[in_plate] = 1 / w
            weights *= _weights
        # isolates on denser plates are less likely to be sampled, such that each sample
        # is equally likely to come from each plate.
        if self.weight_by_plate:
            plate2num_isolates = df["plate"].value_counts().to_dict()
            plate2weight = {k: 1 / v for k, v in plate2num_isolates.items()}
            weights *= df["plate"].map(plate2weight)
        df["weight"] = weights
        return df

    def _calc_training_stats(self, df: pd.DataFrame) -> None:
        pass

    def _sample_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sample training data based on epoch length, batch size, and number of devices.

        Args:
            df (pd.DataFrame): DataFrame to sample from.

        Returns:
            pd.DataFrame: Sampled DataFrame.
        """
        self.num_samples_per_epoch = (
            self.num_batches_per_epoch * self.batch_size * self.num_devices
        )
        k = (
            self.num_samples_per_epoch
            if self._k_per_sample <= 0
            else self._k_per_sample
        )
        k_max = len(df) // (self.batch_size * self.num_devices)
        if k_max > len(df):
            rprint(
                f"Number of samples per sampling ({k_max}) exceeds the length of the DataFrame "
                f"({len(df)}). Reset to the maximum possible value ({k_max})."
            )
            k = k_max
        _t, _p = divmod(self.num_samples_per_epoch, k)
        ks = [k] * _t + [_p] * (_p > 0)
        idx = [self._rng.choice(len(df), size=k, replace=False) for k in ks]
        sampled_df = df.iloc[np.concatenate(idx)].copy()
        # in pytorch ddp, samples are first split into devices and then to batches,
        # like [d1_b1, d1_b2, d1_b3, d2_b1, d2_b2, d2_b3, ...]
        sampled_df["batch_idx"] = np.tile(
            np.repeat(np.arange(self.num_batches_per_epoch), self.batch_size),
            self.num_devices,
        )
        sampled_df["device_idx"] = np.repeat(
            np.arange(self.num_devices), self.batch_size * self.num_batches_per_epoch
        )
        return sampled_df

    def _set_randomness(self, df: pd.DataFrame, split: str = "train") -> pd.DataFrame:
        """
        Add random components to the DataFrame for training.

        Args:
            df (pd.DataFrame): DataFrame to add random components to.

        Returns:
            pd.DataFrame: DataFrame with added random components.
        """
        if split == "train":
            _rng = self._rng
            p_num_igs = self.p_num_igs
            p_igs = self.p_igs
        else:
            _rng = self._rng_val
            p_num_igs = self.p_num_igs_val
            p_igs = self.p_igs_val
        idx = df.index
        # reset index here for simpler code when adding columns, will reset it back at the end.
        df = df.copy().reset_index(drop=True)
        # for idx_start in range(0, len(df), self.batch_size):
        for idx_start in range(len(df)):
            sample_info = df.loc[idx_start]
            img_groups_avail = sample_info["image_groups_available"]
            # -1 since we are using pandas loc
            # idx_end = idx_start + self.batch_size - 1
            p1 = np.array(list(p_num_igs.values()))
            num_igs = _rng.choice(list(p_num_igs.keys()), p=p1 / p1.sum())
            num_igs = min(num_igs, len(img_groups_avail))
            p2 = np.array([p_igs[ig] for ig in img_groups_avail])
            igs = _rng.choice(
                list(img_groups_avail), size=num_igs, replace=False, p=p2 / p2.sum()
            )
            df.loc[idx_start, "rand_num_igs"] = num_igs
            # cannot do df.loc[idx_start, "rand_chosen_igs"] = igs since igs is
            # an iterable and pandas forces it to have the same length as the slice, while
            # I want to copy that iterable into each row.
            df.loc[idx_start:idx_start, "rand_chosen_igs"] = pd.Series(
                [igs], index=[idx_start]
            )
            for ig in igs:
                tps = sorted(df.loc[idx_start, "image_groups_info"][ig].keys())
                # for j, tp in enumerate(tps_all, idx_start):
                j = idx_start
                if len(tps) == 1:
                    rand_tp = tps[0]
                else:
                    p_tps = [(1 - self.p_last_time_point) / (len(tps) - 1)] * (
                        len(tps) - 1
                    ) + [self.p_last_time_point]
                    rand_tp = _rng.choice(tps, p=p_tps)
                df.loc[j, f"rand_image_{ig}_tp"] = rand_tp
                if ig == "hsi":
                    # hsi is different since:
                    # 1. path is stored in a dict.
                    # 2. we also need to calc coords in hsi frame.
                    # 3. channel dropout is applied.
                    df.loc[j, f"rand_image_{ig}_path"] = df.loc[j, "image_groups_info"][
                        ig
                    ][rand_tp]["dir"]
                    transform_info = df.loc[j, "image_groups_info"][ig][rand_tp][
                        "transform"
                    ]
                    func_transform = get_query2target_func(
                        **transform_info["params"],
                        **transform_info["stats"],
                        flip=transform_info["flip"],
                    )
                    df.loc[j, ["coord_x_hsi", "coord_y_hsi"]] = func_transform(
                        df.loc[j:j, ["coord_x_rgb", "coord_y_rgb"]].to_numpy()
                    )[0]
                    # 1 are kept, 0 are dropped.
                    channel_dropout_mask = np.zeros(self.num_hsi_channels, dtype=bool)
                    channel_dropout_mask[
                        _rng.choice(
                            self.num_hsi_channels,
                            size=int(self.num_hsi_channels * self.p_hsi_channels),
                            replace=False,
                        )
                    ] = 1
                    df.loc[j:j, f"rand_channel_dropout_{ig}"] = pd.Series(
                        [channel_dropout_mask], index=[j]
                    )
                else:
                    # rgb-white or rgb-red. just store image path.
                    df.loc[j, f"rand_image_{ig}_path"] = df.loc[j, "image_groups_info"][
                        ig
                    ][rand_tp]
        df.index = idx
        return df

    # def on_epoch_start(self):
    #     """
    #     Hook to refresh the dataset and set batch indices and dropout patterns.
    #     """
    #     self.train_df = self.set_randomness(
    #         self._sample_training_data(self.train_df_all)
    #     )


def get_isolate_label(
    isolate_info: dict,
    min_counts: int = 10,
    min_counts_dominate: int = 0,
    min_ratio: int = 2,
    min_purity: float = 0.0,
    taxon_level_idx: int = -1,
) -> tuple[str, None | list[str]]:
    """Get the label of an isolate given its amplicon sequencing.
    If an isolate is empty, the label is "empty". Otherwise, if an isolate is pure, the
        label is the taxonomy at given label. Otherwise, the label is "impure".
    An isolate is empty if its total counts is less than min_counts and all zotus are #UNKNOWN
    An isolate is pure if all these are true:
        1. The most abundant zotu is not #UNKNOWN
        2. The count of most abundant zotu is at least 10.
        3. The count of most abundant zotu is at least 2 times the count of the second most abundant zotu.
    """
    counts = np.array(isolate_info["counts"])
    zotus = np.array(isolate_info["zotus"])
    taxonomies = np.array(isolate_info["taxonomies"])
    if not zotus.shape[0]:
        return "empty", None
    order = np.argsort(counts)[::-1]
    counts = counts[order]
    zotus = zotus[order]
    taxonomies = taxonomies[order]
    ret = "impure", None
    if sum(counts) < min_counts:
        if zotus[0] == "#UNKNOWN":
            ret = "empty", None
        else:
            ret = "ambiguous", None
    elif zotus[0] != "#UNKNOWN" and counts[0] >= min_counts_dominate:
        purity = counts[0] / sum(counts)
        if purity < min_purity:
            return ret
        # remove "#UNKNOWN" if it is one of the zotus and filter for ratio
        unknown_idx = np.where(zotus == "#UNKNOWN")[0]
        if unknown_idx.shape[0]:
            counts = np.delete(counts, unknown_idx)
            zotus = np.delete(zotus, unknown_idx)
        if len(counts) == 1 or counts[0] / counts[1] >= min_ratio:
            ret = (
                taxonomies[0]["taxon"][taxon_level_idx],
                taxonomies[0]["taxon"][: taxon_level_idx + 1],
            )
    return ret


def recursive_inspect(item):
    if isinstance(item, dict):
        return {k: recursive_inspect(v) for k, v in item.items()}
    elif isinstance(item, list):
        if not any(isinstance(v, (dict, list)) for v in item):
            return recursive_inspect(np.array(item))
        else:
            return [recursive_inspect(v) for v in item]
    else:
        try:
            shape = item.shape
        except AttributeError:
            shape = "N/A"

        try:
            dtype = item.dtype
        except AttributeError:
            dtype = "N/A"

        return {
            "type": str(type(item)),
            "dtype": str(dtype),
            "shape": "[" + ", ".join(map(str, shape)) + "]",
        }


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()

    # add subparsers for test and calc_stats
    # test subcommmand test data loading pipeline and time to setup data module and
    # load one sample
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_test = subparsers.add_parser("test")
    parser_calc_stats = subparsers.add_parser("calc_stats")
    parser_calc_stats.add_argument(
        "--amplicon_type",
        type=str,
        required=True,
        choices=["16s", "its"],
        help="Type of amplicon sequencing data to use.",
    )
    parser_calc_stats.add_argument(
        "--taxon_level",
        type=str,
        required=True,
        choices=["genus", "species"],
        help="Taxonomic level to use.",
    )

    args = parser.parse_args()

    if args.command == "test":
        base_dir = "/mnt/c/aws_data/data/camii"
        base_dir = "/home/ubuntu/data/camii"
        for amplicon_type, taxon_level in zip(["16s", "its"], ["genus", "species"]):
            t0 = time.perf_counter()
            dm = TAMPICDataModule(
                metadata_train_path=f"{base_dir}/all_0531.json",
                weight_by_label=True,
                weight_by_density=False,
                weight_density_kernel_size=50,
                weight_by_plate=False,
                p_num_igs={1: 0.3, 2: 0.7, 3: 0},
                p_igs={"rgb-red": 2, "rgb-white": 3, "hsi": 0},
                p_num_igs_val={1: 0.3, 2: 0.7, 3: 0},
                p_igs_val={"rgb-red": 2, "rgb-white": 3, "hsi": 0},
                p_last_time_point=0.6,
                p_hsi_channels=0.9,
                rng_seed=42,
                amplicon_type=amplicon_type,
                taxon_level=taxon_level,
                min_counts=10,
                min_counts_dominate=0,
                min_ratio=2,
                min_num_isolates=30,
                keep_empty=True,
                keep_others=True,
                crop_size_init={"rgb-red": 224, "rgb-white": 224, "hsi": 128},
                crop_size_final=96,
                target_mask_kernel_size=5,
                num_devices=2,
                batch_size=8,
                num_workers=1,
                num_batches_per_epoch=50,
                # _hsi_group_k=3,
                _hsi_crop_size=196,
                _hsi_norm=True,
                _hsi_avg_dim=100,
                _hsi_wavelengths_overwrite=[],
            )
            dm.setup()
            dl_train = dm.val_dataloader()[1]
            t1 = time.perf_counter()
            rprint(f"Time to setup data module: {t1 - t0:.2f} s")
            first_sample = next(iter(dl_train))
            t2 = time.perf_counter()
            shape = recursive_inspect(first_sample)
            rprint(json.dumps(shape, indent=4))
            rprint(f"Time to load first sample: {t2 - t1:.2f} s")
            ts = [t2]
            for i, batch in enumerate(tqdm(dl_train)):
                ts.append(time.perf_counter())
            rprint(
                f"Time to load one batch: {np.mean(np.diff(ts)):.2f} s, with batch "
                f"size = {dm.batch_size}, num_workers = {dm.num_workers}, "
                f"p_hsichannels = {dm.p_hsi_channels}"
            )
    elif args.command == "calc_stats":
        amplicon_type = args.amplicon_type
        taxon_level = args.taxon_level
        # in this subcommand, we calculate the mean and std of each channel.
        # No data augmentation or image group dropout or hsi channel dropout is applied
        # in this subcommand. All samples have the same sampling weight. Empty
        # isolates are not kept. All other configurations remain tha same, such
        # as time point selection and crop size.
        base_dir = "/mnt/c/aws_data/data/camii"
        base_dir = "/home/ubuntu/data/camii"
        num_channels_rgb_red = 3
        num_channels_rgb_white = 3
        num_channels_hsi = 462
        # Initialize variables for mean and std calculations for each channel
        rgb_red_sum = torch.zeros(num_channels_rgb_red)
        rgb_red_sum_sq = torch.zeros(num_channels_rgb_red)
        rgb_white_sum = torch.zeros(num_channels_rgb_white)
        rgb_white_sum_sq = torch.zeros(num_channels_rgb_white)
        hsi_sum = torch.zeros(num_channels_hsi)
        hsi_sum_sq = torch.zeros(num_channels_hsi)
        n_rgb_red = torch.zeros(num_channels_rgb_red)
        n_rgb_white = torch.zeros(num_channels_rgb_white)
        n_hsi = torch.zeros(num_channels_hsi)
        L.seed_everything(2024)
        rng_seed = 42
        dm = TAMPICDataModule(
            metadata_train_path=f"{base_dir}/all_0531.json",
            weight_by_label=False,
            weight_by_density=False,
            weight_density_kernel_size=None,
            weight_by_plate=False,
            p_num_igs={1: 0, 2: 0, 3: 1},
            # p_igs=None,
            p_last_time_point=0.6,
            p_hsi_channels=1,
            rng_seed=rng_seed,
            amplicon_type=amplicon_type,
            taxon_level=taxon_level,
            min_counts=10,
            min_counts_dominate=0,
            min_purity=0.1,
            min_ratio=2,
            min_num_isolates=30,
            keep_empty=False,
            keep_others=True,
            crop_size_init={"rgb-red": 224, "rgb-white": 224, "hsi": 128},
            crop_size_final={"rgb-red": 224, "rgb-white": 224, "hsi": 128},
            target_mask_kernel_size=5,
            num_devices=1,
            batch_size=8,
            num_workers=12,
            num_batches_per_epoch=-1,
            # _hsi_group_k=3,
            _hsi_crop_size=196,
            _hsi_norm=True,
            _calc_stats_mode=True,
        )
        dm.setup()
        dl_train = dm.train_dataloader()

        # Iterate over the batches
        for batch in tqdm(dl_train):
            rgb_red_images = batch["data"]["rgb-red"]["image"]
            rgb_white_images = batch["data"]["rgb-white"]["image"]
            hsi_images = batch["data"]["hsi"]["image"]
            rgb_red_available = batch["data"]["rgb-red"]["available"]
            rgb_white_available = batch["data"]["rgb-white"]["available"]
            hsi_available = batch["data"]["hsi"]["available"]
            rgb_red_dropped = batch["data"]["rgb-red"]["dropped"]
            rgb_white_dropped = batch["data"]["rgb-white"]["dropped"]
            hsi_dropped = batch["data"]["hsi"]["dropped"]

            for i in range(len(rgb_red_images)):  # loop over batch size
                if rgb_red_available[i] and not rgb_red_dropped[i]:
                    rgb_red_img = rgb_red_images[i]
                    rgb_red_sum += rgb_red_img.sum(dim=(1, 2))
                    rgb_red_sum_sq += (rgb_red_img**2).sum(dim=(1, 2))
                    n_rgb_red += rgb_red_img[0].numel()

                if rgb_white_available[i] and not rgb_white_dropped[i]:
                    rgb_white_img = rgb_white_images[i]
                    rgb_white_sum += rgb_white_img.sum(dim=(1, 2))
                    rgb_white_sum_sq += (rgb_white_img**2).sum(dim=(1, 2))
                    n_rgb_white += rgb_white_img[0].numel()

                if hsi_available[i] and not hsi_dropped[i]:
                    hsi_img = hsi_images[i]
                    hsi_sum += hsi_img.sum(dim=(1, 2))
                    hsi_sum_sq += (hsi_img**2).sum(dim=(1, 2))
                    n_hsi += hsi_img[0].numel()

        # Calculate mean and standard deviation
        rgb_red_mean = rgb_red_sum / n_rgb_red
        rgb_red_std = torch.sqrt((rgb_red_sum_sq / n_rgb_red) - (rgb_red_mean**2))

        rgb_white_mean = rgb_white_sum / n_rgb_white
        rgb_white_std = torch.sqrt(
            (rgb_white_sum_sq / n_rgb_white) - (rgb_white_mean**2)
        )

        hsi_mean = hsi_sum / n_hsi
        hsi_std = torch.sqrt((hsi_sum_sq / n_hsi) - (hsi_mean**2))
