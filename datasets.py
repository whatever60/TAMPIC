import json
import os
from typing import Optional

import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm.auto import tqdm

from affine_transform import get_query2target_func


class GroupTransform:
    def __init__(
        self,
        channel_stats: dict[str, dict[str, float]],
        crop_size: int | dict[str, int],
    ):
        self.crop_size = crop_size
        if isinstance(crop_size, int):
            crop_size = {"rgb-red": crop_size, "rgb-white": crop_size, "hsi": crop_size}

        # Geometric transformations applied per modality
        self.geom_transforms = {
            "rgb-red": A.Compose(
                [
                    A.RandomCrop(width=crop_size, height=crop_size, p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=30, p=0.5),
                ],
                additional_targets={"target_mask": "image"},
            ),
            "rgb-white": A.Compose(
                [
                    A.RandomCrop(width=crop_size, height=crop_size, p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=30, p=0.5),
                ],
                additional_targets={"target_mask": "image"},
            ),
            "hsi": A.Compose(
                [
                    A.RandomCrop(width=crop_size, height=crop_size, p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=30, p=0.5),
                ],
                additional_targets={"target_mask": "image"},
            ),
        }

        # Color transformations are specific to each modality
        self.color_transforms = {
            "rgb-red": A.Compose(
                [
                    A.ColorJitter(
                        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.75
                    ),
                    A.Normalize(
                        mean=channel_stats["rgb-red"]["mean"],
                        std=channel_stats["rgb-red"]["std"],
                    ),
                    ToTensorV2(),
                ]
            ),
            "rgb-white": A.Compose(
                [
                    A.ColorJitter(
                        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.75
                    ),
                    A.Normalize(
                        mean=channel_stats["rgb-white"]["mean"],
                        std=channel_stats["rgb-white"]["std"],
                    ),
                    ToTensorV2(),
                ]
            ),
            "hsi": A.Compose(
                [
                    A.Normalize(
                        mean=channel_stats["hsi"]["mean"],
                        std=channel_stats["hsi"]["std"],
                    ),
                    ToTensorV2(),
                ]
            ),
        }

    def __call__(
        self,
        images: dict[str, np.ndarray],
        target_masks: dict[str, np.ndarray],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Apply geometric and color transformations to the images.

        Args:
            images (dict[str, np.ndarray]): Dictionary of images for each modality group.
            target_masks (dict[str, np.ndarray]): Dictionary of target masks (Gaussian images) for each modality group.

        Returns:
            dict[str, np.ndarray]: Transformed images for each modality group.
        """
        transformed_images = {}
        transformed_target_masks = {}

        for modality, img in images.items():
            # Combine image and target mask
            data = {"image": img, "target_mask": target_masks[modality]}
            augmented = self.geom_transforms[modality](**data)

            # Apply color transformation to the image only
            color_transformed_image = self.color_transforms[modality](
                image=augmented["image"]
            )["image"]

            transformed_images[modality] = color_transformed_image
            transformed_target_masks[modality] = augmented["target_mask"]

        return transformed_images, transformed_target_masks


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
        self.crop_size_init = crop_size_init
        self.crop_size_final = crop_size_final
        self.hsi_wavelengths = hsi_wavelengths

        self.transform = GroupTransform(
            channel_stats=channel_stats, crop_size=crop_size_final
        )

    def __len__(self) -> int:
        """
        Get the total number of images in the dataset.

        Returns:
            int: Total number of images.
        """
        return len(self.df)

    def __getitem__(
        self, idx: int
    ) -> dict[dict[str, torch.Tensor], dict[str, torch.Tensor], int]:
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
        coords_rgb = row[["coord_x_rgb", "coord_y_rgb"]].to_numpy()
        coords_hsi = row[["coord_x_hsi", "coord_y_hsi"]].to_numpy()
        images = {}
        target_masks = {}

        for ig in row["rand_chosen_igs"]:
            # Both groups of rgb-red and rgb-white only have one image and therefore
            # no image dropout is needed.
            if ig == "hsi":
                # 0 means a channel is dropped out, 1 means kept.
                hsi_channel_dropout = row["rand_channel_dropout_hsi"]
                # For efficiency in cropping and augmentation, we stack all the images
                # that are not dropped in HSI group.
                img = self._crop_image(
                    np.dstack(
                        [
                            np.array(
                                Image.open(
                                    os.path.join(
                                        row["rand_image_hsi_path"], f"{wl}.png"
                                    )
                                ).convert("I")
                            )
                            for wl in range(self.hsi_wavelengths[hsi_channel_dropout])
                        ]
                    ),
                    coords_hsi,
                )
            else:
                img = self._crop_image(
                    np.array(Image.open(row[f"rand_image_{ig}_path"]).convert("RGB")),
                    coords_rgb,
                )
                target_mask = gaussian_image(coords_rgb)
            if isinstance(self.crop_size_init, dict):
                crop_size_init = self.crop_size_init[ig]
            else:
                crop_size_init = self.crop_size_init
            target_mask = gaussian_image(
                position=(
                    crop_size_init / 2,
                    crop_size_init / 2,
                ),
                shape=(crop_size_init, crop_size_init),
                fwhm=self.target_mask_kernel_size,
            )
            images[ig] = img
            target_masks[ig] = target_mask

        imgs_aug, target_masks_aug = self.transform(images, target_masks)

        # post processing for HSI group
        if "hsi" in imgs_aug:
            if isinstance(self.crop_size_final, dict):
                crop_size_final = self.crop_size_final["hsi"]
            else:
                crop_size_final = self.crop_size_final
            imgs_aug_hsi = torch.zeros(
                (self.hsi_wavelengths, crop_size_final, crop_size_final)
            )
            imgs_aug_hsi[hsi_channel_dropout] += imgs_aug["hsi"]
            imgs_aug["hsi"] = imgs_aug_hsi

        return {
            "images": imgs_aug,
            "target_masks": target_masks_aug,
            "label": row["label"],
        }

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
    shape: np.ndarray = None,
    pois: np.ndarray = None,
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


class TampicDataModule(pl.LightningDataModule):
    # selecting only 1 image group is the most likely.
    default_p_num_igs = {1: 2, 2: 1, 3: 1}
    # selecting rgb-white is the most likely, followed by rgb-red, and hsi.
    default_p_igs = {"rgb-red": 2, "rgb-white": 3, "hsi": 1}

    def __init__(
        self,
        *,
        metadata_train_path: dict,
        val_easy_ratio: float = 0.3,
        # metadata_val_easy_path: dict,
        # metadata_val_mid_path: dict,
        #
        weight_by_label: bool = True,
        weight_by_density: bool = False,
        weight_density_kernel_size: int = 5,
        weight_by_plate: bool = False,
        p_num_igs: Optional[list[int]] = None,
        p_igs: Optional[list[int]] = None,
        p_last_time_point: Optional[list[int]] = 0.5,
        p_hsi_channels=0.3,
        rng_seed: Optional[int] = 42,
        #
        amplicon_type: str = None,
        taxon_level: str = "genus",
        min_counts: int = 10,
        min_counts_dominate: int = 0,
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
        num_batches_per_epoch: Optional[int] = None,
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

        # arguments for getting labels
        self.amplicon_type = amplicon_type
        self.taxon_level = taxon_level
        self.min_counts = min_counts
        self.min_counts_dominate = min_counts_dominate
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
        # self.train_df = None
        # self.val_df_1 = None
        # self.val_df_2 = None

    def setup(self, stage: Optional[str] = None):
        """
        Set up the datasets for training and validation.

        Args:
            stage (Optional[str]): Stage for which to set up the data. Defaults to None.
        """
        if stage in (None, "fit"):
            df_all = self._json_to_dataframe(self.metadata_train_path)
            df_train = df_all.query("split == 'train'").copy()
            self._fit_labels(df_train)
            df_train = self._transform_labels(df_train)
            self.df_train_all = self._add_weight(df_train)

            df_val_easy = df_all.query("split == 'val_easy'").copy()
            df_val_mid = df_all.query("split == 'val_mid'").copy()
            self.df_val_easy = self._transform_labels(df_val_easy)
            self.df_val_mid = self._transform_labels(df_val_mid)
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
            target_mask_kernel_size=self.target_mask_kernel_size,
            crop_size_init=self.crop_size_init,
            crop_size_final=self.crop_size_final,
            num_hsi_channels=self.num_hsi_channels,
        )

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def val_dataloader(self):
        """
        Get the validation dataloaders.

        Returns:
            list[DataLoader]: list of validation dataloaders.
        """
        val_dataset_1 = ImageDataset(
            df=self.df_val_easy,
            target_mask_kernel_size=self.target_mask_kernel_size,
            crop_size_init=self.crop_size_init,
            crop_size_final=self.crop_size_final,
            num_hsi_channels=self.num_hsi_channels,
        )

        val_dataset_2 = ImageDataset(
            df=self.df_val_mid,
            target_mask_kernel_size=self.target_mask_kernel_size,
            crop_size_init=self.crop_size_init,
            crop_size_final=self.crop_size_final,
            num_hsi_channels=self.num_hsi_channels,
        )

        return [
            DataLoader(
                val_dataset_1,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
            ),
            DataLoader(
                val_dataset_2,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
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
            split (str): Split type ('train' or 'val').

        Returns:
            pd.DataFrame: DataFrame with precomputed project, plate, and image information.
        """
        metadata_dir = os.path.dirname(metadata_path)
        with open(metadata_path) as f:
            metadata = json.load(f)
        # load global properties
        global_properties = metadata["global_properties"]
        self.num_hsi_channels = global_properties["hsi_num_channels"]
        self.hsi_wavelengths = np.loadtxt(
            os.path.join(metadata_dir, global_properties["hsi_wavelengths"])
        )
        self.taxon_levels = np.array(global_properties["taxonomy_levels"])
        self.taxon_level_idx = np.where(self.taxon_levels == self.taxon_level)[0][0]
        self.stats = global_properties["stats"]  # for normalization

        # load information of each isolate (project id, plate id isolate id, available
        # modalities, available time point for each modality, path to all of the images
        # in dict, label)
        dfs = []
        for proj, meta_proj in metadata["data"].items():
            if meta_proj["status"] != "valid":
                continue
            with open(f"{metadata_dir}/{proj}/{meta_proj['amplicon_info']}") as f:
                amplicon_info = json.load(f)
            for plate, meta_plate in meta_proj["plates"].items():
                rows = []
                if meta_plate["status"] != "valid":
                    continue
                sample_type = meta_plate["sample_type"]
                medium_type = meta_plate["medium_type"]
                num_isolates = len(amplicon_info[plate])
                image_groups_info = {}
                if "rgb" in meta_plate["images"]:
                    time_points_rgb = {
                        int(k[1:]): {"red": v["red"], "white": v["white"]}
                        for k, v in meta_plate["images"]["rgb"].items()
                        if k != "default" and v["status"] == "valid"
                    }
                    image_groups_info["rgb-red"] = {
                        k: os.path.join(metadata_dir, proj, v["red"]) for k, v in time_points_rgb.items()
                    }
                    image_groups_info["rgb-white"] = {
                        k: os.path.join(metadata_dir, proj, v["white"]) for k, v in time_points_rgb.items()
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
                            "dir": os.path.join(metadata_dir, proj, v["dir"]),
                            "transform": v["transform"]["from_rgb"],
                        }
                        for k, v in meta_plate["images"]["hsi"].items()
                        if k != "default" and v["status"] == "valid"
                    }
                    image_groups_info["hsi"] = time_points_hsi
                for isolate, meta_isolate in amplicon_info[plate].items():
                    label = get_isolate_label(
                        meta_isolate[self.amplicon_type],
                        self.min_counts,
                        self.min_counts_dominate,
                        self.min_ratio,
                        self.taxon_level_idx,
                    )
                    rows.append(
                        {
                            "project": proj,
                            "plate": plate,
                            "sample_type": sample_type,
                            "medium_type": medium_type,
                            "num_isolates": num_isolates,
                            # "transform_iso2rgb": meta_plate["transform"][
                            #     "isolate_to_rgb"
                            # ],
                            # "transform_rgb2hsi": meta_plate["transform"]["rgb_to_hsi"],
                            "image_groups_info": image_groups_info,
                            "image_groups_available": [
                                k for k, v in image_groups_info.items() if v
                            ],
                            # "available_time_points_rgb-red": available_time_points_rgb-red,
                            # "available_time_points_rgb-white": available_time_points_rgb-white,
                            # "available_time_points_hsi": available_time_points_hsi,
                            "isolate": isolate,
                            "coord_x": meta_isolate["coord"][0],
                            "coord_y": meta_isolate["coord"][1],
                            "label": label,
                            "split": meta_plate["split"][self.amplicon_type],
                        }
                    )
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
                dfs.append(df_plate)
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

        df = pd.concat(dfs).reset_index(drop=True)
        df["split"] = df["split"].replace("val", "val_mid")
        rng = np.random.default_rng(random_split_seed)
        df.loc[
            rng.choice(
                df.query("split == 'train'").index,
                size=int(len(df) * self.val_easy_ratio),
                replace=False,
            ),
            "split",
        ] = "val_easy"
        return df

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
        label2idx = {}
        label_clean2idx = {}
        idx2label_clean = {}

        label_counts = df["label"].value_counts().to_dict()
        iter_label_counts = iter(label_counts.items())
        for i, (label, count) in enumerate(iter_label_counts):
            if count >= self.min_num_isolates:
                if self.keep_empty or label != "empty":
                    label2idx[label] = i
                    label_clean2idx[label] = i
                    idx2label_clean[i] = label
            else:
                break
        if self.keep_others:  # keep iterating
            i += 1
            label_clean2idx["others"] = i
            idx2label_clean[i] = "others"
            for i, (label, count) in enumerate(iter_label_counts, start=i + 1):
                label2idx[label] = i

        self.label2idx = label2idx
        self.label_clean2idx = label_clean2idx
        self.idx2label_clean = idx2label_clean

    def _transform_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the labels in the DataFrame to numerical values."""
        df["label_clean_idx"] = df["label"].apply(
            lambda x: self.label_clean2idx.get(x, self.label_clean2idx["others"])
        )
        if not self.keep_others:
            df = df.query("label_clean_idx != 'others'")
        if not self.keep_empty:
            df = df.query("label_clean_idx != 'empty'")
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

    def _sample_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sample training data based on epoch length, batch size, and number of devices.

        Args:
            df (pd.DataFrame): DataFrame to sample from.

        Returns:
            pd.DataFrame: Sampled DataFrame.
        """
        self.epoch_length = (
            self.num_batches_per_epoch * self.batch_size * self.num_devices
        )
        if self.epoch_length > len(df):
            raise ValueError(
                f"Epoch length ({self.epoch_length}) exceeds the length of the DataFrame "
                f"({len(df)}). This is not supported."
            )
        sampled_df = df.sample(
            n=self.epoch_length,
            replace=False,
            weights=df["weight"],
            random_state=self._rng,
        )
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

    def _set_randomness(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add random components to the DataFrame for training.

        Args:
            df (pd.DataFrame): DataFrame to add random components to.

        Returns:
            pd.DataFrame: DataFrame with added random components.
        """
        idx = df.index
        df = df.copy().reset_index(drop=True)
        # for idx_start in range(0, len(df), self.batch_size):
        for idx_start in range(len(df)):
            sample_info = df.loc[idx_start]
            img_groups_avail = sample_info["image_groups_available"]
            # -1 since we are using pandas loc
            # idx_end = idx_start + self.batch_size - 1
            p1 = np.array(list(self.p_num_igs.values()))
            num_igs = self._rng.choice(list(self.p_num_igs.keys()), p=p1 / p1.sum())
            num_igs = min(num_igs, len(img_groups_avail))
            p2 = np.array([self.p_igs[ig] for ig in img_groups_avail])
            igs = self._rng.choice(
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
                tps_all = sorted(df.loc[idx_start, "image_groups_info"][ig].keys())
                # for j, tp in enumerate(tps_all, idx_start):
                j = idx_start
                tp = tps_all
                if len(tp) == 1:
                    rand_tp = tp[0]
                else:
                    p_tps = [(1 - self.p_last_time_point) / (len(tp) - 1)] * (
                        len(tp) - 1
                    ) + [self.p_last_time_point]
                    rand_tp = self._rng.choice(tp, p=p_tps)
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
                    df.loc[j, ["rand_coord_x_hsi", "rand_coord_y_hsi"]] = (
                        func_transform(
                            df.loc[j:j, ["coord_x_rgb", "coord_y_rgb"]].to_numpy()
                        )[0]
                    )
                    # 1 are kept, 0 are dropped.
                    channel_dropout_mask = np.zeros(self.num_hsi_channels, dtype=bool)
                    channel_dropout_mask[
                        self._rng.choice(
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

    def on_epoch_start(self):
        """
        Hook to refresh the dataset and set batch indices and dropout patterns.
        """
        self.train_df = self.set_randomness(
            self._sample_training_data(self.train_df_all)
        )


def get_isolate_label(
    isolate_info: dict,
    min_counts: int = 10,
    min_counts_dominate: int = 0,
    min_ratio: int = 2,
    taxon_level_idx: int = -1,
) -> str:
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
        return "empty"
    order = np.argsort(counts)[::-1]
    counts = counts[order]
    zotus = zotus[order]
    taxonomies = taxonomies[order]
    ret = "impure"
    if sum(counts) < min_counts:
        if zotus[0] == "#UNKNOWN":
            ret = "empty"
        else:
            ret = "ambiguous"
    elif zotus[0] != "#UNKNOWN" and counts[0] >= min_counts_dominate:
        # remove "#UNKNOWN" if it is one of the zotus
        unknown_idx = np.where(zotus == "#UNKNOWN")[0]
        if unknown_idx.shape[0]:
            counts = np.delete(counts, unknown_idx)
            zotus = np.delete(zotus, unknown_idx)
        if len(counts) == 1 or counts[0] / counts[1] >= min_ratio:
            ret = taxonomies[0]["taxon"][taxon_level_idx]
    return ret


if __name__ == "__main__":
    base_dir = "/mnt/c/aws_data/data/camii"
    for amplicon_type, taxon_level in zip(["16s", "its"], ["genus", "species"]):
        dm = TampicDataModule(
            metadata_train_path=f"{base_dir}/all_0531.json",
            weight_by_label=True,
            weight_by_density=False,
            weight_density_kernel_size=50,
            weight_by_plate=False,
            # p_num_igs=None,
            # p_igs=None,
            p_last_time_point=0.6,
            p_hsi_channels=0.3,
            rng_seed=42,
            amplicon_type="16s",
            taxon_level=taxon_level,
            min_counts=10,
            min_counts_dominate=0,
            min_ratio=2,
            min_num_isolates=30,
            keep_empty=True,
            keep_others=True,
            crop_size_init=128,
            crop_size_final=128,
            target_mask_kernel_size=5,
            num_devices=2,
            batch_size=8,
            num_workers=4,
            num_batches_per_epoch=50,
        )
        dm.setup()
        dataset_train = dm.train_dataloader()
        sample = dataset_train[0]

        import pdb

        pdb.set_trace()
