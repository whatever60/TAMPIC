import warnings

import numpy as np
import cv2 as cv
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from rich import print as rprint



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
