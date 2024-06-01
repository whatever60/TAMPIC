from typing import Callable

import numpy as np
import torch

def affine_transform(params: torch.Tensor, keypoints: torch.Tensor):
    """Apply affine transformation to keypoints. The transformation consists of:
    - rotation and scaling around origin (hence order doesn't matter)
    - translation
    """
    a, b, tx, ty, sx, sy = params
    norm = torch.sqrt(a**2 + b**2)
    sin_t, cos_t = b / norm, a / norm
    # scaling + rotation, then translation
    # rotation_matrix = torch.stack(
    #     [torch.stack([cos_t, -sin_t]), torch.stack([sin_t, cos_t])], dim=0
    # )
    # scaling_matrix = np.diag(torch.stack([sx, sy]))
    # transformed_keypoints = keypoints @ rotation_matrix @ scaling_matrix + torch.stack(
    #     [tx, ty]
    # )
    transformed_keypoints = keypoints @ torch.stack(
        [torch.stack([sx * cos_t, -sy * sin_t]), torch.stack([sx * sin_t, sy * cos_t])],
        dim=0,
    ) + torch.stack([tx, ty])
    # translation, rotation, then scaling
    # transformed_keypoints = (
    #     (keypoints + torch.stack([tx, ty]))
    #     @ torch.stack(
    #         [torch.stack([cos_t, -sin_t]), torch.stack([sin_t, cos_t])],
    #         dim=0,
    #     )
    #     * torch.tensor([sx, sy])
    # )
    return transformed_keypoints


def get_query2target_func(
    a: torch.Tensor,
    b: torch.Tensor,
    tx: torch.Tensor,
    ty: torch.Tensor,
    sx: torch.Tensor,
    sy: torch.Tensor,
    mean_q: torch.Tensor,
    std_q: torch.Tensor,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    flip: tuple[bool, bool] = (False, False),
) -> Callable:
    def ret(query: np.ndarray) -> np.ndarray:
        query_std = torch.from_numpy((query - mean_q) / std_q).float()
        query_mapped_std = affine_transform(
            torch.tensor([a, b, tx, ty, sx, sy]), flip_tensor(query_std, *flip)
        )
        mapped_std = query_mapped_std * torch.tensor(std_t) + torch.tensor(mean_t)
        return mapped_std.numpy()

    return ret


def flip_tensor(t: torch.Tensor, flip_h: bool, flip_v: bool) -> torch.Tensor:
    t = t.clone()
    if flip_h:
        t[:, 0] *= -1
    if flip_v:
        t[:, 1] *= -1
    return t
