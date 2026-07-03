from typing import Literal

import numpy as np
import torch
from scipy.integrate import dblquad


def closest_perfect_square(
    x: int | float, return_type: Literal["prev", "next", "nearest"] = "nearest"
) -> int:
    floor_sqrt = np.floor(np.sqrt(x))

    if np.sqrt(x) - floor_sqrt == 0:
        return floor_sqrt.astype(int) ** 2
    prev_sq = (floor_sqrt) ** 2
    next_sq = (floor_sqrt + 1) ** 2
    if return_type == "prev":
        return prev_sq
    elif return_type == "next":
        return next_sq
    else:
        return next_sq.astype(int) if next_sq - x < x - prev_sq else prev_sq.astype(int)


def torch_polar_to_cartesian_2d(r, theta):
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return x, y


def np_polar_to_cartesian_2d(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def trapezoid_integration(lmbd_fn):
    integration_report = dblquad(
        lambda x, y: lmbd_fn(torch.tensor([x, y], dtype=torch.float32)).numpy(),
        a=0,
        b=1,
        gfun=0,
        hfun=1,
    )
    return integration_report[0], integration_report[1]
