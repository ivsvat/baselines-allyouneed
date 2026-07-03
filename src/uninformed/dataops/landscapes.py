import torch
import numpy as np
from torch.nn import functional as F
from abc import ABC, abstractmethod
from typing import Literal
from uninformed.utils.misc import np_polar_to_cartesian_2d


class Landscape(ABC):
    @property
    @abstractmethod
    def integral(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def maximum_on_01(self):
        raise NotImplementedError

    @abstractmethod
    def lmbd_fn(self, xy: torch.Tensor):
        raise NotImplementedError


class StepLandscape(Landscape):
    r"""Spatial density is a step function across x-axis on [0, 1]^2"""

    def __init__(self, step_x: float = 0.5, step_delta: float = 1) -> None:
        self.step_x = step_x
        self._maximum_on_01 = 2
        self.step_delta = step_delta
        self._integral = 1 + step_delta * step_x

    @property
    def integral(self):
        return self._integral

    @property
    def maximum_on_01(self):
        return self._maximum_on_01

    def lmbd_fn(self, xy: torch.Tensor):
        if xy.dim() == 1:
            xy = xy.unsqueeze(0)

        res = 1 + self.step_delta * (xy[:, 0] < self.step_x).to(torch.float32)
        return res


class SlopeLandscape(Landscape):
    r"""Spatial density decays linearly across x-axis on [0, 1]^2

    f(x; (k, b)) = 1 + max(b-kx, 0)
    """

    def __init__(self, k: float = 1.0, b: float = 1.0) -> None:
        self.k = k
        self._maximum_on_01 = 1 + b
        self.b = b
        self._integral = 1 + b * min(b / k, 1) - 0.5 * k * min(b / k, 1) ** 2

    @property
    def integral(self):
        return self._integral

    @property
    def maximum_on_01(self):
        return self._maximum_on_01

    def lmbd_fn(self, xy: torch.Tensor):
        if xy.dim() == 1:
            xy = xy.unsqueeze(0)

        res = 1 + F.relu(self.b - self.k * xy[:, 0])
        return res


class UniformLandscape(Landscape):
    r"""Spatial density is uniform on [0, 1]^2"""

    def __init__(self) -> None:
        self._integral = 1.0
        self._maximum_on_01 = 1.0

    @property
    def integral(self):
        return self._integral

    @property
    def maximum_on_01(self):
        return self._maximum_on_01

    def lmbd_fn(self, xy: torch.Tensor):
        if xy.dim() == 1:
            xy = xy.unsqueeze(0)

        res = torch.ones_like(xy[:, 0])
        return res


class EmbossedDiscsLandscape(Landscape):
    r"""Spatial density varies at several evenly spaced round regions on [0, 1]^2"""

    def __init__(
        self,
        n_discs: int | None = None,
        base_radius: float = 0.1,
        density_delta: float = 1.0,
        mode: Literal["emboss", "deboss"] = "emboss",
        randomise_single_disc_position: bool = False,
    ) -> None:
        # rough estimate so that circles do not overlap with centroids being at 0.25 from (0.5, 0.5)
        max_n_discs = np.floor(
            2 * np.pi / (np.arctan(base_radius / (0.25 - base_radius)))
        ).astype(int)
        if n_discs is None:
            self.n_discs = np.random.randint(low=1, high=max_n_discs, size=(1,)).astype(
                int
            )
        else:
            n_discs = min(max_n_discs, n_discs)
            self.n_discs = n_discs if n_discs is not None else 1

        self.radii = torch.tensor([base_radius] * self.n_discs, dtype=torch.float32)
        self.density_delta = density_delta
        if self.n_discs == 1:
            self.ab = torch.tensor([[0.5, 0.5]])

        else:
            self.ab = torch.tensor(
                np.array(
                    [
                        np_polar_to_cartesian_2d(
                            r=0.25,
                            theta=2 * np.pi / self.n_discs * i,
                        )
                        for i in range(self.n_discs)
                    ]
                )
            )
            self.ab += 0.5

        self.mode = mode
        self.randomise_single_disc_position = randomise_single_disc_position and (
            self.n_discs == 1
        )
        if self.mode == "emboss":
            self._integral = (
                1 + density_delta * (self.radii**2 * torch.pi).sum()
            ).item()
            self._maximum_on_01 = 1 + self.density_delta
        elif self.mode == "deboss":
            self._integral = (1 - (self.radii**2 * torch.pi).sum()).item()
            self._maximum_on_01 = 1.0

    @property
    def integral(self):
        return self._integral

    @property
    def maximum_on_01(self):
        return self._maximum_on_01

    def lmbd_fn(self, xy: torch.Tensor, generator: torch.Generator | None = None):
        if xy.dim() == 1:
            xy = xy.unsqueeze(0)
        if self.randomise_single_disc_position:
            ab = (
                torch.rand(
                    self.ab.shape,
                    generator=generator,
                    device=self.ab.device,
                    dtype=self.ab.dtype,
                )
                * (1 - self.radii * 2)
                + self.radii
            )
        else:
            ab = self.ab

        a_s = ab[:, 0]
        b_s = ab[:, 1]
        r_s = self.radii

        conditional_term = torch.stack(
            [
                (xy[:, 0] - a_s[i]) ** 2 + (xy[:, 1] - b_s[i]) ** 2 <= r_s[i] ** 2
                for i in range(self.n_discs)
            ]
        ).permute(1, 0)  # n_samples, n_discs
        conditional_term = conditional_term.sum(-1)
        conditional_term = (conditional_term > 0).to(torch.float32)
        if self.mode == "emboss":
            res = 1 + self.density_delta * conditional_term
        else:
            res = 1 - conditional_term

        return res
