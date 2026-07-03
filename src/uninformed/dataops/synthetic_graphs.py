"""Synthetic graph generators"""

# TODO: create a shared public interface e.g. init with generator sample with sample()
import torch
from typing import Callable

from sklearn import datasets
import numpy as np
from typing import Literal

from uninformed.utils.misc import closest_perfect_square


class SklearnGraphMaker:
    def __init__(self, dataset: str, **dataset_config) -> None:
        self.dataset = dataset
        self.dataset_config = dataset_config

    def sample(self, n_samples) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        synth_sample = getattr(datasets, self.dataset)(
            n_samples=n_samples, **self.dataset_config
        )

        if isinstance(synth_sample, tuple):
            _sample = synth_sample[0]
            _labels = synth_sample[1]
            return torch.tensor(_sample, dtype=torch.float32), torch.tensor(
                _labels, dtype=torch.long
            )
        else:
            return torch.tensor(synth_sample, dtype=torch.float32)


class PoissonGraphMaker:
    r"""A rejection-sampling based implementation of a non-homogeneous 2d poisson process"""

    def __init__(
        self,
        lmbd: int | float,
        lmbd_star: float | None = None,
        spatial_density: Callable[[torch.Tensor], torch.Tensor] | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        self.lmbd = lmbd
        self.generator = generator
        self.lmbd_fn = spatial_density
        self.lmbd_star = 1.0 if lmbd_star is None else lmbd_star

    def thin_samples(
        self, points: torch.Tensor, lmbd_star: float | None = None
    ) -> torch.Tensor:
        if self.lmbd_fn is None:
            return points
        if lmbd_star is None:
            lmbd_star = 1.0

        rejection_probs = self.lmbd_fn(points) / lmbd_star
        u = torch.rand(size=(points.shape[0],), generator=self.generator)
        return points[u < rejection_probs]

    def sample(
        self,
    ) -> torch.Tensor:
        n_samples = int(
            torch.poisson(
                input=torch.tensor([self.lmbd], dtype=torch.float32),
                generator=self.generator,
            ).item()
        )
        uniform_samples = torch.rand(size=(n_samples, 2), generator=self.generator)
        thinned_samples = self.thin_samples(uniform_samples, lmbd_star=self.lmbd_star)
        return thinned_samples


class NoisyGridGraphMaker:
    r"""Offset-based implementation of a non-homogeneous noisy grid sampling"""

    def __init__(
        self,
        spatial_density,
        n_voxels: int = 10,
        noise_sigma: float = 0.01,
        generator: torch.Generator | None = None,
    ):
        self.spatial_density = spatial_density
        self.voxel_size = 1 / n_voxels
        self.n_voxels = n_voxels
        xs = torch.linspace(0 + self.voxel_size / 2, 1 - self.voxel_size / 2, n_voxels)
        ys = torch.linspace(0 + self.voxel_size / 2, 1 - self.voxel_size / 2, n_voxels)
        basegrid_x, basegrid_y = torch.meshgrid(xs, ys, indexing="xy")
        self.basegrid_coords_xy = torch.stack(
            [basegrid_x.flatten(), basegrid_y.flatten()], dim=-1
        )
        self.spatial_densities_basegrid = self.spatial_density(self.basegrid_coords_xy)

        self.noise_sigma = noise_sigma
        self.generator = generator

    def sample(
        self,
        mode: Literal["budget", "upsample", "spatially_varying_noise"] = "upsample",
        **kwargs,
    ) -> torch.Tensor:
        if mode == "budget":
            return self._sample_w_budget(**kwargs)
        elif mode == "upsample":
            return self._sample_w_upsampling(**kwargs)
        elif mode == "spatially_varying_noise":
            return self._sample_spatially_varying_noise(**kwargs)

    def _sample_spatially_varying_noise(self, upsample_by: int) -> torch.Tensor:
        r"""given a fixed square grid variance of added noise changes spatially according to spatial density"""
        voxel_densities = self.spatial_densities_basegrid

        n_upsample_voxel = torch.tensor([upsample_by] * len(voxel_densities))
        n_samples_voxel = n_upsample_voxel**2

        repeated_grid = torch.repeat_interleave(
            self.basegrid_coords_xy, n_samples_voxel, dim=0
        )

        step = self.voxel_size

        if upsample_by == 1:
            offsets = torch.tensor([[0.0, 0.0]])
        else:
            # Local coordinates from -step/2 to step/2
            bound = (step / 2) * (1 - 1 / upsample_by)
            lin = torch.linspace(-bound, bound, upsample_by)
            oy, ox = torch.meshgrid(lin, lin, indexing="ij")
            offsets = torch.stack([ox.flatten(), oy.flatten()], dim=-1)

        offsets = torch.cat([offsets for i in range(len(n_upsample_voxel))])

        if self.noise_sigma > 0:
            noise_scales = self.noise_sigma * voxel_densities.unsqueeze(1)
            noise_scales = noise_scales.repeat_interleave(n_samples_voxel, dim=0)
            # print(torch.unique(noise_scales))
            noise = noise_scales * torch.randn(
                repeated_grid.shape, generator=self.generator, dtype=repeated_grid.dtype
            )

            offsets += noise

        adaptive_grid = repeated_grid + offsets

        return adaptive_grid

    def _sample_w_budget(
        self, n_samples: int, use_sample_size_heuristic: bool = False
    ) -> torch.Tensor:
        r"""adaptive grid is drawn given a fixed budget of points"""
        voxel_densities = (
            self.spatial_densities_basegrid / self.spatial_densities_basegrid.sum()
        )
        n_samples_voxel = torch.tensor(
            [
                closest_perfect_square(n_samples * voxel_density, return_type="prev")
                for voxel_density in voxel_densities.tolist()
            ]
        ).to(torch.long)
        # print(n_samples_voxel)
        if use_sample_size_heuristic:
            base_delta = np.abs(n_samples_voxel.sum().item() - n_samples)
            requested_n_samples = n_samples
            while True:
                # NOTE: this is in fact a heuristic
                # the actual fraction of samples that to preserve is not linear wrt n_samples, depends on the spatial density and n_voxels
                # TODO: find a better heuristic

                new_n_samples = n_samples + 500
                new_n_samples_voxel = torch.tensor(
                    [
                        closest_perfect_square(
                            new_n_samples * voxel_density, return_type="prev"
                        )
                        for voxel_density in voxel_densities.tolist()
                    ]
                ).to(torch.long)
                new_base_delta = np.abs(
                    new_n_samples_voxel.sum().item() - requested_n_samples
                )
                if new_base_delta > base_delta:
                    break
                else:
                    base_delta = new_base_delta
                    # print(base_delta)
                    n_samples = new_n_samples
                    n_samples_voxel = new_n_samples_voxel

            # print(n_samples)
            # print(n_samples_voxel.sum())

        if n_samples_voxel.sum() == 0:
            return torch.tensor([], dtype=torch.float32)

        unique_counts = torch.unique(n_samples_voxel).sqrt().to(torch.long)
        repeated_grid = torch.repeat_interleave(
            self.basegrid_coords_xy, n_samples_voxel, dim=0
        )

        offsets = {}
        for c in unique_counts:
            if c == 0:
                # _offsets = torch.zeros((1, 2), dtype=torch.float32)
                continue
            else:
                bound = (self.voxel_size / 2) * (1 - 1 / c)
                spaces = torch.linspace(-bound, bound, c)

                _xx, _yy = torch.meshgrid(spaces, spaces, indexing="xy")
                _offsets = torch.stack([_xx.flatten(), _yy.flatten()], dim=-1)
            offsets[int(c**2)] = _offsets

        iterated_offsets = torch.cat(
            [offsets[i] for i in n_samples_voxel.tolist() if i != 0]
        )
        if self.noise_sigma > 0:
            noise = self.noise_sigma * torch.randn(
                repeated_grid.shape, generator=self.generator, dtype=repeated_grid.dtype
            )
            iterated_offsets += noise
        return repeated_grid + iterated_offsets

    def _sample_w_upsampling(self, max_upsample: int = 4):
        r"""given a base square grid create an adaptive grid by upsampling"""
        voxel_densities = (
            self.spatial_densities_basegrid - self.spatial_densities_basegrid.min()
        ) / (
            self.spatial_densities_basegrid.max()
            - self.spatial_densities_basegrid.min()
        )

        n_upsample_voxel = (voxel_densities * (max_upsample - 1)).round().int() + 1
        n_samples_voxel = n_upsample_voxel**2

        repeated_grid = torch.repeat_interleave(
            self.basegrid_coords_xy, n_samples_voxel, dim=0
        )

        all_offsets = []
        step = self.voxel_size
        for s in range(1, max_upsample + 1):
            if s == 1:
                off = torch.tensor([[0.0, 0.0]])
            else:
                # Local coordinates from -step/2 to step/2
                bound = (step / 2) * (1 - 1 / s)
                lin = torch.linspace(-bound, bound, s)
                oy, ox = torch.meshgrid(lin, lin, indexing="ij")
                off = torch.stack([ox.flatten(), oy.flatten()], dim=-1)
            all_offsets.append(off)

        indices = n_upsample_voxel - 1

        offset_list = [all_offsets[i.item()] for i in indices]
        iterated_offsets = torch.cat(offset_list, dim=0)

        if self.noise_sigma > 0:
            noise = self.noise_sigma * torch.randn(
                repeated_grid.shape, generator=self.generator, dtype=repeated_grid.dtype
            )
            iterated_offsets += noise
        adaptive_grid = repeated_grid + iterated_offsets

        return adaptive_grid
