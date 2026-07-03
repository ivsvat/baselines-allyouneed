from uninformed.dataops.synthetic_graphs import (
    NoisyGridGraphMaker,
    PoissonGraphMaker,
)
from uninformed.dataops.landscapes import (
    UniformLandscape,
    SlopeLandscape,
    StepLandscape,
    EmbossedDiscsLandscape,
)
from math import sqrt

LAMBDA_FUNCTIONS = {
    "UniformLandscape": UniformLandscape(),
    "SlopeLandscape(1,1)": SlopeLandscape(k=1.0, b=1.0),
    "SlopeLandscape(2,2)": SlopeLandscape(k=2.0, b=2.0),
    "SlopeLandscape(3,3)": SlopeLandscape(k=3.0, b=3.0),
    "StepLandscape(0.5,1)": StepLandscape(step_x=0.5, step_delta=1),
    # Embossed/ debossed 1 disc
    "EmbossedDiscsLandscape(1,0.1,emboss,2)": EmbossedDiscsLandscape(
        n_discs=1,
        base_radius=0.1,
        mode="emboss",
        density_delta=2.0,
        randomise_single_disc_position=True,
    ),
    "EmbossedDiscsLandscape(1,0.1,deboss)": EmbossedDiscsLandscape(
        n_discs=1, base_radius=0.1, mode="deboss", randomise_single_disc_position=True
    ),
    # Embossed/ debossed 3 discs increased total mass
    "EmbossedDiscsLandscape(3,0.1,emboss,2)": EmbossedDiscsLandscape(
        n_discs=3, base_radius=0.1, mode="emboss", density_delta=2.0
    ),
    "EmbossedDiscsLandscape(3,0.1,deboss)": EmbossedDiscsLandscape(
        n_discs=3, base_radius=0.1, mode="deboss"
    ),
    # Embossed/ debossed 1 disc with larger mass
    "EmbossedDiscsLandscape(1,0.2,emboss,2)": EmbossedDiscsLandscape(
        n_discs=1,
        base_radius=0.2,
        mode="emboss",
        density_delta=2.0,
        randomise_single_disc_position=True,
    ),
    "EmbossedDiscsLandscape(1,0.2,deboss)": EmbossedDiscsLandscape(
        n_discs=1, base_radius=0.2, mode="deboss", randomise_single_disc_position=True
    ),
    # Embossed/ debossed 3 discs, constant total mass
    "EmbossedDiscsLandscape(3,sqrt(0.2^2/3),emboss,2)": EmbossedDiscsLandscape(
        n_discs=3, base_radius=sqrt((0.2**2) / 3), mode="emboss", density_delta=2.0
    ),
    "EmbossedDiscsLandscape(3,sqrt(0.2^2/3),deboss)": EmbossedDiscsLandscape(
        n_discs=3, base_radius=sqrt((0.2**2) / 3), mode="deboss"
    ),
    # Embossed/ debossed 5 discs, constant total mass
    "EmbossedDiscsLandscape(5,sqrt(0.2^2/5),emboss,2)": EmbossedDiscsLandscape(
        n_discs=5, base_radius=sqrt((0.2**2) / 5), mode="emboss", density_delta=2.0
    ),
    "EmbossedDiscsLandscape(5,sqrt(0.2^2/5),deboss)": EmbossedDiscsLandscape(
        n_discs=5, base_radius=sqrt((0.2**2) / 5), mode="deboss"
    ),
}

CLASSES = {
    "class_0": {
        "fn_class": LAMBDA_FUNCTIONS["UniformLandscape"],
        "sampler": NoisyGridGraphMaker(
            spatial_density=LAMBDA_FUNCTIONS["UniformLandscape"].lmbd_fn,
            n_voxels=10,
            noise_sigma=0.01,
        ),
        "sampler_kwargs": {"mode": "spatially_varying_noise", "upsample_by": 3},
    },
    "class_1": {
        "fn_class": LAMBDA_FUNCTIONS["SlopeLandscape(3,3)"],
        "sampler": NoisyGridGraphMaker(
            spatial_density=LAMBDA_FUNCTIONS["SlopeLandscape(3,3)"].lmbd_fn,
            n_voxels=10,
            noise_sigma=0.01,
        ),
        "sampler_kwargs": {"mode": "spatially_varying_noise", "upsample_by": 3},
    },
    "class_2": {
        "fn_class": LAMBDA_FUNCTIONS["SlopeLandscape(3,3)"],
        "sampler": NoisyGridGraphMaker(
            spatial_density=LAMBDA_FUNCTIONS["SlopeLandscape(3,3)"].lmbd_fn,
            n_voxels=10,
            noise_sigma=0.01,
        ),
        "sampler_kwargs": {
            "mode": "budget",
            "n_samples": 1200,
        },  # 600-> 390 # 1200->890
    },
    "class_3": {
        "fn_class": LAMBDA_FUNCTIONS["StepLandscape(0.5,1)"],
        "sampler": NoisyGridGraphMaker(
            spatial_density=LAMBDA_FUNCTIONS["StepLandscape(0.5,1)"].lmbd_fn,
            n_voxels=4,
            noise_sigma=0.01,
        ),
        "sampler_kwargs": {
            "mode": "budget",
            "n_samples": 1000,
        },  # 1200 - > 1000 # 1000 -> 848
    },
    "class_4": {
        "fn_class": LAMBDA_FUNCTIONS["EmbossedDiscsLandscape(1,0.1,emboss,2)"],
        "sampler": NoisyGridGraphMaker(
            spatial_density=LAMBDA_FUNCTIONS[
                "EmbossedDiscsLandscape(1,0.1,emboss,2)"
            ].lmbd_fn,
            n_voxels=10,
            noise_sigma=0.01,
        ),
        "sampler_kwargs": {"mode": "budget", "n_samples": 1000},  # 1000 -> 964
    },
    "class_5": {
        "fn_class": LAMBDA_FUNCTIONS["EmbossedDiscsLandscape(3,0.1,emboss,2)"],
        "sampler": NoisyGridGraphMaker(
            spatial_density=LAMBDA_FUNCTIONS[
                "EmbossedDiscsLandscape(3,0.1,emboss,2)"
            ].lmbd_fn,
            n_voxels=10,
            noise_sigma=0.01,
        ),
        "sampler_kwargs": {"mode": "budget", "n_samples": 1100},  # 1100 -> 1028
    },
    "class_6": {
        "fn_class": LAMBDA_FUNCTIONS["EmbossedDiscsLandscape(1,0.1,deboss)"],
        "sampler": NoisyGridGraphMaker(
            spatial_density=LAMBDA_FUNCTIONS[
                "EmbossedDiscsLandscape(1,0.1,deboss)"
            ].lmbd_fn,
            n_voxels=10,
            noise_sigma=0.01,
        ),
        "sampler_kwargs": {"mode": "budget", "n_samples": 1100},  # 1100 -> 864
    },
    "class_7": {
        "fn_class": LAMBDA_FUNCTIONS["EmbossedDiscsLandscape(3,0.1,deboss)"],
        "sampler": NoisyGridGraphMaker(
            spatial_density=LAMBDA_FUNCTIONS[
                "EmbossedDiscsLandscape(3,0.1,deboss)"
            ].lmbd_fn,
            n_voxels=10,
            noise_sigma=0.01,
        ),
        "sampler_kwargs": {"mode": "budget", "n_samples": 900},  # 900 -> 819
    },
    "class_8": {
        "fn_class": LAMBDA_FUNCTIONS["EmbossedDiscsLandscape(3,0.1,emboss,2)"],
        "sampler": PoissonGraphMaker(
            spatial_density=LAMBDA_FUNCTIONS[
                "EmbossedDiscsLandscape(3,0.1,emboss,2)"
            ].lmbd_fn,
            lmbd=2271,  # ~900
            lmbd_star=3,
        ),
        "sampler_kwargs": {},  # 2500 -> 934
    },
    "class_9": {
        "fn_class": LAMBDA_FUNCTIONS["EmbossedDiscsLandscape(1,0.1,deboss)"],
        "sampler": PoissonGraphMaker(
            spatial_density=LAMBDA_FUNCTIONS[
                "EmbossedDiscsLandscape(1,0.1,deboss)"
            ].lmbd_fn,
            lmbd=930,  # ~900
            lmbd_star=1,
        ),
        "sampler_kwargs": {},
    },
    "class_10": {
        "fn_class": LAMBDA_FUNCTIONS["UniformLandscape"],
        "sampler": PoissonGraphMaker(
            spatial_density=LAMBDA_FUNCTIONS["UniformLandscape"].lmbd_fn,
            lmbd=900,
            lmbd_star=1,
        ),
        "sampler_kwargs": {},
    },
    "class_11": {
        "fn_class": LAMBDA_FUNCTIONS["EmbossedDiscsLandscape(1,0.1,emboss,2)"],
        "sampler": PoissonGraphMaker(
            spatial_density=LAMBDA_FUNCTIONS[
                "EmbossedDiscsLandscape(1,0.1,emboss,2)"
            ].lmbd_fn,
            lmbd=2540,  # ~900
            lmbd_star=3,
        ),
        "sampler_kwargs": {},  # 2500 -> 934
    },
    "class_12": {
        "fn_class": LAMBDA_FUNCTIONS["EmbossedDiscsLandscape(1,0.1,deboss)"],
        "sampler": PoissonGraphMaker(
            spatial_density=LAMBDA_FUNCTIONS[
                "EmbossedDiscsLandscape(1,0.1,deboss)"
            ].lmbd_fn,
            lmbd=930,
            lmbd_star=1,  # 1100 -> 919
        ),
        "sampler_kwargs": {},
    },
    "class_13": {
        "fn_class": LAMBDA_FUNCTIONS["StepLandscape(0.5,1)"],
        "sampler": NoisyGridGraphMaker(
            spatial_density=LAMBDA_FUNCTIONS["StepLandscape(0.5,1)"].lmbd_fn,
            n_voxels=10,
            noise_sigma=0.01,
        ),
        "sampler_kwargs": {"mode": "spatially_varying_noise", "upsample_by": 3},
    },
    "class_14": {
        "fn_class": LAMBDA_FUNCTIONS[
            "EmbossedDiscsLandscape(3,sqrt(0.2^2/3),emboss,2)"
        ],
        "sampler": PoissonGraphMaker(
            spatial_density=LAMBDA_FUNCTIONS[
                "EmbossedDiscsLandscape(3,sqrt(0.2^2/3),emboss,2)"
            ].lmbd_fn,
            lmbd=2158,  # ~900
            lmbd_star=3,
        ),
        "sampler_kwargs": {},
    },
    "class_15": {
        "fn_class": LAMBDA_FUNCTIONS["EmbossedDiscsLandscape(3,sqrt(0.2^2/3),deboss)"],
        "sampler": PoissonGraphMaker(
            spatial_density=LAMBDA_FUNCTIONS[
                "EmbossedDiscsLandscape(3,sqrt(0.2^2/3),deboss)"
            ].lmbd_fn,
            lmbd=1029,  # ~900
            lmbd_star=1,
        ),
        "sampler_kwargs": {},
    },
    "class_16": {
        "fn_class": LAMBDA_FUNCTIONS[
            "EmbossedDiscsLandscape(5,sqrt(0.2^2/5),emboss,2)"
        ],
        "sampler": PoissonGraphMaker(
            spatial_density=LAMBDA_FUNCTIONS[
                "EmbossedDiscsLandscape(5,sqrt(0.2^2/5),emboss,2)"
            ].lmbd_fn,
            lmbd=2158,  # ~900
            lmbd_star=3,
        ),
        "sampler_kwargs": {},
    },
    "class_17": {
        "fn_class": LAMBDA_FUNCTIONS["EmbossedDiscsLandscape(5,sqrt(0.2^2/5),deboss)"],
        "sampler": PoissonGraphMaker(
            spatial_density=LAMBDA_FUNCTIONS[
                "EmbossedDiscsLandscape(5,sqrt(0.2^2/5),deboss)"
            ].lmbd_fn,
            lmbd=1029,  # ~900
            lmbd_star=1,
        ),
        "sampler_kwargs": {},
    },
    "class_18": {
        "fn_class": LAMBDA_FUNCTIONS["SlopeLandscape(3,3)"],
        "sampler": PoissonGraphMaker(
            spatial_density=LAMBDA_FUNCTIONS["SlopeLandscape(3,3)"].lmbd_fn,
            lmbd=1440,  # ~900
            lmbd_star=LAMBDA_FUNCTIONS["SlopeLandscape(3,3)"].maximum_on_01,
        ),
        "sampler_kwargs": {},
    },
    "class_19": {
        "fn_class": LAMBDA_FUNCTIONS["SlopeLandscape(2,2)"],
        "sampler": PoissonGraphMaker(
            spatial_density=LAMBDA_FUNCTIONS["SlopeLandscape(2,2)"].lmbd_fn,
            lmbd=1350,  # ~900
            lmbd_star=LAMBDA_FUNCTIONS["SlopeLandscape(2,2)"].maximum_on_01,
        ),
        "sampler_kwargs": {},
    },
    "class_20": {
        "fn_class": LAMBDA_FUNCTIONS["SlopeLandscape(1,1)"],
        "sampler": PoissonGraphMaker(
            spatial_density=LAMBDA_FUNCTIONS["SlopeLandscape(1,1)"].lmbd_fn,
            lmbd=1200,  # ~900
            lmbd_star=LAMBDA_FUNCTIONS["SlopeLandscape(1,1)"].maximum_on_01,
        ),
        "sampler_kwargs": {},
    },
    "class_21": {
        "fn_class": LAMBDA_FUNCTIONS["EmbossedDiscsLandscape(1,0.2,emboss,2)"],
        "sampler": PoissonGraphMaker(
            spatial_density=LAMBDA_FUNCTIONS[
                "EmbossedDiscsLandscape(1,0.2,emboss,2)"
            ].lmbd_fn,
            lmbd=2158,  # ~900
            lmbd_star=3,
        ),
        "sampler_kwargs": {},
    },
    "class_22": {
        "fn_class": LAMBDA_FUNCTIONS["EmbossedDiscsLandscape(1,0.2,deboss)"],
        "sampler": PoissonGraphMaker(
            spatial_density=LAMBDA_FUNCTIONS[
                "EmbossedDiscsLandscape(1,0.2,deboss)"
            ].lmbd_fn,
            lmbd=1029,  # ~900
            lmbd_star=1,
        ),
        "sampler_kwargs": {},
    },
    "class_23": {
        "fn_class": LAMBDA_FUNCTIONS["StepLandscape(0.5,1)"],
        "sampler": PoissonGraphMaker(
            spatial_density=LAMBDA_FUNCTIONS["StepLandscape(0.5,1)"].lmbd_fn,
            lmbd=1200,  # ~900
            lmbd_star=LAMBDA_FUNCTIONS["StepLandscape(0.5,1)"].maximum_on_01,
        ),
        "sampler_kwargs": {},
    },
}
