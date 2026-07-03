import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.v2 import functional as v2F
from tqdm import tqdm

from uninformed.configs.synthetic_graphs import CLASSES
from uninformed.dataops.synthetic_graphs import NoisyGridGraphMaker, PoissonGraphMaker
from uninformed.utils.graph_drawing import build_and_draw_graph


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["img_png", "coordinates_json"],
        default="coordinates_json",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./output/synthetic_graphs_geojson/val",
    )
    parser.add_argument(
        "--classes_to_sample",
        type=str,
        nargs="+",
        default=[
            # "class_0",
            # "class_1",
            # "class_2",
            # "class_3",
            # "class_4",
            # "class_5",
            # "class_6",
            # "class_7",
            # "class_8",
            # "class_9",
            # "class_10",
            # "class_11",
            # "class_12",
            # "class_13",
            # "class_14",
            # "class_15",
            # "class_16",
            # "class_17",
            # "class_18",
            # "class_19",
            # "class_20",
            # "class_21",
            # "class_22",
            "class_23",
        ],
    )
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    #
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    g = torch.Generator().manual_seed(args.seed)
    for class_to_sample in tqdm(args.classes_to_sample):
        Path(Path(args.out_dir + f"_seed{args.seed}") / class_to_sample).mkdir(
            parents=True, exist_ok=True
        )
        fn = CLASSES[class_to_sample]["fn_class"].lmbd_fn
        sampler = CLASSES[class_to_sample]["sampler"]
        sampler.generator = g
        print(
            f"Class: {class_to_sample}, \t Using lambda fn from: {type(CLASSES[class_to_sample]['fn_class'])}"
        )
        print(f"Class: {class_to_sample}, \t Using a sampler from:{type(sampler)}")

        assert isinstance(sampler, PoissonGraphMaker) or isinstance(
            sampler, NoisyGridGraphMaker
        )
        if isinstance(sampler, PoissonGraphMaker):
            poisson_samples = []
        else:
            poisson_samples = None

        for i in tqdm(range(args.n_samples)):
            points = sampler.sample(**CLASSES[class_to_sample]["sampler_kwargs"])
            if poisson_samples is not None:
                poisson_samples.append(points.shape[0])

            source_h, source_w = 800, 800
            if args.mode == "img_png":
                img = build_and_draw_graph(
                    node_positions=points * source_h,
                    k=5,
                    coord_x0y0=torch.tensor([0.0, 0.0]),
                    source_size=(source_h, source_w),
                    line_thickness=2,
                )
                v2F.to_pil_image(img).save(
                    Path(args.out_dir + f"_seed{args.seed}")
                    / class_to_sample
                    / f"img-{i}.png"
                )
            elif args.mode == "coordinates_json":
                graph_dict = {
                    "type": "Feature",
                    "geometry": {
                        "coordinates": points.tolist(),
                        "type": "MultiPoint",
                    },
                    "properties": {
                        "label": class_to_sample,
                        "cell_count": points.shape[0],
                        "sampler": type(sampler).__name__,
                        "lambda_fn": type(
                            CLASSES[class_to_sample]["fn_class"]
                        ).__name__,
                        "bbox": [0, 0, 1, 1],
                        "bbox_format": "xyxy",
                    },
                }
                with open(
                    Path(args.out_dir + f"_seed{args.seed}")
                    / class_to_sample
                    / f"graph-{i}.geojson",
                    "w",
                ) as f:
                    json.dump(graph_dict, f, indent=4)

        if poisson_samples is not None:
            print(f"Estimated poisson mean is {np.mean(poisson_samples)}")
            print(f"Estimated poisson std is {np.std(poisson_samples)}")
