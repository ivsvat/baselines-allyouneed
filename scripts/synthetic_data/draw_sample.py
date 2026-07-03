import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from einops import repeat

from uninformed.utils.graph_drawing import (
    draw_graph_to_u8,
    prepare_graph_from_nodes,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="graphV2_drawing_onesample_3x3.log", encoding="utf-8", level=logging.INFO
)


def get_parser():
    parser = argparse.ArgumentParser(
        prog="Draw cellseg alignment to spots",
        description="Run for images in st patches and full-sized WSIs",
    )
    parser.add_argument(
        "--slide_alignment_df",
        type=str,
        default="./assets/patch_alignment.csv",
    )
    parser.add_argument(
        "--hest_bench_root",
        type=str,
        default="PATH_TO_HEST1k/bench/",
    )
    parser.add_argument(
        "--geojson_dir",
        type=str,
        default="coordinates-3x3",
    )
    parser.add_argument(
        "--line_thickness",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--overwrite",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--px_size_scaler",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--rescale_patches",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--slide_id",
        type=str,
        default="TENX147",
    )
    parser.add_argument(
        "--feature_id",
        type=int,
        default="103",
    )
    return parser


def main(cfg):
    logger.info(cfg)
    df_slide_alignment = pd.read_csv(cfg.slide_alignment_df)
    df_slide_alignment["magnification_factor"] = df_slide_alignment.magnification.apply(
        lambda x: 20.0 / int(x.removesuffix("x"))
    )

    df_slide_alignment = df_slide_alignment[df_slide_alignment.slide_id == cfg.slide_id]
    row = df_slide_alignment.iloc[0]

    slide_geojson = row.slide_id + ".geojson"
    logger.info(
        f"processing {Path(cfg.hest_bench_root) / row.bench / cfg.geojson_dir / slide_geojson}"
    )
    with open(
        Path(cfg.hest_bench_root) / row.bench / cfg.geojson_dir / slide_geojson, "r"
    ) as f:
        features = json.load(f)
    logger.info(
        msg=f"processing slide: {row.slide_id},\t origin: {row.origin},\t invert_xy: {row.invert_spot_xy}, \t px size: {row.pixel_size_um_estimated}, \t px factor: {row.magnification_factor}"
    )

    feature = features["features"][cfg.feature_id]
    img = None
    if len(feature["geometry"]["coordinates"]) > 0:
        pos = np.array(feature["geometry"]["coordinates"])
        bbox = np.array(feature["properties"]["bbox"])
        bbox_w = int(bbox[2] - bbox[0])
        barcode = feature["properties"]["barcode"]
        bbox_h = int(bbox[3] - bbox[1])
        scale_factor = max(bbox_w, bbox_h)

        coords_xy = np.array([bbox[0], bbox[1]])

        if cfg.rescale_patches:
            pos = pos - coords_xy.reshape(1, 2)
            pos *= float(row.pixel_size_um_estimated) * cfg.px_size_scaler
            bbox_h = np.rint(
                bbox_h * float(row.pixel_size_um_estimated) * cfg.px_size_scaler
            ).astype(int)
            bbox_w = np.rint(
                bbox_w * float(row.pixel_size_um_estimated) * cfg.px_size_scaler
            ).astype(int)

            coords_xy *= 0
        g = prepare_graph_from_nodes(node_positions=pos, k=cfg.k)

        img = draw_graph_to_u8(
            graph=g,
            coord_x0y0=coords_xy,
            line_thickness=cfg.line_thickness,
            source_size=(bbox_w, bbox_h),
            output_size=(1024, 1024),
        )
    if img is not None:
        img = repeat(img, "h w -> h w c", c=cfg.out_channels)
        fig = plt.figure(figsize=(6, 6), frameon=False)
        fig.set_size_inches(6, 6)
        ax = plt.Axes(fig, [0, 0, 1, 1])
        ax.imshow((1 - img / 255.0) * (np.array([137, 203, 240]) / 255.0) + img / 255.0)
        ax.set_axis_off()
        fig.add_axes(ax)
        fig.savefig(
            f"{cfg.slide_id}-feat_{cfg.feature_id}.pdf",
            format="pdf",
            dpi=300,
        )
        plt.close()


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
