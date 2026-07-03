import argparse
import json
import logging
import os
from glob import glob
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from einops import repeat
from tqdm import tqdm

from uninformed.utils.graph_drawing import draw_graph_to_u8, prepare_graph_from_nodes

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="graphV2_drawing_1x1.log", encoding="utf-8", level=logging.INFO
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
        default="coordinates-1x1",
    )
    parser.add_argument(
        "--line_thickness",
        type=int,
        default=1,
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
        default=2,
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
        "--out_dir",
        type=str,
        default="graphsV2-1x1",
    )
    return parser


def main(cfg):
    logger.info(cfg)
    df_slide_alignment = pd.read_csv(cfg.slide_alignment_df)
    df_slide_alignment["magnification_factor"] = df_slide_alignment.magnification.apply(
        lambda x: 20.0 / int(x.removesuffix("x"))
    )
    # df_slide_alignment = df_slide_alignment[df_slide_alignment["bench"] == "CCRCC"]
    for i, row in tqdm(df_slide_alignment.iterrows()):
        slides_analysed = glob(
            os.path.join(Path(cfg.hest_bench_root) / row.bench / cfg.out_dir, "*.h5")
        )
        slides_analysed = {Path(_slide).stem for _slide in slides_analysed}
        if row.slide_id in slides_analysed and not cfg.overwrite:
            logger.info(f"slide {row.slide_id} has been analysed, skipping")
            continue
        slide_geojson = row.slide_id + ".geojson"
        logger.info(
            f"processing {Path(cfg.hest_bench_root) / row.bench / cfg.geojson_dir / slide_geojson}"
        )
        with open(
            Path(cfg.hest_bench_root) / row.bench / cfg.geojson_dir / slide_geojson, "r"
        ) as f:
            features = json.load(f)

        logger.info(
            msg=f"processing slide: {row.slide_id},"
            + "\t"
            + "origin: {row.origin},"
            + "\t"
            + "invert_xy: {row.invert_spot_xy},"
            + "\t"
            + "px size: {row.pixel_size_um_estimated},"
            + "\t"
            + "px factor: {row.magnification_factor}"
        )

        graph_list = []
        img_list = []
        bboxes = []
        barcodes = []
        coords = []
        slide_wise_hw_source = []
        slide_wise_hw = []
        for feature in tqdm(features["features"]):
            if len(feature["geometry"]["coordinates"]) > 0:
                pos = np.array(feature["geometry"]["coordinates"])
                bbox = np.array(feature["properties"]["bbox"])
                bbox_w = int(bbox[2] - bbox[0])
                barcode = feature["properties"]["barcode"]
                bbox_h = int(bbox[3] - bbox[1])
                scale_factor = max(bbox_w, bbox_h)
                slide_wise_hw_source.append([bbox_w, bbox_h])
                coords_xy = np.array([bbox[0], bbox[1]])
                coords.append(coords_xy)
                if cfg.rescale_patches:
                    pos = pos - coords_xy.reshape(1, 2)
                    pos *= float(row.pixel_size_um_estimated) * cfg.px_size_scaler
                    bbox_h = np.rint(
                        bbox_h * float(row.pixel_size_um_estimated) * cfg.px_size_scaler
                    ).astype(int)
                    bbox_w = np.rint(
                        bbox_w * float(row.pixel_size_um_estimated) * cfg.px_size_scaler
                    ).astype(int)
                    slide_wise_hw.append([bbox_w, bbox_h])
                    coords_xy *= 0
                g = prepare_graph_from_nodes(node_positions=pos, k=cfg.k)
                graph_list.append(g)
                bboxes.append([bbox_w, bbox_h])
                img = draw_graph_to_u8(
                    graph=g,
                    coord_x0y0=coords_xy,
                    line_thickness=cfg.line_thickness,
                    source_size=(bbox_w, bbox_h),
                )
                img_list.append(img)
                barcodes.append(barcode)

        slide_wise_hw_source = np.array(slide_wise_hw_source)
        logger.info(f"source slide mean hw = {slide_wise_hw_source.mean(axis=0)}")
        if cfg.rescale_patches:
            slide_wise_hw = np.array(slide_wise_hw)
            logger.info(f"rescaled slide mean hw = {slide_wise_hw.mean(axis=0)}")
        img_array = np.stack(img_list, axis=0).squeeze()
        img_array = repeat(img_array, "b h w -> b h w c", c=cfg.out_channels)
        coords = np.array(coords)
        logging.info(f"produced img array of shape {img_array.shape}")
        logging.info("saving ... ")
        os.makedirs(Path(cfg.hest_bench_root) / row.bench / cfg.out_dir, exist_ok=True)
        save_fname = (
            Path(cfg.hest_bench_root) / row.bench / cfg.out_dir / (row.slide_id + ".h5")
        )
        with h5py.File(name=save_fname, mode="w") as f:
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset(
                name="barcode", data=np.array(barcodes, dtype=object), dtype=dt
            )
            f.create_dataset(name="img", data=img_array, compression="gzip")
            f.create_dataset(name="coords", data=coords, compression="gzip")
        logging.info(f"wrote data into {save_fname}")


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
