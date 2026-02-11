from __future__ import annotations

import argparse
import json
from collections import Counter
from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Tuple
import numpy as np
from preprocess_scand_a_chop import _process_annotation_file, _JsonArrayWriter

import os
import cv2
from dataclasses import dataclass
import math

from vis_utils import point_to_traj, make_corridor_polygon, draw_polyline, draw_corridor, transform_points, \
    project_points_cam, load_calibration, camray_to_ground_in_base
from traj_utils import solve_arc_from_point


Annotation = MutableMapping[str, Any]
PathDict = Dict[str, Any]

@dataclass
class FrameItem:
    idx: int
    stamp: object   # rospy.Time
    img: np.ndarray
    r: float
    theta: float
    sub_goal : list   # [x, y, z] in base_link
    width_m : float
    u : float
    v : float
    stop : bool


def draw(frame_item: FrameItem, K: np.ndarray, dist, T_cam_from_base: np.ndarray, window: str = "SCAND Verification"):
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    COLOR_PATH = (0, 0, 255)  # red
    COLOR_CLICK = (255, 0, 0)  # green

    current_img = frame_item.img
    x_base, y_base, z_base = frame_item.sub_goal
    width_m = frame_item.width_m
    r, theta = frame_item.r, frame_item.theta
    u, v = frame_item.u, frame_item.v
    stop = frame_item.stop
    T_horizon = 2.0  # Path generation options
    num_t_samples = 10

    if not stop:
        r, theta = solve_arc_from_point(x_base, y_base)
        # print(x_base, y_base)
        path_points, v_x, w, t_samples, theta_samples = point_to_traj(r, theta, T_horizon, num_t_samples, x_base, y_base)
        left_b, right_b, poly_b = make_corridor_polygon(path_points, theta_samples, width_m, bridge_pts=20)

        # print(path_points[-10:, :])

        # 4) Transform to camera and project
        traj_c = transform_points(T_cam_from_base, path_points)
        left_c = transform_points(T_cam_from_base, left_b)
        right_c = transform_points(T_cam_from_base, right_b)
        poly_c = transform_points(T_cam_from_base, poly_b)

        # print(traj_c[-10:, :])

        ctr_2d = project_points_cam(K, dist, traj_c)
        left_2d = project_points_cam(K, dist, left_c)
        right_2d = project_points_cam(K, dist, right_c)
        poly_2d = project_points_cam(K, dist, poly_c)
        # Project to image

        # print(ctr_2d[-10:, :])

        # print("\n")
        # print("Next")
        # print("\n")

        if current_img is None:
            return
        img = current_img.copy()

        draw_polyline(img, ctr_2d, 2, COLOR_PATH)
        draw_corridor(img, poly_2d, left_2d, right_2d,
                      fill_alpha=0.35, fill_color=COLOR_PATH, edge_color=COLOR_PATH, edge_thickness=2)

        cv2.circle(img, (int(u), int(v)), 5, COLOR_CLICK, -1)
    else:
        ## transclucent red overlay for stop
        img = current_img.copy()
        overlay = img.copy()
        overlay[:] = (0, 0, 255)
        img[:] = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

    cv2.imshow(window, img)

def visualize_scand(
    scand_dir: Path,
    images_root: Path,
    output_dir: Path,
    test_train_split_json: Path,
    image_ext: str = "jpg",
    default_split: str = "train",
    num_points: int = 8,
) -> Tuple[int, int]:
    """Generate per-split SCAND-A indices, grouping samples by bag within train/test files."""
    split_map = json.load(test_train_split_json.open("r")) if test_train_split_json.exists() else {}
    output_dir.mkdir(parents=True, exist_ok=True)

    """constants for visualization"""
    fx, fy, cx, cy = 640.0, 637.0, 640.0, 360.0  # SCAND Kinect intrinsics ### DO NOT CHANGE
    calib_path = "./tf.json"

    K, dist, T_base_from_cam = load_calibration(calib_path, fx, fy, cx, cy, mode="spot")
    T_cam_from_base = np.linalg.inv(T_base_from_cam)

    for json_file in tqdm(sorted(scand_dir.glob("*.json"))):
        entries = _process_annotation_file(json_file, images_root, image_ext, num_points)
        for idx, i_entry in enumerate(entries):
            image_full_path = os.path.join(images_root, i_entry['image_path'])
            fr = FrameItem(idx=idx,
                           stamp=None,
                           img=cv2.imread(image_full_path),
                           r=1.0,
                           theta=1.0,
                           sub_goal=[50, 50, 0],
                           width_m=1.0,
                           u=1.0,
                           v=1.0,
                           stop=False)
            draw(fr, K , dist, T_cam_from_base)
            print(entries)
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess SCAND-A annotations into a flat index of image/trajectory pairs."
    )
    parser.add_argument(
        "--scand-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "annotations" / "preferences",
        help="Directory containing SCAND annotation JSON files.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        # default=Path(__file__).resolve().parent.parent / "data" / "images",
        default=Path("/media/jim/Ironwolf/datasets/scand_data/images"),
        help="Root directory containing extracted SCAND images (organized by bag name).",
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "lora-data",
        help="Directory to write the train/test annotation indices.",
    )
    parser.add_argument(
        "--image-ext",
        type=str,
        default="png",
        help="Image extension to use when constructing image paths (e.g., jpg or png).",
    )
    parser.add_argument(
        "--test-train-split-json",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "annotations" / "test-train-split.json",
        help="Path to the JSON file defining the train/test split.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=8,
        help="Number of points to sample from each trajectory.",
    )
    args = parser.parse_args()

    train_count, test_count = visualize_scand(
        args.scand_dir,
        args.images_root,
        args.output_dir,
        args.test_train_split_json,
        args.image_ext,
        args.num_points,
    )

    print(f"Wrote {train_count} train samples to {args.output_dir / 'train.json'} (bag-grouped)")
    print(f"Wrote {test_count} test samples to {args.output_dir / 'test.json'} (bag-grouped)")


if __name__ == "__main__":
    main()
