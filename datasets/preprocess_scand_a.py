from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Tuple


Annotation = MutableMapping[str, Any]
PathDict = Dict[str, Any]


def _get_rankings(annotation: Annotation) -> Optional[List[str]]:
    """Return ordered path rankings as strings."""
    preference = annotation.get("preference", [])
    if not isinstance(preference, Iterable):
        return None
    pref_list = [str(p) for p in preference]
    return pref_list if pref_list else None

def _process_annotation_file(
    json_path: Path, images_root: Path, image_ext: str
) -> List[Dict[str, Any]]:
    with json_path.open("r") as f:
        raw = json.load(f)

    bag_name = Path(raw.get("bag", json_path.stem)).stem
    annotations = raw.get("annotations_by_stamp") or {}

    processed: List[Dict[str, Any]] = []
    for stamp, annotation in annotations.items():
        rankings = _get_rankings(annotation)
        if rankings is None or len(rankings) < 2:
            continue

        paths: Dict[str, PathDict] = annotation.get("paths") or {}
        path_0_data = paths.get(rankings[0])
        path_1_data = paths.get(rankings[1])

        if not path_0_data or not path_1_data:
            continue

        def _extract_path(data: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "points": data.get("points", []),
                "left_boundary": data.get("left_boundary", []),
                "right_boundary": data.get("right_boundary", []),
            }

        image_path = images_root / bag_name / f"{stamp}.{image_ext}"
        processed.append(
            {
                "timestamp": stamp,
                "frame_idx": annotation.get("frame_idx"),
                "robot_width": annotation.get("robot_width"),
                "image_path": str(image_path),
                "path_0": _extract_path(path_0_data),
                "path_1": _extract_path(path_1_data),
                "position": annotation.get("position"),
                "yaw": annotation.get("yaw"),
                "stop": annotation.get("stop", False),
            }
        )

    return processed


class _JsonArrayWriter:
    """Minimal streaming JSON array writer to avoid holding all data in memory."""

    def __init__(self, path: Path, pretty: bool = False):
        self.path = path
        self.pretty = pretty
        self._fh = path.open("w")
        self._empty = True

    def write(self, obj: Any) -> None:
        if self._empty:
            self._fh.write("[\n" if self.pretty else "[")
            json.dump(obj, self._fh, separators=(",", ":"), indent=2 if self.pretty else None)
            self._empty = False
        else:
            self._fh.write(",\n" if self.pretty else ",")
            json.dump(obj, self._fh, separators=(",", ":"), indent=2 if self.pretty else None)

    def close(self) -> None:
        if self._fh is None:
            return
        if self._empty:
            self._fh.write("[]\n")
        else:
            if self.pretty:
                self._fh.write("\n")
            self._fh.write("]\n")
        self._fh.close()
        self._fh = None

    def __enter__(self) -> "_JsonArrayWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def preprocess_scand(
    scand_dir: Path,
    images_root: Path,
    output_dir: Path,
    test_train_split_json: Path,
    image_ext: str = "jpg",
    default_split: str = "train",
) -> Tuple[int, int]:
    """Generate per-split SCAND-A indices, grouping samples by bag within train/test files."""
    split_map = json.load(test_train_split_json.open("r")) if test_train_split_json.exists() else {}
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.json"
    test_path = output_dir / "test.json"
    train_count = 0
    test_count = 0

    with _JsonArrayWriter(train_path, pretty=True) as train_writer, _JsonArrayWriter(test_path, pretty=True) as test_writer:
        for json_file in sorted(scand_dir.glob("*.json")):
            entries = _process_annotation_file(json_file, images_root, image_ext)
            if not entries:
                continue

            bag_name = Path(json_file).stem
            split = split_map.get(bag_name, default_split)
            writer = train_writer if split == "train" else test_writer

            writer.write({"bag": bag_name, "samples": entries})
            if writer is train_writer:
                train_count += len(entries)
            else:
                test_count += len(entries)

    return train_count, test_count


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
        default=Path(__file__).resolve().parent.parent / "data" / "images",
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
        default="jpg",
        help="Image extension to use when constructing image paths (e.g., jpg or png).",
    )
    parser.add_argument(
        "--test-train-split-json",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "annotations" / "test-train-split.json",
        help="Path to the JSON file defining the train/test split.",
    )
    args = parser.parse_args()

    train_count, test_count = preprocess_scand(
        args.scand_dir,
        args.images_root,
        args.output_dir,
        args.test_train_split_json,
        args.image_ext,
    )
    print(f"Wrote {train_count} train samples to {args.output_dir / 'train.json'} (bag-grouped)")
    print(f"Wrote {test_count} test samples to {args.output_dir / 'test.json'} (bag-grouped)")


if __name__ == "__main__":
    main()
