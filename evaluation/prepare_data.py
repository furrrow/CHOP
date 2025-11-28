#!/usr/bin/env python3
"""Placeholder for data preparation (downloading/formatting indices)."""

from pathlib import Path
import json


def main() -> None:
    data_root = Path("data")
    data_root.mkdir(parents=True, exist_ok=True)
    # TODO: implement preprocessing / counterfactual annotation merge
    placeholder_index = data_root / "train_index.json"
    if not placeholder_index.exists():
        placeholder_index.write_text(json.dumps([], indent=2))
    print(f"Created placeholder index at {placeholder_index}")


if __name__ == "__main__":
    main()
