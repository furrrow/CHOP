#!/usr/bin/env python3
"""Visualize counterfactual annotations for quick QA."""

from pathlib import Path
import json


def main() -> None:
    index_path = Path("data/train_index.json")
    if not index_path.exists():
        raise SystemExit(f"Index not found: {index_path}. Run scripts/prepare_data.py first.")

    rows = json.loads(index_path.read_text())
    print(f"Loaded {len(rows)} counterfactual rows. Implement visualization here.")


if __name__ == "__main__":
    main()
