from __future__ import annotations

import argparse
import json

import pandas as pd

from experiments.ablation import (
    EXPERIMENTS,
    AblationMessageFactory,
    condition_manifest,
    conditions_for,
    get_condition,
)
from experiments.common import COMIC_TYPES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect ablation conditions and rendered prompts without inference.")
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    parser.add_argument("--condition", help="Render one condition; omit to list the condition registry")
    parser.add_argument("--type", dest="content_type", default="article", choices=COMIC_TYPES)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--dataset-path", default="dataset.csv")
    parser.add_argument("--dataset-image-dir", default="dataset")
    parser.add_argument("--ablation-image-dir", default="ablation_images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conditions = conditions_for(args.experiment)
    if args.condition is None:
        print(json.dumps(condition_manifest(conditions), indent=2))
        return

    condition = get_condition(args.condition)
    if condition.experiment != args.experiment:
        raise ValueError(
            f"Condition {condition.condition_id} belongs to {condition.experiment}, not {args.experiment}"
        )
    dataset = pd.read_csv(args.dataset_path)
    if not 0 <= args.row < len(dataset):
        raise IndexError(f"Row must be between 0 and {len(dataset) - 1}")
    factory = AblationMessageFactory(
        condition,
        args.content_type,
        image_formatter=lambda path: {"type": "image", "path": path},
        dataset_image_dir=args.dataset_image_dir,
        ablation_image_dir=args.ablation_image_dir,
    )
    messages = factory.build(args.row, dataset.iloc[args.row])
    if messages is None:
        raise ValueError(f"Row {args.row} is incompatible with template {args.content_type}")
    print(json.dumps(messages, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
