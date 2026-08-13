from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from experiments.ablation import EXPERIMENTS, conditions_for
from experiments.common import COMIC_TYPES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a selected ablation matrix sequentially, reusing one CUDA model in local mode."
    )
    parser.add_argument("--backend", choices=("local", "openrouter"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--types", nargs="+", choices=COMIC_TYPES, default=list(COMIC_TYPES))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--dataset-path", default="dataset.csv")
    parser.add_argument("--dataset-image-dir", default="dataset")
    parser.add_argument("--ablation-image-dir", default="ablation_images")
    parser.add_argument("--behavior-output-dir", default="ablation_responses/behavior")
    parser.add_argument("--recognition-output-dir", default="ablation_responses/recognition")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    return parser.parse_args()


def selected_conditions(experiments: list[str]):
    selected = set(experiments)
    return [
        condition
        for experiment in EXPERIMENTS
        if experiment in selected
        for condition in conditions_for(experiment)
    ]


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", value).strip("-")


def _output_path(output_dir: str, model: str, condition_id: str, content_type: str) -> Path:
    return Path(output_dir) / f"{_slug(model)}__{condition_id}__{content_type}.csv"


def _reuse_original_visual_anchor(args: argparse.Namespace, condition, content_type: str) -> bool:
    if condition.condition_id != "structure_original":
        return False
    source = _output_path(
        args.behavior_output_dir,
        args.model,
        "visual_r1_c1",
        content_type,
    )
    destination = _output_path(
        args.behavior_output_dir,
        args.model,
        condition.condition_id,
        content_type,
    )
    if not source.exists() or (destination.exists() and not args.overwrite):
        return False
    frame = pd.read_csv(source)
    if args.backend == "local":
        settings_match = frame["Max_Tokens"].eq(args.max_tokens).all()
    else:
        settings_match = (
            frame["Max_Tokens"].eq(args.max_tokens).all()
            and frame["Temperature"].eq(args.temperature).all()
            and frame["Seed"].eq(args.seed).all()
        )
    if not settings_match:
        return False
    frame["Experiment"] = "visual_structure"
    frame["Condition"] = "structure_original"
    frame["Structure"] = "original"
    frame["Aliased_From"] = "visual_r1_c1"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)

    source_metadata = source.with_suffix(".metadata.json")
    metadata = {}
    if source_metadata.exists():
        metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
    metadata.update(
        condition=asdict(condition),
        aliased_from="visual_r1_c1",
        template=content_type,
    )
    destination.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Reused visual_r1_c1 as {destination}")
    return True


def _runner_args(args: argparse.Namespace, condition, content_type: str) -> argparse.Namespace:
    common = {
        "model": args.model,
        "experiment": condition.experiment,
        "condition": condition.condition_id,
        "content_type": content_type,
        "start": args.start,
        "end": args.end,
        "dataset_path": args.dataset_path,
        "dataset_image_dir": args.dataset_image_dir,
        "ablation_image_dir": args.ablation_image_dir,
        "output_dir": (
            args.recognition_output_dir if condition.recognition else args.behavior_output_dir
        ),
        "checkpoint_every": args.checkpoint_every,
        "overwrite": args.overwrite,
    }
    if args.backend == "local":
        common["max_new_tokens"] = args.max_tokens
    else:
        common.update(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )
    return argparse.Namespace(**common)


def main() -> None:
    args = parse_args()
    conditions = selected_conditions(args.experiments)
    if args.list_only:
        for condition in conditions:
            print(asdict(condition))
        print(f"{len(conditions)} conditions x {len(args.types)} templates")
        return

    dataset = pd.read_csv(args.dataset_path)
    if args.backend == "local":
        from experiments.local_ablation import LocalAblationRunner
        from utils.helper import get_local_backend

        backend = get_local_backend(args.model)
        for condition in conditions:
            for content_type in args.types:
                if _reuse_original_visual_anchor(args, condition, content_type):
                    continue
                LocalAblationRunner(
                    _runner_args(args, condition, content_type),
                    backend=backend,
                    dataset=dataset,
                ).run()
    else:
        from experiments.openrouter_ablation import OpenRouterAblationRunner

        for condition in conditions:
            for content_type in args.types:
                if _reuse_original_visual_anchor(args, condition, content_type):
                    continue
                OpenRouterAblationRunner(_runner_args(args, condition, content_type)).run()


if __name__ == "__main__":
    main()
