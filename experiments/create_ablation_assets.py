from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from create_dataset import PLACEMENTS, create_image
from experiments.ablation import STRUCTURES, ablation_image_path, original_image_path, type_value
from experiments.common import COMIC_TYPES

PANEL_ORDERS = {
    "reversed": (1, 0, 2),
    "repeat_p1": (0, 0, 2),
    "repeat_p2": (1, 1, 2),
}
PANEL_PREFIX = {
    "article": "art",
    "speech": "spe",
    "instruction": "ins",
    "message": "msg",
    "code": "cod",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive visual-structure ablations from the original rendered three-panel comics."
    )
    parser.add_argument("--dataset-path", default="dataset.csv")
    parser.add_argument("--dataset-image-dir", default="dataset")
    parser.add_argument("--output-dir", default="ablation_images")
    parser.add_argument("--panel-dir", default="template_panels")
    parser.add_argument(
        "--source-mode",
        choices=("panels", "rendered"),
        default="rendered",
        help="Split rendered originals (recommended), or compose from local individual panel assets.",
    )
    parser.add_argument("--type", dest="content_type", choices=(*COMIC_TYPES, "all"), default="all")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def split_panels(image: Image.Image) -> tuple[Image.Image, Image.Image, Image.Image]:
    width, height = image.size
    if width % 3:
        raise ValueError(f"Comic width must be divisible by three, got {width}")
    panel_width = width // 3
    return tuple(
        image.crop((panel_width * index, 0, panel_width * (index + 1), height))
        for index in range(3)
    )


def build_structure(image: Image.Image, structure: str) -> Image.Image:
    if structure not in STRUCTURES:
        raise ValueError(f"Unknown structure: {structure}")
    if structure == "original":
        return image.copy()

    panels = split_panels(image)
    if structure == "goal_only":
        return panels[2].copy()

    order = PANEL_ORDERS[structure]
    output = Image.new(image.mode, image.size)
    panel_width = image.width // 3
    for position, source_index in enumerate(order):
        output.paste(panels[source_index], (position * panel_width, 0))
    return output


def _panel_tile(path: Path) -> Image.Image:
    with Image.open(path) as panel:
        resized = panel.convert("RGB").resize((496, 752), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (512, 768), "white")
    tile.paste(resized, (8, 8))
    return tile


def compose_panel_template(panel_dir: str | Path, content_type: str, structure: str) -> Image.Image:
    if structure not in STRUCTURES:
        raise ValueError(f"Unknown structure: {structure}")
    prefix = PANEL_PREFIX[content_type]
    order = PANEL_ORDERS.get(structure, (0, 1, 2))
    tiles = []
    for source_index in order:
        path = Path(panel_dir) / f"{prefix}_{source_index + 1}.png"
        if not path.exists():
            raise FileNotFoundError(f"Missing individual panel asset: {path}")
        tiles.append(_panel_tile(path))
    output = Image.new("RGB", (1536, 768), "white")
    for position, tile in enumerate(tiles):
        output.paste(tile, (position * 512, 0))
    return output


def generate_assets(
    dataset: pd.DataFrame,
    dataset_image_dir: str,
    output_dir: str,
    content_types: tuple[str, ...],
    *,
    start: int = 0,
    end: int | None = None,
    overwrite: bool = False,
) -> tuple[int, int]:
    end = len(dataset) if end is None else min(end, len(dataset))
    created = 0
    skipped = 0
    work = [(idx, content_type) for idx in range(start, end) for content_type in content_types]
    for idx, content_type in tqdm(work, desc="Building ablation images"):
        row = dataset.iloc[idx]
        if type_value(row, content_type) is None:
            continue
        source_path = original_image_path(dataset_image_dir, content_type, idx)
        if not source_path.exists():
            raise FileNotFoundError(
                f"Missing original comic {source_path}. Supply the original rendered dataset assets first."
            )
        with Image.open(source_path) as source:
            source.load()
            for structure in STRUCTURES:
                if structure == "original":
                    continue
                destination = ablation_image_path(
                    dataset_image_dir,
                    output_dir,
                    content_type,
                    idx,
                    structure,
                )
                if destination.exists() and not overwrite:
                    skipped += 1
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                build_structure(source, structure).save(destination, format="PNG")
                created += 1
    return created, skipped


def generate_assets_from_panels(
    dataset: pd.DataFrame,
    dataset_image_dir: str,
    output_dir: str,
    panel_dir: str,
    content_types: tuple[str, ...],
    *,
    start: int = 0,
    end: int | None = None,
    overwrite: bool = False,
) -> tuple[int, int]:
    end = len(dataset) if end is None else min(end, len(dataset))
    created = 0
    skipped = 0
    work = [(idx, content_type) for idx in range(start, end) for content_type in content_types]
    for idx, content_type in tqdm(work, desc="Building ablation images from panels"):
        goal = type_value(dataset.iloc[idx], content_type)
        if goal is None:
            continue
        for structure in STRUCTURES:
            if structure == "original":
                continue
            destination = ablation_image_path(
                dataset_image_dir,
                output_dir,
                content_type,
                idx,
                structure,
            )
            if destination.exists() and not overwrite:
                skipped += 1
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            template = compose_panel_template(panel_dir, content_type, structure)
            create_image(
                PLACEMENTS[content_type],
                template,
                goal,
                str(destination.with_suffix("")),
            )
            if structure == "goal_only":
                with Image.open(destination) as rendered:
                    cropped = rendered.crop((1024, 0, 1536, 768))
                    cropped.save(destination, format="PNG")
            created += 1
    return created, skipped


def main() -> None:
    args = parse_args()
    if args.start < 0 or (args.end is not None and args.end <= args.start):
        raise ValueError("Expected 0 <= start < end")
    dataset = pd.read_csv(args.dataset_path)
    content_types = COMIC_TYPES if args.content_type == "all" else (args.content_type,)
    if args.source_mode == "panels":
        created, skipped = generate_assets_from_panels(
            dataset,
            args.dataset_image_dir,
            args.output_dir,
            args.panel_dir,
            content_types,
            start=args.start,
            end=args.end,
            overwrite=args.overwrite,
        )
    else:
        created, skipped = generate_assets(
            dataset,
            args.dataset_image_dir,
            args.output_dir,
            content_types,
            start=args.start,
            end=args.end,
            overwrite=args.overwrite,
        )
    print(f"Created {created} images; skipped {skipped} existing images.")


if __name__ == "__main__":
    main()
