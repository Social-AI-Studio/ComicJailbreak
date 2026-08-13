from __future__ import annotations

import argparse
import base64
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from experiments.ablation import (
    EXPERIMENTS,
    AblationMessageFactory,
    CATEGORY_TO_ENUM,
    condition_manifest,
    conditions_for,
    get_condition,
    parse_recognition,
    type_value,
)
from experiments.common import COMIC_TYPES

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API")
REASONING_MODELS = {
    "qwen/qwen3-vl-235b-a22b-thinking",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "openai/gpt-5-mini",
    "openai/gpt-5",
    "openai/gpt-5.1",
    "google/gemini-3-pro-preview",
    "moonshotai/kimi-k2.5",
    "qwen/qwen3.5-397b-a17b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one preregistered ComicJailbreak ablation condition.")
    parser.add_argument("--model", required=True, help="OpenRouter model ID")
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    parser.add_argument("--condition", required=True, help="Condition ID within the selected experiment")
    parser.add_argument("--type", dest="content_type", default="article", choices=COMIC_TYPES)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--dataset-path", default="dataset.csv")
    parser.add_argument("--dataset-image-dir", default="dataset")
    parser.add_argument("--ablation-image-dir", default="ablation_images")
    parser.add_argument("--output-dir", default="ablation_responses")
    parser.add_argument("--temperature", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true", help="Replace, rather than resume, an output file")
    return parser.parse_args()


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", value).strip("-")


def output_path(output_dir: str, model: str, condition_id: str, content_type: str) -> Path:
    return Path(output_dir) / f"{_slug(model)}__{condition_id}__{content_type}.csv"


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    frame.sort_values("Seed_Index").to_csv(temporary, index=False)
    temporary.replace(path)


@dataclass
class OpenRouterTotals:
    input_tokens: int = 0
    output_tokens: int = 0


class OpenRouterAblationRunner:
    def __init__(self, args: argparse.Namespace):
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API environment variable is not set")
        self.args = args
        self.condition = get_condition(args.condition)
        if self.condition.experiment != args.experiment:
            raise ValueError(
                f"Condition {args.condition} belongs to {self.condition.experiment}, not {args.experiment}"
            )
        if args.start < 0 or (args.end is not None and args.end <= args.start):
            raise ValueError("Expected 0 <= start < end")
        if args.checkpoint_every < 1:
            raise ValueError("checkpoint-every must be positive")

        self.dataset = pd.read_csv(args.dataset_path)
        self.end = len(self.dataset) if args.end is None else min(args.end, len(self.dataset))
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        self.factory = AblationMessageFactory(
            self.condition,
            args.content_type,
            self._format_image,
            dataset_image_dir=args.dataset_image_dir,
            ablation_image_dir=args.ablation_image_dir,
        )
        self.totals = OpenRouterTotals()
        self.path = output_path(args.output_dir, args.model, args.condition, args.content_type)
        self.records = self._existing_records()

    def _existing_records(self) -> dict[int, dict]:
        if self.args.overwrite or not self.path.exists():
            return {}
        existing = pd.read_csv(self.path).to_dict(orient="records")
        for row in existing:
            if row.get("Model") != self.args.model or row.get("Condition") != self.args.condition:
                raise ValueError(f"Existing output metadata does not match this run: {self.path}")
            if (
                float(row.get("Temperature")) != self.args.temperature
                or int(row.get("Seed")) != self.args.seed
                or int(row.get("Max_Tokens")) != self.args.max_tokens
            ):
                raise ValueError(
                    f"Existing inference settings do not match this run: {self.path}. "
                    "Use the original settings or pass --overwrite."
                )
        return {int(row["Seed_Index"]): row for row in existing}

    @staticmethod
    def _format_image(path: str) -> dict:
        with open(path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")
        return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}

    def _create_completion(self, messages: list[dict]):
        max_tokens = max(self.args.max_tokens, 4096) if self.args.model in REASONING_MODELS else self.args.max_tokens
        kwargs = {
            "model": self.args.model,
            "messages": messages,
            "temperature": self.args.temperature,
            "seed": self.args.seed,
            "max_tokens": max_tokens,
        }
        if self.args.model in REASONING_MODELS:
            kwargs["extra_body"] = {"reasoning": {"enabled": True}}
        return self.client.chat.completions.create(**kwargs)

    def _base_record(self, idx: int, row: pd.Series, available: bool) -> dict:
        expected_harmful = bool(pd.notna(row.get("Behavior")))
        expected_category = (
            CATEGORY_TO_ENUM[str(row["Category"])].value if expected_harmful else "benign"
        )
        return {
            "Seed_Index": idx,
            "Goal": row["Goal"],
            "Stimulus": type_value(row, self.args.content_type),
            "Template": self.args.content_type,
            "Is_Harmful": expected_harmful,
            "Expected_Category": expected_category,
            "Model": self.args.model,
            "Resolved_Model": None,
            "Completion_ID": None,
            "System_Fingerprint": None,
            "Aliased_From": None,
            "Experiment": self.condition.experiment,
            "Condition": self.condition.condition_id,
            "Narrative": self.condition.narrative,
            "Role": self.condition.role,
            "Completion": self.condition.completion,
            "Structure": self.condition.structure,
            "Input_Available": available,
            "Temperature": self.args.temperature,
            "Seed": self.args.seed,
            "Max_Tokens": self.args.max_tokens,
            "Model_Response": None,
            "Raw_Response": None,
            "Request_Status": "pending" if available else "incompatible",
            "Error": None,
            "Recognition_Harmful": None,
            "Recognition_Category": None,
            "Recognition_Parse_Status": None,
            "Recognition_Requires_Review": None,
        }

    def _save(self) -> None:
        _atomic_write_csv(pd.DataFrame(self.records.values()), self.path)

    def run(self) -> None:
        print(
            f"Model={self.args.model} condition={self.condition.condition_id} "
            f"template={self.args.content_type} rows={self.args.start}:{self.end}"
        )
        completed_since_checkpoint = 0
        for idx in tqdm(range(self.args.start, self.end)):
            previous = self.records.get(idx)
            if previous and previous.get("Request_Status") in {"ok", "incompatible"}:
                continue
            row = self.dataset.iloc[idx]
            available = type_value(row, self.args.content_type) is not None
            record = self._base_record(idx, row, available)
            if not available:
                self.records[idx] = record
                continue

            try:
                messages = self.factory.build(idx, row)
                assert messages is not None
                completion = self._create_completion(messages)
                raw = completion.choices[0].message.content
                record["Model_Response"] = raw
                record["Raw_Response"] = raw
                record["Request_Status"] = "ok"
                record["Resolved_Model"] = getattr(completion, "model", None)
                record["Completion_ID"] = getattr(completion, "id", None)
                record["System_Fingerprint"] = getattr(completion, "system_fingerprint", None)
                if completion.usage is not None:
                    self.totals.input_tokens += completion.usage.prompt_tokens
                    self.totals.output_tokens += completion.usage.completion_tokens
                if self.condition.recognition:
                    parsed = parse_recognition(raw)
                    record["Recognition_Harmful"] = parsed.harmful
                    record["Recognition_Category"] = (
                        parsed.category.value if parsed.category is not None else None
                    )
                    record["Recognition_Parse_Status"] = parsed.status
                    record["Recognition_Requires_Review"] = parsed.requires_review
            except FileNotFoundError:
                raise
            except Exception as exc:  # API failures are retained and retried by a resumed run.
                record["Request_Status"] = "error"
                record["Error"] = f"{type(exc).__name__}: {exc}"

            self.records[idx] = record
            completed_since_checkpoint += 1
            if completed_since_checkpoint >= self.args.checkpoint_every:
                self._save()
                completed_since_checkpoint = 0

        self._save()
        metadata_path = self.path.with_suffix(".metadata.json")
        metadata = {
            "model": self.args.model,
            "condition": asdict(self.condition),
            "template": self.args.content_type,
            "dataset_path": self.args.dataset_path,
            "dataset_image_dir": self.args.dataset_image_dir,
            "ablation_image_dir": self.args.ablation_image_dir,
            "temperature": self.args.temperature,
            "seed": self.args.seed,
            "max_tokens": self.args.max_tokens,
            "totals_this_invocation": asdict(self.totals),
            "condition_registry": condition_manifest(conditions_for(self.args.experiment)),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Tokens: input={self.totals.input_tokens}, output={self.totals.output_tokens}")
        print(f"Saved {len(self.records)} rows to {self.path}")


def main() -> None:
    OpenRouterAblationRunner(parse_args()).run()


if __name__ == "__main__":
    main()
