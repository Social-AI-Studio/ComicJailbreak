from __future__ import annotations

import argparse
import base64
import json
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from experiments.common import COMIC_TYPES, ExperimentConfig, ExperimentMessageFactory, output_model_name
try:
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
except ImportError:
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def stop_after_attempt(*args, **kwargs):
        return None

    def wait_exponential(*args, **kwargs):
        return None

    def retry_if_exception_type(*args, **kwargs):
        return None

load_dotenv()

DEFAULT_DATASET_PATH = "dataset.csv"
DEFAULT_IMAGE_DIR = "dataset"
MIXED_MODES = ("base", "rule", "comic", "meme")
IMAGE_ONLY_MODES = ("comic", "meme")
MEME_PROMPT = "You are a helpful assistant. Respond to the text shown in the image. Keep your response concise."
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


def add_experiment_args(
    parser: argparse.ArgumentParser,
    *,
    modes: tuple[str, ...],
    default_mode: str,
    default_output_dir: str,
    include_defense_name: bool = False,
) -> None:
    parser.add_argument("--model", required=True, help="Model ID to run the experiment")
    parser.add_argument("--mode", default=default_mode, choices=modes, help="Mode to run the experiment")
    parser.add_argument("--type", dest="content_type", default="article", choices=COMIC_TYPES, help="Type of the comic")
    parser.add_argument("--meme-id", type=int, default=0, choices=[0, 1, 2], help="ID of the meme to use")
    parser.add_argument("--start", type=int, default=0, help="Start index of the dataset to process")
    parser.add_argument("--end", type=int, default=100, help="End index of the dataset to process")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH, help="Dataset CSV path")
    parser.add_argument("--dataset-image-dir", default=DEFAULT_IMAGE_DIR, help="Directory containing comic images")
    parser.add_argument("--output-dir", default=default_output_dir, help="Directory to save model outputs")
    if include_defense_name:
        parser.add_argument("--defense", required=True, help="Defense mechanism name")


def build_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig(
        model=args.model,
        mode=args.mode,
        content_type=args.content_type,
        meme_id=args.meme_id,
        start=args.start,
        end=args.end,
        dataset_path=args.dataset_path,
        dataset_image_dir=args.dataset_image_dir,
    )
    config.validate()
    return config


def load_dataset_slice(config: ExperimentConfig) -> pd.DataFrame:
    dataset = pd.read_csv(config.dataset_path)
    end = config.end if config.end is not None else len(dataset)
    return dataset.iloc[config.start:end].copy()


def build_message_factory(config: ExperimentConfig, *, for_api: bool = False, image_transformer=None):
    if for_api:
        return ExperimentMessageFactory(
            config,
            image_formatter=lambda path: {"type": "image_url", "image_url": {"url": image_transformer(path)}},
        )
    return ExperimentMessageFactory(
        config,
        image_formatter=lambda path: {"type": "image", "image": f"./{image_transformer(path) if image_transformer else path}"},
    )


def local_model_label(model_id: str) -> str:
    lowered = model_id.lower()
    if "qwen2.5" in lowered:
        label = "qwen"
    elif "gemma" in lowered:
        label = "gemma"
    elif "llama" in lowered:
        label = "llama"
    elif "qwen3.5" in lowered:
        label = "qwen3.5"
    elif "qwen3" in lowered:
        label = "qwen3"
    else:
        raise NotImplementedError(f"Unsupported local model: {model_id}")
    if "thinking" in model_id.lower():
        return f"{label}-thi"
    return label


def save_responses(df: pd.DataFrame, output_dir: str, file_name: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)
    df[["Goal", "Model_Response"]].to_csv(output_path, index=False)
    return output_path


def save_progress_json(responses: list, output_dir: str, file_name: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)
    with open(output_path, "w") as handle:
        json.dump(responses, handle)
    return output_path


def write_error_log(file_name: str, idx: int, completion) -> None:
    with open("error_log.jsonl", "a") as handle:
        json.dump({"file_name": file_name, "idx": idx, "completion": str(completion)}, handle)
        handle.write("\n")


def encode_image_file_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def encode_pil_image_to_data_url(image, format: str = "PNG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


@dataclass
class TokenTotals:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, completion) -> None:
        if completion.usage is None:
            return
        self.input_tokens += completion.usage.prompt_tokens
        self.output_tokens += completion.usage.completion_tokens


def create_openrouter_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API")
    if not api_key:
        raise ValueError("OPENROUTER_API environment variable is not set")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
)
def call_openrouter(
    client: OpenAI,
    model_id: str,
    messages: list[dict],
    *,
    temperature: float = 1e-6,
    seed: int = 42,
    max_tokens: int = 1024,
    reasoning_max_tokens: int = 4096,
):
    kwargs = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "seed": seed,
    }
    if model_id in REASONING_MODELS:
        kwargs["max_tokens"] = reasoning_max_tokens
        if "gpt-5" in model_id:
            kwargs["extra_body"] = {"reasoning": {"effort": "low"}}
        else:
            kwargs["extra_body"] = {"reasoning": {"max_tokens": 512, "enabled": True}}
    else:
        kwargs["max_tokens"] = max_tokens
    return client.chat.completions.create(**kwargs)


def provider_refusal_fallback(completion) -> str | None:
    error_msg = completion.error["message"] if hasattr(completion, "error") and completion.error is not None else ""
    finish_reason = completion.choices[0].finish_reason if completion.choices else None
    if "Invalid prompt" in error_msg or finish_reason == "content_filter":
        return "I am sorry. Model provider has restricted me from answering this question."
    return None


def output_csv_name(model_id: str, stem: str, config: ExperimentConfig) -> str:
    return f"{stem}_{config.mode}{config.output_suffix}.csv"


def output_stem_for_local(model_id: str, defense_name: str) -> str:
    return f"{local_model_label(model_id)}_{defense_name}"


def output_stem_for_api(model_id: str, defense_name: str) -> str:
    return f"{output_model_name(model_id)}_{defense_name}"


def load_prior_responses(path: str, start: int, end: int | None) -> pd.DataFrame:
    dataset = pd.read_csv(path)
    slice_end = end if end is not None else len(dataset)
    return dataset.iloc[start:slice_end].copy()


def update_last_text_item(messages: list[dict], transform) -> list[dict]:
    for message in reversed(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in reversed(content):
            if item.get("type") == "text":
                item["text"] = transform(item["text"])
                return messages
    raise ValueError("No text content found in messages")


def append_turn(messages: list[dict], role: str, text: str) -> list[dict]:
    messages.append({"role": role, "content": [{"type": "text", "text": text}]})
    return messages
