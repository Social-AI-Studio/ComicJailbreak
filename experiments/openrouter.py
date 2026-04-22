import argparse
import base64
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from .common import (
    COMIC_TYPES,
    ExperimentConfig,
    ExperimentMessageFactory,
    MODES,
    load_dataset,
    output_model_name,
)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model ID to run the experiment")
    parser.add_argument("--mode", default="base", choices=MODES, help="Mode to run the experiment")
    parser.add_argument("--type", dest="content_type", default="article", choices=COMIC_TYPES, help="Type of the comic")
    parser.add_argument("--meme-id", type=int, default=0, choices=[0, 1, 2], help="ID of the meme to use")
    parser.add_argument("--start", type=int, default=0, help="Start index of the dataset to process")
    parser.add_argument("--end", type=int, default=None, help="End index of the dataset to process")
    parser.add_argument("--dataset-path", default="dataset.csv", help="Dataset CSV path")
    parser.add_argument("--dataset-image-dir", default="dataset", help="Directory containing comic images")
    parser.add_argument("--output-dir", default="model_responses", help="Directory to save model outputs")
    parser.add_argument("--temperature", type=float, default=1e-6, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed sent to OpenRouter")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum completion tokens")
    return parser.parse_args()


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Track total input and output tokens for the experiment
@dataclass
class OpenRouterTotals:
    input_tokens: int = 0
    output_tokens: int = 0


class OpenRouterExperimentRunner:
    def __init__(self, args: argparse.Namespace):
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API environment variable is not set")

        self.config = ExperimentConfig(
            model=args.model,
            mode=args.mode,
            content_type=args.content_type,
            meme_id=args.meme_id,
            start=args.start,
            end=args.end,
            dataset_path=args.dataset_path,
            dataset_image_dir=args.dataset_image_dir,
        )
        self.config.validate()
        self.output_dir = args.output_dir
        self.temperature = args.temperature
        self.seed = args.seed
        self.max_tokens = args.max_tokens
        # reasoning models may require more tokens, so we set a higher limit at minimum of 4096
        self.reasoning_max_tokens = max(args.max_tokens, 4096)
        self.totals = OpenRouterTotals()
        self.dataset = load_dataset(self.config)
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        self.message_factory = ExperimentMessageFactory(self.config, self._format_image)

    def _format_image(self, path: str) -> dict:
        base64_image = encode_image_to_base64(path)
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}

    def _create_completion(self, messages: list[dict]):
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.temperature,
            "seed": self.seed,
        }
        if self.config.model in REASONING_MODELS:
            kwargs["max_tokens"] = self.reasoning_max_tokens
            kwargs["extra_body"] = {"reasoning": {"enabled": True}}
        else:
            kwargs["max_tokens"] = self.max_tokens
        return self.client.chat.completions.create(**kwargs)

    def run(self) -> None:
        print(f"Model: {self.config.model}")
        print(
            f"Running in {self.config.mode} mode"
            f"{f' with type {self.config.content_type}' if self.config.mode in {'comic', 'text', 'base_comic'} else ''}"
            f"{f' and meme ID {self.config.meme_id}' if self.config.mode == 'meme' else ''}."
        )

        responses = []
        for idx, row in tqdm(self.dataset.iterrows(), total=self.dataset.shape[0]):
            messages = self.message_factory.build(idx, row)
            if messages is None:
                print(f"Skipping idx {idx}.")
                responses.append(None)
                continue

            try:
                completion = self._create_completion(messages)
                if completion.usage is not None:
                    self.totals.input_tokens += completion.usage.prompt_tokens
                    self.totals.output_tokens += completion.usage.completion_tokens
                responses.append(completion.choices[0].message.content)
            except Exception as exc:
                print(f"Error for row {idx}: {exc}")
                responses.append("Model Response Error")

        self.dataset["Model_Response"] = responses
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(
            self.output_dir,
            f"{output_model_name(self.config.model)}_{self.config.mode}{self.config.output_suffix}.csv",
        )
        self.dataset[["Goal", "Model_Response"]].to_csv(output_path, index=False)

        print(
            f"Total input tokens: {self.totals.input_tokens}, total output tokens: {self.totals.output_tokens}"
        )
        print(f"Saved results to {output_path}")


def main() -> None:
    runner = OpenRouterExperimentRunner(parse_args())
    runner.run()


if __name__ == "__main__":
    main()
