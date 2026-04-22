import argparse
import os
from dataclasses import dataclass
from time import time

from tqdm import tqdm

from experiments.common import (
    COMIC_TYPES,
    ExperimentConfig,
    ExperimentMessageFactory,
    MODES,
    load_dataset,
)
from utils.helper import get_local_backend


@dataclass
class LocalBackend:
    label: str
    model: object
    processor: object
    process_input: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model ID to run the experiment")
    parser.add_argument("--mode", default="base", choices=MODES, help="Mode to run the experiment")
    parser.add_argument("--type", dest="content_type", default="article", choices=COMIC_TYPES, help="Type of the comic")
    parser.add_argument("--meme-id", type=int, default=0, choices=[0, 1, 2], help="ID of the meme to use")
    parser.add_argument("--start", type=int, default=0, help="Start index of the dataset to process")
    parser.add_argument("--end", type=int, default=100, help="End index of the dataset to process")
    parser.add_argument("--dataset-path", default="dataset.csv", help="Dataset CSV path")
    parser.add_argument("--dataset-image-dir", default="dataset", help="Directory containing comic images")
    parser.add_argument("--output-dir", default="model_responses", help="Directory to save model outputs")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Maximum tokens to generate per sample")
    return parser.parse_args()


class LocalExperimentRunner:
    def __init__(self, args: argparse.Namespace):
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
        self.max_new_tokens = args.max_new_tokens
        self.dataset = load_dataset(self.config)
        self.message_factory = ExperimentMessageFactory(
            self.config,
            image_formatter=lambda path: {"type": "image", "image": f"./{path}"},
        )
        self.backend = self._load_backend(self.config.model)

    def _load_backend(self, model_id: str) -> LocalBackend:
        backend = get_local_backend(model_id)
        label = backend.family
        if "thinking" in model_id.lower():
            label = f"{label}-thi"
        return LocalBackend(
            label=label,
            model=backend.model,
            processor=backend.processor,
            process_input=backend.process_input,
        )

    def run(self) -> None:
        print("Dataset:", self.config.dataset_image_dir)
        print(f"Model: {self.config.model}")
        print(
            f"Running in {self.config.mode} mode"
            f"{f' with type {self.config.content_type}' if self.config.mode in {'comic', 'text', 'base_comic'} else ''}"
            f"{f' and meme ID {self.config.meme_id}' if self.config.mode == 'meme' else ''}."
        )

        model_responses = []
        start_time = time()

        for idx, row in tqdm(self.dataset.iterrows(), total=self.dataset.shape[0]):
            messages = self.message_factory.build(idx, row)
            if messages is None:
                print(f"Skipping idx {idx}.")
                model_responses.append(None)
                continue

            inputs = self.backend.process_input(self.backend.processor, messages)
            input_len = inputs["input_ids"].shape[-1]
            output = self.backend.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
            generated_tokens = output[0][input_len:]
            output_text = self.backend.processor.decode(generated_tokens, skip_special_tokens=True)
            model_responses.append(output_text)

        elapsed = time() - start_time
        print(f"Total time: {elapsed} seconds")

        self.dataset["Model_Response"] = model_responses
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(
            self.output_dir,
            f"{self.backend.label}_{self.config.mode}{self.config.output_suffix}.csv",
        )
        self.dataset[["Goal", "Model_Response"]].to_csv(output_path, index=False)
        print(f"Saved results to {output_path}")


def main() -> None:
    runner = LocalExperimentRunner(parse_args())
    runner.run()


if __name__ == "__main__":
    main()
