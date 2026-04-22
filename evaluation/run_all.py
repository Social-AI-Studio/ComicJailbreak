import argparse
from time import time

from evaluation.beavertails.evaluate import BeaverTailsEvaluator
from evaluation.common import ensure_output_dir, iter_csv_files
from evaluation.harmbench import HarmBenchEvaluator
from evaluation.strongreject import StrongRejectEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="model_responses", help="Directory containing CSV files to evaluate")
    parser.add_argument("--output-dir", default="evaluated", help="Directory to save evaluated CSV files")
    parser.add_argument("--file-contains", default="", help="Only process files whose names contain this substring")
    parser.add_argument("--overwrite", action="store_true", help="Recompute scores even if output columns already exist")
    parser.add_argument("--skip-strongreject", action="store_true", help="Skip StrongReject scoring")
    parser.add_argument("--skip-harmbench", action="store_true", help="Skip HarmBench scoring")
    parser.add_argument("--skip-beavertails", action="store_true", help="Skip BeaverTails scoring")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)

    evaluators = []
    if not args.skip_strongreject:
        evaluators.append(("StrongReject", StrongRejectEvaluator()))
    if not args.skip_harmbench:
        evaluators.append(("HarmBench", HarmBenchEvaluator()))
    if not args.skip_beavertails:
        evaluators.append(("BeaverTails", BeaverTailsEvaluator()))

    start_time = time()
    files = iter_csv_files(args.input_dir, args.file_contains)

    for file_name in files:
        print(f"Evaluating file: {file_name}")
        current_input_path = f"{args.input_dir}/{file_name}"
        output_path = f"{args.output_dir}/{file_name}"

        for evaluator_name, evaluator in evaluators:
            print(f"  -> {evaluator_name}")
            evaluator.evaluate_file(
                current_input_path,
                output_path,
                overwrite=args.overwrite,
            )
            current_input_path = output_path

    print(f"Total evaluation time: {time() - start_time} seconds")


if __name__ == "__main__":
    main()
