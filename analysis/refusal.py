import argparse
import json
import os
import pprint

import pandas as pd
from tabulate import tabulate

from analysis.common import (
    DEFAULT_MODELS,
    DEFENSE_DISPLAY,
    DEFENSE_METHODS,
    MODEL_MAPPING,
    ensemble_binary_score,
    extract_model_and_type,
    format_table_rows,
    list_csv_files,
    merge_with_dataset,
    refusal_label,
    render_plain_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="dataset.csv", help="Path to the combined dataset")
    parser.add_argument("--base-dir", default="evaluated", help="Directory containing base evaluated CSV files")
    parser.add_argument("--defense-dir", default="evaluated/defense", help="Directory containing defense evaluated CSV files")
    parser.add_argument("--mode", choices=["table", "defense"], default="table", help="Output style to generate")
    parser.add_argument("--json-out", default="err_results.json", help="JSON output used in defense mode")
    return parser.parse_args()


def compute_table_metrics(dataset: pd.DataFrame, directory: str):
    rr = {}
    err_inputs = {}

    for file_name in list_csv_files(directory, include_text=True):
        model, metric_type = extract_model_and_type(file_name)
        merged = merge_with_dataset(dataset, os.path.join(directory, file_name)).iloc[100:200].copy()
        merged["Reject"] = merged.apply(refusal_label, axis=1)

        rr.setdefault(model, {})
        err_inputs.setdefault(model, pd.DataFrame())

        if "_comic_" in file_name or "_text_" in file_name:
            err_inputs[model][metric_type] = merged["Reject"].tolist()
            merged = merged.dropna(subset=[metric_type.capitalize()])

        rr[model][metric_type] = round((merged["Reject"] == "refusal").sum() / 100 * 100, 1)

    err_scores = {
        model: ensemble_binary_score(frame, "refusal")
        for model, frame in err_inputs.items()
        if not frame.empty
    }
    return rr, err_scores


def run_table_mode(args: argparse.Namespace) -> None:
    dataset = pd.read_csv(args.dataset_path)
    rr, err_scores = compute_table_metrics(dataset, args.base_dir)
    table = format_table_rows(rr, err_scores, models=DEFAULT_MODELS)
    headers = ["Model", "base", "rule", "meme0", "meme1", "meme2", "art.", "code", "ins.", "mes.", "spe.", "comic"]
    if tabulate is None:
        print(render_plain_table(headers, table))
    else:
        print(tabulate(table, headers=headers, tablefmt="latex", floatfmt=".1f"))


def run_defense_mode(args: argparse.Namespace) -> None:
    dataset = pd.read_csv(args.dataset_path)
    final_rr = {}
    final_err = {}
    err_list = {}

    for method in DEFENSE_METHODS:
        directory = args.base_dir if method == "" else args.defense_dir
        method_rr = {}
        method_err_inputs = {}

        for file_name in list_csv_files(directory, method=method, comic_only=True):
            model, metric_type = extract_model_and_type(file_name)
            if model not in MODEL_MAPPING:
                continue

            merged = merge_with_dataset(dataset, os.path.join(directory, file_name)).iloc[100:200].copy()
            merged["Reject"] = merged.apply(refusal_label, axis=1)

            method_rr.setdefault(model, {})
            method_err_inputs.setdefault(model, pd.DataFrame())
            method_err_inputs[model][metric_type] = merged["Reject"].tolist()
            merged = merged.dropna(subset=[metric_type.capitalize()])
            method_rr[model][metric_type] = round((merged["Reject"] == "refusal").sum() / 100 * 100, 1)

        err_scores = {}
        for model, frame in method_err_inputs.items():
            if frame.empty:
                continue
            flags = frame.apply(lambda row: int("refusal" in row.values), axis=1)
            err_list.setdefault(MODEL_MAPPING[model], {})[DEFENSE_DISPLAY[method]] = flags.tolist()
            err_scores[model] = round(flags.mean() * 100, 1)

        final_rr[DEFENSE_DISPLAY[method]] = {
            MODEL_MAPPING[model]: scores
            for model, scores in method_rr.items()
            if model in MODEL_MAPPING
        }
        final_err[DEFENSE_DISPLAY[method]] = {
            MODEL_MAPPING[model]: score
            for model, score in err_scores.items()
            if model in MODEL_MAPPING
        }

    with open(args.json_out, "w") as handle:
        json.dump(err_list, handle, indent=4)

    pprint.pprint(final_rr)
    print("\n")
    pprint.pprint(final_err)


def main() -> None:
    args = parse_args()
    if args.mode == "defense":
        run_defense_mode(args)
    else:
        run_table_mode(args)


if __name__ == "__main__":
    main()
