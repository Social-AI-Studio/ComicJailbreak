from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class EvaluationPaths:
    input_dir: str
    output_dir: str


def iter_csv_files(directory: str, file_contains: str = "") -> list[str]:
    files = sorted(name for name in os.listdir(directory) if name.endswith(".csv"))
    if file_contains:
        files = [name for name in files if file_contains in name]
    return files


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clean_model_response(response) -> str | None:
    if pd.isna(response) or not isinstance(response, str):
        return None
    return response.split("</think>")[-1].strip()


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
