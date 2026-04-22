#!/bin/bash

#####################
# EVALUATION SCRIPT #
#####################

# Change the paths below
INPUT_DIR="model_responses"
OUTPUT_DIR="evaluated"

uv run python -m evaluation.run_all --input-dir $INPUT_DIR --output-dir $OUTPUT_DIR

# Change the paths below
INPUT_DIR="model_responses/defense"
OUTPUT_DIR="evaluated/defense"

uv run python -m evaluation.run_all --input-dir $INPUT_DIR --output-dir $OUTPUT_DIR