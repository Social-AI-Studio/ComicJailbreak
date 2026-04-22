#!/bin/bash

#####################################
# ATTACK SCRIPT FOR LOCAL INFERENCE #
#####################################

### EDIT HERE ###
model="Qwen/Qwen2.5-VL-7B-Instruct"
DATASET_PATH="dataset.csv"
OUTPUT_DIR="model_responses"

# Run base and rule attack
START=0
END=300

########################
# RUN TEXT-ONLY ATTACK #
########################
for mode in "base" "rule"
do
    uv run python -m experiments.local --model $model --mode $mode --dataset-path $DATASET_PATH --output-dir $OUTPUT_DIR --start $START --end $END
done

#######################
# RUN MEME-IMG ATTACK #
#######################

for meme_id in 0 1 2
do
    uv run python -m experiments.local --model $model --mode "meme" --meme-id $meme_id --dataset-path $DATASET_PATH --dataset-image-dir "meme_images" --output-dir $OUTPUT_DIR --start $START --end $END
done

####################
# RUN COMIC ATTACK #
####################

for type in "article" "speech" "instruction" "message" "code"
do
    uv run python -m experiments.local --model $model --mode "comic" --type $type --dataset-path $DATASET_PATH --dataset-image-dir "dataset" --output-dir $OUTPUT_DIR --start $START --end $END
done
