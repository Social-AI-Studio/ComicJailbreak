#!/bin/bash

###################################
# ATTACK SCRIPT FOR API INFERENCE #
###################################

### EDIT HERE ###
model="openai/gpt-5-mini"
DATASET_PATH="dataset.csv"
OUTPUT_DIR="model_responses"

# Run base and rule attack
START=0
END=2

########################
# RUN TEXT-ONLY ATTACK #
########################
for mode in "base" "rule"
do
    uv run python -m experiments.openrouter --model $model --mode $mode --dataset-path $DATASET_PATH --output-dir $OUTPUT_DIR --start $START --end $END
done

#######################
# RUN MEME-IMG ATTACK #
#######################

for meme_id in 0 1 2
do
    uv run python -m experiments.openrouter --model $model --mode "meme" --meme-id $meme_id --dataset-path $DATASET_PATH --output-dir $OUTPUT_DIR --dataset-image-dir "meme_images" --start $START --end $END
done

####################
# RUN COMIC ATTACK #
####################

for type in "article" "speech" "instruction" "message" "code"
do
    uv run python -m experiments.openrouter --model $model --mode "comic" --type $type --dataset-path $DATASET_PATH --output-dir $OUTPUT_DIR --dataset-image-dir "dataset" --start $START --end $END
done
