#!/bin/bash

####################################
# DEFENSE SCRIPT FOR API INFERENCE #
####################################

### EDIT MODEL NAME HERE ###
model="openai/gpt-5-mini"
DATASET_PATH="dataset.csv"
OUTPUT_DIR="model_responses/defense"

START=0
END=2

# ###################
# # RUN ASD DEFENSE #
# ###################

# # Run comic attack
# for type in "article" "speech" "instruction" "message" "code"
# do
#     uv run python -m defense.openrouter_asd --model $model --mode "comic" --type $type --start $START --end $END --dataset-path $DATASET_PATH --output-dir $OUTPUT_DIR --dataset-image-dir "dataset" 
# done

# # Run random_image attack
# for meme_id in 0 1 2
# do
#     uv run python -m defense.openrouter_asd --model $model --mode "meme" --meme-id $meme_id --start $START --end $END --dataset-path $DATASET_PATH --output-dir $OUTPUT_DIR --dataset-image-dir "meme_images" 
# done

# ################################
# # RUN ADASHIELD-STATIC DEFENSE #
# ################################

# # Run base and rule attack
# for mode in "base" "rule"
# do
#     uv run python -m defense.openrouter_adashield --model $model --mode $mode --start $START --end $END --dataset-path $DATASET_PATH --output-dir $OUTPUT_DIR 
# done

# # Run comic attack
# for type in "article" "speech" "instruction" "message" "code"
# do
#     uv run python -m defense.openrouter_adashield --model $model --mode "comic" --type $type --start $START --end $END --dataset-path $DATASET_PATH --output-dir $OUTPUT_DIR --dataset-image-dir "dataset" 
# done

# # Run random_image attack
# for meme_id in 2
# do
#     uv run python -m defense.openrouter_adashield --model $model --mode "meme" --meme-id $meme_id --start $START --end $END --dataset-path $DATASET_PATH --output-dir $OUTPUT_DIR --dataset-image-dir "meme_images" 
# done

##########################
# RUN REFLECTION DEFENSE #
##########################

# # Run base and rule attack
# for mode in "base" "rule"
# do
#     uv run python -m defense.openrouter_reflection --model $model --mode $mode --start $START --end $END --dataset-path $DATASET_PATH --output-dir $OUTPUT_DIR --input-dir "model_responses"
# done

# # Run comic attack
# for type in "article" "speech" "instruction" "message" "code"
# do
#     uv run python -m defense.openrouter_reflection --model $model --mode "comic" --type $type --start $START --end $END --dataset-path $DATASET_PATH --output-dir $OUTPUT_DIR --dataset-image-dir "dataset" --input-dir "model_responses"
# done

# Run random_image attack
for meme_id in 2
do
    uv run python -m defense.openrouter_reflection --model $model --mode "meme" --meme-id $meme_id --start $START --end $END --dataset-path $DATASET_PATH --output-dir $OUTPUT_DIR --dataset-image-dir "meme_images" --input-dir "model_responses"
done