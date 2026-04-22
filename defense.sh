#!/bin/bash

######################################
# DEFENSE SCRIPT FOR LOCAL INFERENCE #
######################################

### EDIT HERE ###
model="Qwen/Qwen2.5-VL-7B-Instruct"
DATASET_PATH="dataset.csv"
OUTPUT_DIR="model_responses/defense"

START=0
END=2

###################
# RUN ASD DEFENSE #
###################

# Run comic attack
for type in "article" "speech" "instruction" "message" "code"
do
    uv run python -m defense.asd --model $model --mode "comic" --type $type --dataset-path $DATASET_PATH --output-dir $OUTPUT_DIR --dataset-image-dir "dataset" --start $START --end $END
done

# Run random_image attack
for meme_id in 0 1 2
do
    uv run python -m defense.asd --model $model --mode "meme" --meme-id $meme_id --dataset-path $DATASET_PATH --output-dir $OUTPUT_DIR --dataset-image-dir "meme_images" --start $START --end $END
done

################################
# RUN ADASHIELD-STATIC DEFENSE #
################################

# Run base and rule attack
for mode in "base" "rule"
do
    uv run python -m defense.adashield --model $model --mode $mode --dataset-path $DATASET_PATH --output-dir $OUTPUT_DIR --start $START --end $END
done

# Run comic attack
for type in "article" "speech" "instruction" "message" "code"
do
    uv run python -m defense.adashield --model $model --mode "comic" --type $type --dataset-path $DATASET_PATH --output-dir $OUTPUT_DIR --dataset-image-dir "dataset" --start $START --end $END
done

# Run random_image attack
for meme_id in 0 1 2
do
    uv run python -m defense.adashield --model $model --mode "meme" --meme-id $meme_id --dataset-path $DATASET_PATH --output-dir $OUTPUT_DIR --dataset-image-dir "meme_images" --start $START --end $END
done

##########################
# RUN REFLECTION DEFENSE #
##########################

# Run base and rule attack
for mode in "base" "rule"
do
    uv run python -m defense.reflection --model $model --mode $mode --dataset-path $DATASET_PATH --output-dir $OUTPUT_DIR --start $START --end $END
done

# Run comic attack
for type in "article" "speech" "instruction" "message" "code"
do
    uv run python -m defense.reflection --model $model --mode "comic" --type $type --dataset-path $DATASET_PATH --output-dir $OUTPUT_DIR --dataset-image-dir "dataset" --start $START --end $END
done

# Run random_image attack
for meme_id in 0 1 2
do
    uv run python -m defense.reflection --model $model --mode "meme" --meme-id $meme_id --dataset-path $DATASET_PATH --output-dir $OUTPUT_DIR --dataset-image-dir "meme_images" --start $START --end $END
done
