#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com



ORIGIN_DATASET_PATH=origin_song_eval_dataset
MUSICFM_CKPT_PATH=MusicFM
DATASET_PKL_PATH=dataset_pkl


# stage 1: download dataset
huggingface-cli download --repo-type dataset ASLP-lab/SongEval --local-dir $ORIGIN_DATASET_PATH

# stage2: 
curl -o val_ids.txt https://raw.githubusercontent.com/ASLP-lab/Automatic-Song-Aesthetics-Evaluation-Challenge/refs/heads/main/static/val_ids.txt
find $ORIGIN_DATASET_PATH/mp3 -name "*.mp3" | grep -f "val_ids.txt" > test.txt
find $ORIGIN_DATASET_PATH/mp3 -name "*.mp3" | grep -vf "val_ids.txt" > train.txt

# stage 2: Audio Augmentation
python audio_augmentation.py \
    --input_txt train.txt \
    --output_txt train_aug.txt \
    --output_dir ${ORIGIN_DATASET_PATH}/mp3_aug \
    --n_augmentations 3 

# stage 3: Extract MuQ and MusicFM feature
bash download_musicfm.sh ${MUSICFM_CKPT_PATH}

# extract train_aug_embedding
python extract_muq_and_musicfm_feature.py \
    --input_txt train_aug.txt \
    --output_txt train_aug_embedding.txt \
    --output_dir train_aug_embedding \

# extract eval_embedding
python extract_muq_and_musicfm_feature.py \
    --input_txt test.txt \
    --output_txt test_embedding.txt \
    --output_dir test_embedding \

# stage 4: generate train_set.pkl and eval_set.pkl
mkdir -p ${DATASET_PKL_PATH} > /dev/null 2>&1
python generate_pkl.py \
    --input_embedding_txt_path train_aug_embedding.txt \
    --save_pkl_path ${DATASET_PKL_PATH}/train_set.pkl \
    --song_eval_metadata_jsonl ${ORIGIN_DATASET_PATH}/metadata.jsonl

python generate_pkl.py \
    --input_embedding_txt_path test_embedding.txt \
    --save_pkl_path ${DATASET_PKL_PATH}/test_set.pkl \
    --song_eval_metadata_jsonl ${ORIGIN_DATASET_PATH}/metadata.jsonl
