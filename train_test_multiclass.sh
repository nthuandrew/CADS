#!/bin/bash

SEED=1
PRETRAINED_MODEL_NAME=hfl/chinese-bert-wwm-ext
MODEL_DIR=./ckpt/final/sentence/

# Loop through folds 1 to 5
for FOLD in {1..5}
do
  VERSION=seed_${SEED}
  MODEL_NAME=article_type3_fold_$FOLD
  TRAIN_DATA_PATH=data/final/sentence/article/seed_$SEED/article_type3_train_raw_fold_$FOLD.xlsx
#   TRAIN_DATA_PATH=data/final/sentence/four_class/seed_$SEED/four_class_0530_train_raw_fold_$FOLD.xlsx
#   TRAIN_DATA_PATH=data/final/sentence/seven_class/seed_$SEED/seven_class_0530_train_augmented_fold_$FOLD.xlsx
#   TEST_DATA_PATH=data/final/sentence/seven_class/seed_$SEED/seven_class_0530_test_fold_$FOLD.xlsx
  TEST_DATA_PATH=data/final/sentence/article/seed_$SEED/article_type3_test_fold_$FOLD.xlsx
  LOAD_MODEL_PATH=./ckpt/final/sentence/version_seed_$SEED/$MODEL_NAME.pkl

#   Train multi-class classification
  python3 main_sentence.py \
      --mode train \
      --pretrained_model_name $PRETRAINED_MODEL_NAME \
      --max_seq_len 512 \
      --train_data_path $TRAIN_DATA_PATH \
      --model_dir $MODEL_DIR \
      --model_name $MODEL_NAME \
      --model_version $VERSION \
      --epoch 8 \
      --train_test_split_ratio 0.001 \
      --batch_size 32 \
      --lr 2e-5 \
      --seed $SEED \
      --label_column_list d:0 c:C b:B a:A \
      --text_column_name Content \
    #   --text_column_name Sentence \
    #   --label_column_list 無標註 自殺與憂鬱 無助或無望 正向文字 其他負向文字 生理反應或醫療狀況 自殺行為 \
    #   --label_column_list 無標註 自殺與憂鬱 自殺行為 其他類型 \

  # Test multi-class classification
  python3 main_sentence.py \
      --mode test \
      --text_column_name Content \
      --pretrained_model_name $PRETRAINED_MODEL_NAME \
      --load_model_path $LOAD_MODEL_PATH \
      --test_data_path $TEST_DATA_PATH \
      --model_dir $MODEL_DIR \
      --model_version $VERSION \
      --label_column_list d:0 c:C b:B a:A \
    #   --label_column_list 無標註 自殺與憂鬱 自殺行為 其他類型 \

# --label_column_list 無標註 自殺與憂鬱 無助或無望 正向文字 其他負向文字 生理反應或醫療狀況 自殺行為 \
done
