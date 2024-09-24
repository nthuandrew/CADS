MODEL_DIR=./ckpt
# MODEL_NAME=augmented_by_categories_0428_bert_no_neutral_v2_128 #date_time
MODEL_NAME=two_class_0428_v4 #date_time
VERSION=0.0

TRAIN_DATA_PATH=./data/raw/two_class_train.xlsx
# TRAIN_DATA_PATH=./data/raw/new_clean_data_partial.xlsx
# Train multi-class classification
python main.py \
    --mode train \
    --pretrained_model_name hfl/chinese-bert-wwm-ext \
    --max_seq_len 128 \
    --train_data_path $TRAIN_DATA_PATH \
    --text_column_name Sentence \
    --label_column_list 無標註 有標註 \
    --model_dir $MODEL_DIR \
    --model_name $MODEL_NAME \
    --model_version $VERSION \
    --epoch 4 \
    --train_test_split_ratio 0.1 \
    --batch_size 64 \
    --lr 1e-5 
    # --label_column_list 自殺與憂鬱 無助或無望 正向文字 其他負向文字 生理反應或醫療狀況 自殺行為 \

