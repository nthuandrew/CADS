MODEL_DIR=./ckpt
MODEL_NAME=AND
VERSION=0.0

TRAIN_DATA_PATH=./data/raw/data_criminal_drug.xlsx
EXTERNAL_DATA_PATH=./data/raw/data_criminal_drug_neutral.xlsx
# Train mlti-class classification
python main.py \
    --mode train \
    --pretrained_model_name hfl/chinese-macbert-base \
    --max_seq_len 256 \
    --train_data_path $TRAIN_DATA_PATH \
    --text_column_name Sentence \
    --external_column_idx 2 \
    --label_column_list 不利 有利 中性 \
    --model_dir $MODEL_DIR \
    --model_name $MODEL_NAME \
    --model_version $VERSION \
    --external_data_path $EXTERNAL_DATA_PATH \

