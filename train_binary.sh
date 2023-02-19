MODEL_DIR=./ckpt
MODEL_NAME=AND
VERSION=0.0

TRAIN_DATA_PATH=./data/raw/data_criminal_drug.xlsx
EXTERNAL_DATA_PATH=./data/raw/data_criminal_drug_neutral.xlsx
# Train binary classification
python main.py \
    --mode train \
    --train_data_path $TRAIN_DATA_PATH \
    --external_data_path $EXTERNAL_DATA_PATH \
    --external_column_idx 0 \
    --text_column_name Sentence \
    --label_column_list 犯罪後之態度 \
    --model_dir $MODEL_DIR \
    --model_name $MODEL_NAME \
    --model_version $VERSION

# rm -r $MODEL_ DIR

