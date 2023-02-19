# export PYTHONPATH=/home/yining_juan/Documents/Bert-sentence/py_law
LOAD_MODEL_PATH=./ckpt/version_0.0/AND.pkl
TEST_DATA_PATH=./data/pred/data_test_drug.xlsx
# Test multi-class classification
# python main.py \
#     --mode test \
#     --text_column_name Sentence \
#     --label_column_list 不利 有利 中性 \
#     --load_model_path $LOAD_MODEL_PATH \
#     # --test_data_path $TEST_DATA_PATH \
#     --test_sample 爰審酌被告前有觀察勒戒、毒品等前科紀錄， \

# Test binary classification
python main.py \
    --mode test \
    --text_column_name Sentence \
    --label_column_list 犯罪後之態度 \
    --load_model_path $LOAD_MODEL_PATH \
    --test_sample 被告犯後坦承犯行、態度尚可 \
    # --test_data_path $TEST_DATA_PATH \

#被告犯後坦承犯行、態度尚可