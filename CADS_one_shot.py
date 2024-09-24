# from flask import Flask, request, jsonify
# app = Flask(__name__)
import os
import sys
import time
start_time = time.time()
from module.util import *
from src.class_BertForClassification_Pipeline import Pipeline as PL
from src.article_predict import *
from transformers import BertTokenizer, BertConfig
import warnings
import logging
print(f"import over：{time.time() - start_time:.2f} 秒")
warnings.simplefilter(action='ignore', category=Warning)
logging.getLogger('transformers').setLevel(logging.ERROR)
print("CADS TIME START")
input_path = './data/article/A2_final_test_40_with_pred_v2.xlsx'
output_path = './data/article/A2_final_test_40_with_pred_v3.xlsx'
sentence_model_path = './ckpt/version_0.0/four_class_augmented_0819.pkl'
article_model_path = './ckpt/article/0819_400limit_3epoch_binary_short_5.pkl'
pretrained_model_name = 'hfl/chinese-bert-wwm-ext'

label_column_list = ['無標註','自殺與憂鬱', '自殺行為', '其他類型']
# article_label_list = ['無危機文章', '高危機文章']
article_label_list = [0, 1]
seed = 1234

pl = PL(pretrained_model_name=pretrained_model_name, load_model_path=sentence_model_path, \
        num_labels = len(label_column_list), seed = seed, batch_size=64)
pl.initialize_training()
print(f"sentences model load over：{time.time() - start_time:.2f} 秒")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
config = BertConfig.from_pretrained(pretrained_model_name)
config.max_length = 512
article_model = MyModel(config, num_labels=2).to(device)
article_model.load_state_dict(torch.load(article_model_path, map_location='cpu'))
article_model.eval()
# optimizer = AdamW(model.parameters(), lr=lr)
print(f"article model load over：{time.time() - start_time:.2f} 秒")
print("PREPARE OVER")
Title = "1995"
TextIDs = '660a141cc9'
TextTimes = '2021-09-08 17:44:13'
Authors = '匿名/男'

# @app.route('/')
# def index():
#     return "CADS API"

# @app.route('/predict', methods=['POST'])
# def predict():
    # 從請求中獲取數據，進行預測
print("start predict")
pred_time = time.time()
df = pd.read_excel(input_path)
contents = df['Content(remove_tag)'].tolist()
answer = []
# with open(input_path, 'r', encoding='utf-8') as file:
#     content = file.read()

for content in tqdm(contents):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    null = open(os.devnull, 'w')
    sys.stdout = null
    sys.stderr = null
    sample_sentences = [sent for sent in split_sentence(content) if sent not in ['', ' ', '。', '，', '\n']]
    # print(f"article content load and split over：{time.time() - pred_time:.2f} 秒")
    pred_loader = pl.prepare_dataloader(sample_sentences, for_prediction=True, not_df=True)
    predicts = pl.get_predictions(pl.model, pred_loader)
    # print(f"sentence model predict over：{time.time() - pred_time:.2f} 秒")
    predicts_label = [label_column_list[i] for i in predicts]
    # paragraphs = [label+'的段落:' for label in label_column_list]
    paragraphs = ['','','','']
    for idx, pred in enumerate(predicts):
        paragraphs[pred] += ' ' + sample_sentences[idx]
    paragraphs = np.array([paragraphs])
    # ... 進行預測 ...
    test_dataset = TextClassificationDataset(paragraphs, [[0.0, 0.0]], tokenizer, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    # 返回預測結果
    y_true = []
    y_pred = []
    train_total = 0.0
    for batch_data in tqdm(test_dataloader):
        tokens_tensors, segments_tensors, masks_tensors,labels  = [t.to(device) for t in batch_data]
        with torch.no_grad(): 
            loss, logits = article_model(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors,labels=labels)
        # print(loss)
            prob = logits.data
            _, pred = torch.max(prob, 1)
            _, truth = torch.max(labels, 1)
            train_total += labels.size(0)
            y_true.append(truth)
            y_pred.append(pred)
            del tokens_tensors
            del segments_tensors
            del masks_tensors
            del labels
    pred = torch.cat(y_pred)
    ground_truth = torch.cat(y_true)
    Crisis_level = article_label_list[pred[0]]
    answer.append(Crisis_level)
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    null.close()
print(f"article model predict over：{time.time() - pred_time:.2f} 秒")
    # with open(output_path, 'w', encoding='utf-8') as file:
    #     file.write('Crisis_level, ' + Crisis_level + '\n')
    #     file.write('Title, ' + Title + '\n')
    #     file.write('TextID, ' + TextIDs + '\n')
    #     file.write('Time, ' + TextTimes + '\n')
    #     file.write('Author, ' + Authors + '\n')
    #     file.write('---'+ '\n')
    #     for idx, sentence in enumerate(sample_sentences):
    #         file.write(sentence + ',' + predicts_label[idx] + '\n')
    # print(f"save final prediction over：{time.time() - pred_time:.2f} 秒")
    # return jsonify({"MESSAGE": "SUCCESS"})
df['pred'] = answer
# print(df['pred'])
# print(df['pred_v2'])
df.to_excel(output_path, index=False)
truth_label = df['Crisis_Level'].tolist()
truth = [1 if i == 'a:A' else 0 for i in truth_label]
cm = confusion_matrix(y_true=truth, y_pred=answer, labels=[0,1])
acc = accuracy_score(y_true=truth, y_pred=answer)
f1 = f1_score(y_true=truth, y_pred=answer, average=None, labels=[0,1])
p_score = precision_score(y_true=truth, y_pred=answer, average=None, labels=[0,1])
r_score = recall_score(y_true=truth, y_pred=answer, average=None, labels=[0,1])
print("accuracy: ", round(acc, 3))
print("precision_score: ", [round(x, 3) for x in p_score])
print("recall_score: ", [round(x, 3) for x in r_score])
print("f1-score: ", [round(x, 3) for x in f1])
print("confusion matrix:\n", cm)
# if __name__ == '__main__':
#     port_num = 5050
#     print("RUN ON PORT", port_num)
#     app.run(port=port_num)
