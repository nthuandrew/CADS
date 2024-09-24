# print execution time
from flask import Flask, request, jsonify, send_file
app = Flask(__name__)
import time
start_time = time.time()
# print start time nby Date, hour, minute, second
import datetime
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
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
input_path = '../article/article.txt'
output_path = '../article/prediction.txt'
sentence_model_path = '../CADS/ckpt/version_0.0/four_class_augmented_0819.pkl'
article_model_path = '../CADS/ckpt/article/0819_400limit_3epoch_binary_short_5.pkl'
pretrained_model_name = 'hfl/chinese-bert-wwm-ext'

label_column_list = ['無標註','自殺與憂鬱', '自殺行為', '其他類型']
# article_label_list = ['無危機文章', '高危機文章']
article_label_list = ['0', 'A']
seed = 1234

sentence_model = PL(pretrained_model_name=pretrained_model_name, load_model_path=sentence_model_path, \
        num_labels = len(label_column_list), seed = seed, batch_size=64)
sentence_model.initialize_training()
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
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
Title = "1995"
TextIDs = '660a141cc9'
TextTimes = '2021-09-08 17:44:13'
Authors = '匿名/男'

@app.route('/')
def index():
    return "CADS API"

@app.route('/predict', methods=['POST'])
def predict():
    # 從請求中獲取數據，進行預測
    print("start predict")
    pred_time = time.time()
    file = request.files['file']
    content = file.read().decode('utf-8')
    # with open(input_path, 'r', encoding='utf-8') as file:
    #     content = file.read()
    # print("Content:\n", content)
    sample_sentences = [sent for sent in split_sentence(content) if sent not in ['', ' ', '。', '，',',', '\n']]
    print(f"article content load and split over：{time.time() - pred_time:.2f} 秒")
    pred_loader = sentence_model.prepare_dataloader(sample_sentences, for_prediction=True, not_df=True)
    predicts = sentence_model.get_predictions(sentence_model.model, pred_loader)
    print(f"sentence model predict over：{time.time() - pred_time:.2f} 秒")
    predicts_label = [label_column_list[i] for i in predicts]
    # paragraphs = [label+'的段落:' for label in label_column_list]
    paragraphs = ['','','','']
    for idx, pred in enumerate(predicts):
        paragraphs[pred] += ' ' + sample_sentences[idx]
    paragraphs = np.array([paragraphs])
    # ... 進行預測 ...
    test_dataset = TextClassificationDataset(paragraphs, [[0.0, 0.0]], tokenizer, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    pred = article_model.get_predictions(test_dataloader)
    Crisis_level = article_label_list[pred[0]]
    print(f"article model predict over：{time.time() - pred_time:.2f} 秒")
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write('Crisis_level, ' + Crisis_level + '\n')
        file.write('Title, ' + Title + '\n')
        file.write('TextID, ' + TextIDs + '\n')
        file.write('Time, ' + TextTimes + '\n')
        file.write('Author, ' + Authors + '\n')
        file.write('---'+ '\n')
        for idx, sentence in enumerate(sample_sentences):
            file.write(sentence + ',' + predicts_label[idx] + '\n')
    print(f"save final prediction over：{time.time() - pred_time:.2f} 秒")
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return send_file(output_path, as_attachment=True, download_name="prediction.txt")

if __name__ == '__main__':
    port_num = 5555
    print("RUN ON PORT", port_num)
    app.run(port=port_num)
