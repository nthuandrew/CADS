import pandas as pd
import torch
from module.util import *
from src.class_BertForClassification_Pipeline import Pipeline as PL
from src.article_predict import *
from transformers import BertTokenizer, BertConfig
import json
import time
from tqdm import tqdm

# 讀取內容
content_file = '/cluster/home/andrewchen/andrewchen/dcard/dcard_mood_post_content_recrawl.json'
with open(content_file, 'r', encoding='utf-8') as f:
    content_list = f.readlines()
content_list = [json.loads(post) for post in content_list]
df = pd.DataFrame(content_list)
df['createdAt'] = pd.to_datetime(df['createdAt'])
# df = df[df['createdAt'] > '2023-06-30']
# df = df[df['createdAt'] < '2024-07-01']
print("Number of posts:", len(df))
# 模型設定和初始化
pretrained_model_name = 'hfl/chinese-bert-wwm-ext'
sentence_model_path = './ckpt/final/sentence/version_seed_1/four_class_augmented_fold_5.pkl'
# article_model_path = './ckpt/final/article/A1_train_augmented_type3_fold_5.pkl' # class 4
article_model_path = './ckpt/final/article/A1A2_train_raw_fold_5.pkl' # class 2

label_column_list = ['中性句','自殺與憂鬱', '自殺行為', '其他類型']
# article_label_list = ['無危機文章', '高危機文章']
# article_label_list = ['0', '1', '2', '3']
article_label_list = ['0', '1']
seed = 1234

sentence_model = PL(pretrained_model_name=pretrained_model_name, load_model_path=sentence_model_path, \
        num_labels = len(label_column_list), seed = seed, batch_size=64)
sentence_model.initialize_training()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
config = BertConfig.from_pretrained(pretrained_model_name)
config.max_length = 512
article_model = MyModel(config, num_labels=len(article_label_list)).to(device)
article_model.load_state_dict(torch.load(article_model_path, map_location='cpu'))
article_model.eval()
# 預處理和預測
df['content'] = df['content'].astype(str)

Crisis_level_list = []
for content in tqdm(df['content']):
    try: 
        sample_sentences = [sent for sent in split_sentence(content) if sent not in ['', ' ', '。', '，',',', '\n']]
        pred_loader = sentence_model.prepare_dataloader(sample_sentences, for_prediction=True, not_df=True)
        predicts = sentence_model.get_predictions(sentence_model.model, pred_loader)
        predicts_label = [label_column_list[i] for i in predicts]
        # paragraphs = [label+'的段落:' for label in label_column_list]
        paragraphs = ['','','','']
        for idx, pred in enumerate(predicts):
            paragraphs[pred] += ' ' + sample_sentences[idx]
        paragraphs = np.array([paragraphs])
        # ... 進行預測 ...
        test_dataset = TextClassificationDataset(paragraphs, [[0.0]*len(article_label_list)], tokenizer, 'test')
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        pred = article_model.get_predictions(test_dataloader)
        Crisis_level = article_label_list[pred[0]]
        Crisis_level_list.append(Crisis_level)
    except:
        Crisis_level_list.append('0')
        print('error')
        print(content)
        continue
df['crisis_level'] = Crisis_level_list

# 儲存預測結果
output_path = './dcard_predictions_2_class_all.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print("Predictions saved to:", output_path)
