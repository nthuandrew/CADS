import pandas as pd
import torch
from module.util import *
from src.class_BertForClassification_Pipeline import Pipeline as PL
from src.article_predict import *
from transformers import BertTokenizer, BertConfig
import json
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
# 讀取內容
content_file = '/cluster/home/andrewchen/andrewchen/dcard/dcard_mood_post_content_recrawl.json'
with open(content_file, 'r', encoding='utf-8') as f:
    content_list = f.readlines()
content_list = [json.loads(post) for post in content_list][:64]
print("Number of posts:", len(content_list))
df = pd.DataFrame(content_list)

# 模型設定和初始化
pretrained_model_name = 'hfl/chinese-bert-wwm-ext'
sentence_model_path = './ckpt/final/sentence/version_seed_1/four_class_augmented_fold_5.pkl'
article_model_path = './ckpt/final/article/A1_train_augmented_type3_fold_5.pkl'
label_column_list = ['中性句','自殺與憂鬱', '自殺行為', '其他類型']
# article_label_list = ['無危機文章', '高危機文章']
article_label_list = ['0', '1', '2', '3']
seed = 1234

sentence_model = PL(pretrained_model_name=pretrained_model_name, load_model_path=sentence_model_path, \
        num_labels = len(label_column_list), seed = seed, batch_size=64)
sentence_model.initialize_training()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
config = BertConfig.from_pretrained(pretrained_model_name)
config.max_length = 512
article_model = MyModel(config, num_labels=4).to(device)
article_model.load_state_dict(torch.load(article_model_path, map_location='cpu'))
article_model.eval()
# 預處理和預測

# Preparing data structures
df['content'] = df['content'].astype(str)
paragraphs_list = []  # This will store all paragraphs arrays for each content
content_indices = []  # To track which content index each paragraph array belongs to

for content in tqdm(df['content']):
    sample_sentences = [sent for sent in split_sentence(content) if sent not in ['', ' ', '。', '，',',', '\n']]
    pred_loader = sentence_model.prepare_dataloader(sample_sentences, for_prediction=True, not_df=True)
    predicts = sentence_model.get_predictions(sentence_model.model, pred_loader)
    
    paragraphs = ['','','','']
    for idx, pred in enumerate(predicts):
        paragraphs[pred] += ' ' + sample_sentences[idx]
    
    paragraphs_list.append(paragraphs)
    content_indices.append(df.index[df['content'] == content].tolist())

# Flatten the list of indices since there might be duplicated indices
flat_content_indices = [item for sublist in content_indices for item in sublist]


# Preparing datasets and dataloaders
test_datasets = [TextClassificationDataset(np.array([paras]), [[0.0]*len(article_label_list)], tokenizer, 'test')
                 for paras in paragraphs_list]
# Combine all datasets into one for batch processing
combined_dataset = ConcatDataset(test_datasets)
test_dataloader = DataLoader(combined_dataset, batch_size=32)  # Adjust batch size according to your GPU capacity

# Predicting crisis level in batches
Crisis_level_list = []
for batch in tqdm(test_dataloader):
    print("Batch size:", len(batch))
    print("Batch:", batch)
    print("Batch shape:", batch[0].shape, batch[1].shape, batch[2].shape, batch[3].shape)
    batch_predictions = article_model.get_predictions(batch)
    Crisis_levels = [article_label_list[pred] for pred in batch_predictions]
    Crisis_level_list.extend(Crisis_levels)

# Mapping crisis levels back to original DataFrame
df['crisis_level'] = pd.Series(Crisis_level_list).values[flat_content_indices]


# 儲存預測結果
output_path = './prediction_with_crisis_level.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print("Predictions saved to:", output_path)
