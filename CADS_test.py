import time
start_time = time.time()
from module.util import *
from src.class_BertForClassification_Pipeline import Pipeline as PL
from src.article_predict import *
from transformers import BertTokenizer, BertConfig
import warnings
import logging
from glob import glob

print(f"import over：{time.time() - start_time:.2f} 秒")
warnings.simplefilter(action='ignore', category=Warning)
logging.getLogger('transformers').setLevel(logging.ERROR)
print("CADS TIME START")
input_path_array = sorted(glob('./data/3min_逐字稿_csv/*.csv'))
output_path = './data/3min_逐字稿_csv/predict_csv/prediction.csv'
output_dict = {}
# output_path = '../article/prediction.txt'
sentence_model_path = './ckpt/version_0.0/four_class_augmented_0819.pkl'
article_model_path = './ckpt/article/0819_400limit_3epoch_binary_short_5.pkl'
pretrained_model_name = 'hfl/chinese-bert-wwm-ext'

label_column_list = ['中性句','自殺與憂鬱', '自殺行為', '其他類型']
# article_label_list = ['無危機文章', '高危機文章']
article_label_list = ['0', 'A']
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

Title = "1995"
TextIDs = '660a141cc9'
TextTimes = '2021-09-08 17:44:13'
Authors = '匿名/男'
# 從請求中獲取數據，進行預測
print("input_path_array", input_path_array)
for input_path in input_path_array:
    print("start predict")
    pred_time = time.time()
    df = pd.read_csv(input_path)
    sample_sentences = df['Text'].tolist()
    sample_sentences = [str(sent) for sent in sample_sentences]
    # with open(input_path, 'r', encoding='utf-8') as file:
    #     content = file.read()
    # sample_sentences = [sent for sent in split_sentence(content) if sent not in ['', ' ', '。', '，', '\n']]
    print(f"article content load and split over：{time.time() - pred_time:.2f} 秒")
    pred_loader = pl.prepare_dataloader(sample_sentences, for_prediction=True, not_df=True)
    predicts = pl.get_predictions(pl.model, pred_loader)
    print(f"sentence model predict over：{time.time() - pred_time:.2f} 秒")
    predicts_label = [label_column_list[i] for i in predicts]
    df['sentence_predict'] = predicts_label
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
    print(f"article model predict over：{time.time() - pred_time:.2f} 秒")
    df['Crisis_level'] = ['']*len(df)
    df['Crisis_level'][0] = Crisis_level
    df.to_csv(input_path, index=False, encoding='utf-8-sig')
    # with open(output_path, 'w', encoding='utf-8') as file:
    #     file.write('Crisis_level, ' + Crisis_level + '\n')
    #     file.write('Title, ' + Title + '\n')
    #     file.write('TextID, ' + TextIDs + '\n')
    #     file.write('Time, ' + TextTimes + '\n')
    #     file.write('Author, ' + Authors + '\n')
    #     file.write('---'+ '\n')
    #     for idx, sentence in enumerate(sample_sentences):
    #         file.write(sentence + ',' + predicts_label[idx] + '\n')
    print(f"save final prediction over：{time.time() - pred_time:.2f} 秒")


