from module.util import *
from class_Segmentation import Segmentation
from class_BertForClassification_Wrapper import Bert_Wrapper

criminal_type = 'gun'

# Segmentation
df = pd.read_excel(f'data/raw/data_criminal_{criminal_type}.xlsx')
seg = Segmentation(type="bert")
# modify col names
if criminal_type == 'sex':
    categorical_info = ['法條', '犯罪後之態度', '犯罪之手段與所生損害', '被害人的態度', '被告之品行', '其他審酌事項', '有利', '中性', '不利']
elif criminal_type=='gun':
    categorical_info = ['法條', '犯罪後之態度', '犯罪所生之危險或違反義務之程度', '被告之品行', '其他審酌事項', '有利', '中性', '不利']
elif criminal_type=='drug':
    categorical_info = ['法條', '犯罪後之態度', '犯罪所生之危險或損害或違反義務之程度', '被告之品行', '其他審酌事項', '有利', '中性', '不利']
# run
df = seg.segment_criminal_sentiment_analysis_articles_wrapper(df, categorical_info, criminal_type)
del df, seg
gc.collect()


# BERT: Classification for criminal sentiment analysis
seed_list = [1234, 5678, 7693145, 947, 13, 27, 1, 5, 9, 277]
df = pd.read_pickle(f'./data/cleaned/criminal_{criminal_type}_seg_bert.pkl')
# for i in range(10):
#     print("Start test:", )
#     bw = Bert_Wrapper(num_labels = 3, seed = seed_list[])
#     trainloader, validloader, testloader = bw.prepare_criminal_sentiment_analysis_dataloader(df)
#     bw.initialize_training()
#     bw.train()
#     bw.evaluate(path=f"{criminal_type}.txt")
bw = Bert_Wrapper(num_labels = 3, seed = seed_list[0])
trainloader, validloader, testloader = bw.prepare_criminal_sentiment_analysis_dataloader(df)
bw.initialize_training()
bw.train()
bw.evaluate(path=f"{criminal_type}.txt")

