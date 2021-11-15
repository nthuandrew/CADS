from module.util import *
from class_Segmentation import Segmentation
from class_BertForClassification_Wrapper import Bert_Wrapper


# model_setting = {'criminal_type': 'gun', 'classify': 'sentence'}
# Sex
# model_setting = {'criminal_type': 'sex', 'classify': 'factor', 'factor_lst': ['犯罪後之態度', '犯罪之手段與所生損害', '被害人的態度', '被告之品行', '其他審酌事項']}
# Gun
# model_setting = {'criminal_type': 'gun', 'classify': 'factor', 'factor_lst': ['犯罪後之態度', '犯罪所生之危險或違反義務之程度', '被告之品行', '其他審酌事項']}
# Drug
model_setting = {'criminal_type': 'drug', 'classify': 'factor', 'factor_lst': ['犯罪後之態度', '犯罪所生之危險或損害或違反義務之程度', '被告之品行', '其他審酌事項']}


# Segmentation
# df = pd.read_excel(f'data/raw/data_criminal_%s.xlsx' % model_setting['criminal_type'])
# seg = Segmentation(type="bert")
# # modify col names
# if criminal_type == 'sex':
#     categorical_info = ['法條', '犯罪後之態度', '犯罪之手段與所生損害', '被害人的態度', '被告之品行', '其他審酌事項', '有利', '中性', '不利']
# elif criminal_type=='gun':
#     categorical_info = ['法條', '犯罪後之態度', '犯罪所生之危險或違反義務之程度', '被告之品行', '其他審酌事項', '有利', '中性', '不利']
# elif criminal_type=='drug':
#     categorical_info = ['法條', '犯罪後之態度', '犯罪所生之危險或損害或違反義務之程度', '被告之品行', '其他審酌事項', '有利', '中性', '不利']
# # run
# df = seg.segment_criminal_sentiment_analysis_articles_wrapper(df, categorical_info, model_setting['criminal_type'])
# del df, seg
# gc.collect()

if model_setting['classify'] == 'sentence':
    # BERT: Classification for criminal sentiment analysis
    seed_list = [1234, 5678, 7693145, 947, 13, 27, 1, 5, 9, 277]
    df = pd.read_pickle(f'./data/cleaned/criminal_%s_seg_bert.pkl' % model_setting['criminal_type'])
    df_neu = pd.read_pickle(f'./data/cleaned/criminal_%s_neutral_seg_bert.pkl' % model_setting['criminal_type'])
    # for i in range(10):
    #     print("Start test:", )
    #     bw = Bert_Wrapper(num_labels = 3, seed = seed_list[])
    #     trainloader, validloader, testloader = bw.prepare_criminal_sentiment_analysis_dataloader(df)
    #     bw.initialize_training()
    #     bw.train()
    #     bw.evaluate(path=f"{criminal_type}.txt")
    bw = Bert_Wrapper(num_labels = 3, seed = seed_list[0])
    trainloader, validloader, testloader = bw.prepare_criminal_sentiment_analysis_dataloader(df, df_neu)
    bw.initialize_training()
    bw.train()
    bw.evaluate(path=f"%s.txt" % model_setting['criminal_type'])

elif model_setting['classify'] == 'factor':
    # BERT: Classification for criminal factor classification #############
    df = pd.read_pickle(f'./data/cleaned/criminal_%s_seg_bert.pkl' % model_setting['criminal_type'])
    df_neu = pd.read_pickle(f'./data/cleaned/criminal_%s_neutral_seg_bert.pkl' % model_setting['criminal_type'])
    for fac in model_setting['factor_lst']:
        bw = Bert_Wrapper(num_labels = 2)
        trainloader, validloader, testloader = bw.prepare_criminal_judgement_factor_dataloader(df, df_neu, fac)
        bw.initialize_training()
        bw.train()
        bw.evaluate(path=f"%s_%s.txt" % (model_setting['criminal_type'], fac))
        del trainloader, validloader, testloader , bw
        gc.collect()
# %%
