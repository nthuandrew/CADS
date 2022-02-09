# %%
from module.util import *
from class_Segmentation import Segmentation
from class_BertForClassification_Wrapper import Bert_Wrapper
import argparse
# %%
parser = argparse.ArgumentParser(description='Still thinking...')
parser.add_argument("--mode", type=str, default="train_sentence")
parser.add_argument("--train_data", type=str, default="sex")
# factor_list 可以不要
parser.add_argument("--factor_lst", type=str, default="sex_factor_lst")
parser.add_argument("--pred_data", type=str, default="data_test_sex")
# 要不要再斷詞一遍
parser.add_argument("--do_segment", type=bool, default=True)
# 有沒有中性句
parser.add_argument("--neutral_data", type=str, default="")
# Model 的版本
parser.add_argument("--version", type=int, default=0)
# Model save name
parser.add_argument("--model_name", type=str, default="")
# Model setting
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epoch", type=int, default=2)
parser.add_argument("--pooling_strategy", type=str, default='reduce_mean')
parser.add_argument("--lr", type=float, default=2e-5)

args = parser.parse_args()
# %%

'''
Define factor list for different cases.
'''
# For sex
sex_factor_lst = ['犯罪後之態度', '犯罪手段與所生之損害', '被害人的態度',
    '犯罪行為人之品行', '其他審酌事項']
# For gun
gun_factor_lst = ['犯罪後之態度', '犯罪所生之危險或損害', '犯罪行為人之品行',
     '其他審酌事項']
# For drug
drug_factor_lst = ['犯罪後之態度', '犯罪所生之危險或損害', '犯罪行為人之品行',
    '其他審酌事項']

'''
Modify model setting before use it.
'''
# model_setting = {'mode': 'train_sentence', \
# 'train_data': 'sex', 'factor_lst': sex_factor_lst}

model_setting = {'mode': 'train_factor', \
'train_data': 'sex', 'factor_lst': sex_factor_lst}

# model_setting = {'mode': 'pred_sentence', \
# 'train_data': 'gun', 'pred_data':'data_test_gun', 'factor_lst': gun_factor_lst}

# model_setting = {'mode': 'pred_factor', \
# 'train_data': 'gun', 'pred_data':'data_test_gun', 'factor_lst': gun_factor_lst}

# %%%

# Segmentation (data preparation)
# Training
bar = progressbar.ProgressBar()
for i in bar([model_setting['train_data'], '_'.join([model_setting['train_data'], 'neutral'])]): # df, df_neu
    print(f'>>>>> Segmenting training {i} data >>>>>')
    df = pd.read_excel(f'data/raw/data_criminal_%s.xlsx' % i)
    seg = Segmentation(type="bert")
    categorical_info = model_setting['factor_lst']+['有利', '中性', '不利'] # modify col names
    df = seg.segment_criminal_sentiment_analysis_articles_wrapper(df, categorical_info, criminal_type=i)
    del df, seg
gc.collect()
# %%
# Predict
# print(f'>>>>> Segmenting predicted data >>>>>')
# df = pd.read_excel(f'data/pred/%s.xlsx' % model_setting['pred_data'])
# seg = Segmentation(type="bert")
# categorical_info = model_setting['factor_lst']+['有利', '中性', '不利'] # modify col names
# df = seg.segment_criminal_sentiment_analysis_articles_wrapper(df, categorical_info, criminal_type=model_setting['pred_data'])
# del df, seg
# gc.collect()

# %%
# df, df_neu 拿到這邊宣告
# %%
if model_setting['mode'] == 'train_sentence':
    ############# BERT: Classification for criminal sentiment analysis #############
    seed_list = [1234, 5678, 7693145, 947, 13, 27, 1, 5, 9, 277]
    df = pd.read_pickle(f'./data/cleaned/criminal_%s_seg_bert.pkl' % model_setting['train_data'])
    # df_neu = pd.read_pickle(f'./data/cleaned/criminal_%s_neutral_seg_bert.pkl' % model_setting['train_type'])
    # for i in range(10):
    #     print("Start test:", )
    #     bw = Bert_Wrapper(num_labels = 3, seed = seed_list[])
    #     trainloader, validloader, testloader = bw.prepare_criminal_sentiment_analysis_dataloader(df)
    #     bw.initialize_training()
    #     bw.train()
    #     bw.evaluate(path=f"{criminal_type}.txt")
    model_name = f"version_1/{model_setting['train_data']}_{model_setting['mode']}_epoch4_seed1234_0209"
    bw = Bert_Wrapper(save_model_name=model_name, num_labels = 3, seed = seed_list[0])
    # trainloader, validloader, testloader = bw.prepare_criminal_sentiment_analysis_dataloader(df, df_neu)
    trainloader, validloader, testloader = bw.prepare_criminal_dataloader(
        bw._extract_criminal_sentiment_analysis_datalst(df=df, target_feature=['不利', '有利', '中性'])
    )
    bw.initialize_training()
    bw.train()
    bw.evaluate(path=f"%s.txt" % model_setting['train_data'])

elif model_setting['mode'] == 'train_factor':
    ############# BERT: Classification for criminal factor classification #############
    df = pd.read_pickle(f'./data/cleaned/criminal_%s_seg_bert.pkl' % model_setting['train_data'])
    df_neu = pd.read_pickle(f'./data/cleaned/criminal_%s_neutral_seg_bert.pkl' % model_setting['train_data'])
    for fac in model_setting['factor_lst']:
        model_name = f"version_1/{model_setting['train_data']}_{model_setting['mode']}_{fac}_epoch2_seed1234_0209"
        bw = Bert_Wrapper(save_model_name=model_name, num_labels = 2)
        # trainloader, validloader, testloader = bw.prepare_criminal_judgement_factor_dataloader(df, df_neu, fac)
        trainloader, validloader, testloader = bw.prepare_criminal_dataloader(
        bw._extract_criminal_judgement_factor_datalst(df, df_neu, target_feature=fac)
        )
        bw.initialize_training()
        bw.train()
        bw.evaluate(path=f"%s_%s.txt" % (model_setting['train_data'], fac))
        del trainloader, validloader, testloader , bw
        gc.collect()

elif model_setting['mode'] == 'pred_sentence':
    ############# BERT: Classification for criminal sentiment analysis #############
    #training data
    df = pd.read_pickle(f'./data/cleaned/criminal_%s_seg_bert.pkl' % model_setting['train_data'])
    df_neu = pd.read_pickle(f'./data/cleaned/criminal_%s_neutral_seg_bert.pkl' % model_setting['train_data'])
    # predict data
    df_final = pd.read_excel(f'data/pred/%s.xlsx' % model_setting['pred_data'])    # word
    df_pred = pd.read_pickle('./data/cleaned/criminal_%s_seg_bert.pkl' % model_setting['pred_data'])    # vector
    
    model_name = f"{model_setting['train_data']}_{model_setting['mode']}_epoch2_seed1234_1128"
    bw = Bert_Wrapper(save_model_name=model_name, num_labels = 3)
    # trainloader, validloader, testloader = bw.prepare_criminal_sentiment_analysis_dataloader(df, df_neu)
    trainloader, validloader, testloader = bw.prepare_criminal_dataloader(
        bw._extract_criminal_sentiment_analysis_datalst(df, df_neu, ['不利', '有利', '中性'])
    )
    bw.initialize_training()
    bw.train()
    bw.evaluate(path=f"%s.txt" % model_setting['train_data'])

    # predloader = bw.prepare_criminal_sentiment_analysis_dataloader(df_pred, for_prediction=True)
    predloader = bw.prepare_criminal_dataloader(df_pred, for_prediction=True)
    predictions = bw.predict(predloader)
    df_final['不利'] = predictions[:, 0]
    df_final['有利'] = predictions[:, 1]
    df_final['中性'] = predictions[:, 2]
    df_final.to_excel(f'data/pred/%s.xlsx' % model_setting['pred_data'], index=False)
    del bw, trainloader, validloader, testloader , predloader, predictions
    gc.collect()

elif model_setting['mode'] == 'pred_factor':
    ############# BERT: Classification for criminal factor classification #############
    # training data
    df = pd.read_pickle(f'./data/cleaned/criminal_%s_seg_bert.pkl' % model_setting['train_data'])   # vector
    df_neu = pd.read_pickle(f'./data/cleaned/criminal_%s_neutral_seg_bert.pkl' % model_setting['train_data'])   # vector
    # predict data
    df_final = pd.read_excel(f'data/pred/%s.xlsx' % model_setting['pred_data'])    # word
    df_pred = pd.read_pickle('./data/cleaned/criminal_%s_seg_bert.pkl' % model_setting['pred_data'])    # vector
    
    bar = progressbar.ProgressBar()
    for fac in bar(model_setting['factor_lst']):
        # m_name = 'bert_%s_%s' % (model_setting['train_data'], fac)
        # bw = Bert_Wrapper(save_model_name=m_name, num_labels = 2)
        model_name = f"{model_setting['train_data']}_{model_setting['mode']}_{fac}_epoch2_seed1234_1128"
        bw = Bert_Wrapper(save_model_name=model_name, num_labels = 2)
        # trainloader, validloader, testloader = bw.prepare_criminal_judgement_factor_dataloader(df, df_neu, target_feature=fac)
        trainloader, validloader, testloader = bw.prepare_criminal_dataloader(
        bw._extract_criminal_judgement_factor_datalst(df, df_neu, target_feature=fac)
        )
        bw.initialize_training()
        bw.train()  # call trained model or train a new model
        bw.evaluate(path=f"%s_%s.txt" % (model_setting['train_data'], fac))

        # predloader = bw.prepare_criminal_judgement_factor_dataloader(df_pred, target_feature=fac, for_prediction=True)
        predloader = bw.prepare_criminal_dataloader(df_pred, for_prediction=True)
        predictions = bw.predict(predloader)
        df_final[fac] = predictions[:, 0]
        del bw, trainloader, validloader, testloader , predloader, predictions
        gc.collect()
    
    df_final.to_excel(f'data/pred/%s.xlsx' % model_setting['pred_data'], index=False)
