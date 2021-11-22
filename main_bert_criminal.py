from module.util import *
from class_Segmentation import Segmentation
from class_BertForClassification_Wrapper import Bert_Wrapper



# model_setting = {'mode': 'train_sentence', 'train_data': 'gun'}
# model_setting = {'mode': 'train_factor', 'train_data': 'gun', 'factor_lst': ['犯罪後之態度', '犯罪所生之危險或違反義務之程度', '被告之品行', '其他審酌事項']}
model_setting = {'mode': 'pred_sentence', 'train_data': 'gun', 'pred_data':'全文標註的測試句'}
# model_setting = {'mode': 'pred_factor', 'train_data': 'gun', 'pred_data':'全文標註的測試句', 'factor_lst': ['犯罪後之態度', '犯罪所生之危險或違反義務之程度', '被告之品行', '其他審酌事項']}


# Segmentation (data preparation)
# Training
# bar = progressbar.ProgressBar()
# for i in bar([model_setting['train_data'], '_'.join([model_setting['train_data'], 'neutral'])]): # df, df_neu
#     df = pd.read_excel(f'data/raw/data_criminal_%s.xlsx' % i)
#     seg = Segmentation(type="bert")
#     categorical_info = model_setting['factor_lst']+['有利', '中性', '不利'] # modify col names
#     df = seg.segment_criminal_sentiment_analysis_articles_wrapper(df, categorical_info, i)
#     del df, seg
# gc.collect()
# Predict
# df = pd.read_excel(f'data/pred/%s.xlsx' % model_setting['pred_data'])
# seg = Segmentation(type="bert")
# categorical_info = model_setting['factor_lst']+['有利', '中性', '不利'] # modify col names
# df = seg.segment_criminal_sentiment_analysis_articles_wrapper(df, categorical_info, model_setting['pred_data'])
# del df, seg
# gc.collect()


if model_setting['mode'] == 'train_sentence':
    ############# BERT: Classification for criminal sentiment analysis #############
    seed_list = [1234, 5678, 7693145, 947, 13, 27, 1, 5, 9, 277]
    df = pd.read_pickle(f'./data/cleaned/criminal_%s_seg_bert.pkl' % model_setting['train_data'])
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
    bw.evaluate(path=f"%s.txt" % model_setting['train_data'])

elif model_setting['mode'] == 'train_factor':
    ############# BERT: Classification for criminal factor classification #############
    df = pd.read_pickle(f'./data/cleaned/criminal_%s_seg_bert.pkl' % model_setting['train_data'])
    df_neu = pd.read_pickle(f'./data/cleaned/criminal_%s_neutral_seg_bert.pkl' % model_setting['criminal_type'])
    for fac in model_setting['factor_lst']:
        bw = Bert_Wrapper(num_labels = 2)
        trainloader, validloader, testloader = bw.prepare_criminal_judgement_factor_dataloader(df, df_neu, fac)
        bw.initialize_training()
        bw.train()
        bw.evaluate(path=f"%s_%s.txt" % (model_setting['train_data'], fac))
        del trainloader, validloader, testloader , bw
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
        bw = Bert_Wrapper(num_labels = 2)
        trainloader, validloader, testloader = bw.prepare_criminal_judgement_factor_dataloader(df, df_neu, target_feature=fac)
        bw.initialize_training()
        bw.train()  # call trained model or train a new model
        bw.evaluate(path=f"%s_%s.txt" % (model_setting['train_data'], fac))

        predloader = bw.prepare_criminal_judgement_factor_dataloader(df_pred, target_feature=fac, for_prediction=True)
        predictions = bw.predict(predloader)
        df_final[fac] = predictions[:, 1]
        del bw, trainloader, validloader, testloader , predloader, predictions
        gc.collect()
    
    df_final.to_csv(f'data/pred/%s.csv' % model_setting['pred_data'], index=False)


elif model_setting['mode'] == 'pred_sentence':
    ############# BERT: Classification for criminal sentiment analysis #############
    #training data
    df = pd.read_pickle(f'./data/cleaned/criminal_%s_seg_bert.pkl' % model_setting['train_data'])
    df_neu = pd.read_pickle(f'./data/cleaned/criminal_%s_neutral_seg_bert.pkl' % model_setting['criminal_type'])
    # predict data
    df0 = pd.read_csv(f'data/pred/%s.xlsx' % model_setting['pred_data'])    # word
    df_final = pd.read_csv(f'data/pred/%s.xlsx' % model_setting['pred_data'])    # word
    df_pred = pd.read_pickle('./data/cleaned/criminal_%s_seg_bert.pkl' % model_setting['pred_data'])    # vector
    
    
    bw = Bert_Wrapper(num_labels = 3, seed = seed_list[0])
    trainloader, validloader, testloader = bw.prepare_criminal_sentiment_analysis_dataloader(df, df_neu)
    bw.initialize_training()
    bw.train()
    bw.evaluate(path=f"%s.txt" % model_setting['train_data'])