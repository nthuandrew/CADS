from module.util import *
from class_Segmentation import Segmentation
from class_BertForClassification_Wrapper import Bert_Wrapper


# model_setting = {'mode': 'train_sentence', 'train_data': 'gun'}
# model_setting = {'mode': 'train_factor', 'train_data': 'gun', 'factor_lst': ['犯罪後之態度', '犯罪所生之危險或違反義務之程度', '被告之品行', '其他審酌事項']}
model_setting = {'mode': 'pred_factor', 'train_data': 'gun', 'pred_data':'全文標註的測試句', 'factor_lst': ['犯罪後之態度', '犯罪所生之危險或違反義務之程度', '被告之品行', '其他審酌事項']}


# Segmentation
# df = pd.read_excel(f'data/raw/data_criminal_%s.xlsx' % model_setting['train_data'])
# seg = Segmentation(type="bert")
# # modify col names
# categorical_info = model_setting['factor_lst']+['有利', '中性', '不利']
# # run
# df = seg.segment_criminal_sentiment_analysis_articles_wrapper(df, categorical_info, model_setting['train_data'])
# del df, seg
# gc.collect()

# if model_setting['mode'] == 'pred_factor':
#     df = pd.read_excel(f'data/pred/%s.xlsx' % model_setting['pred_data'])
#     seg = Segmentation(type="bert")
#     # modify col names
#     categorical_info = model_setting['factor_lst']+['有利', '中性', '不利']
#     # run
#     df = seg.segment_criminal_sentiment_analysis_articles_wrapper(df, categorical_info, model_setting['pred_data'])
#     del df, seg
#     gc.collect()


if model_setting['mode'] == 'train_sentence':
    # BERT: Classification for criminal sentiment analysis
    seed_list = [1234, 5678, 7693145, 947, 13, 27, 1, 5, 9, 277]
    df = pd.read_pickle(f'./data/cleaned/criminal_%s_seg_bert.pkl' % model_setting['train_data'])
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
    bw.evaluate(path=f"%s.txt" % model_setting['train_data'])

elif model_setting['mode'] == 'train_factor':
    # BERT: Classification for criminal factor classification #############
    df = pd.read_pickle(f'./data/cleaned/criminal_%s_seg_bert.pkl' % model_setting['train_data'])
    for fac in model_setting['factor_lst']:
        bw = Bert_Wrapper(num_labels = 2)
        trainloader, validloader, testloader = bw.prepare_criminal_judgement_factor_dataloader(df, fac)
        bw.initialize_training()
        bw.train()
        bw.evaluate(path=f"%s_%s.txt" % (model_setting['train_data'], fac))
        del trainloader, validloader, testloader , bw
        gc.collect()

elif model_setting['mode'] == 'pred_factor':
    # BERT: Classification for criminal factor classification #############
    df = pd.read_pickle(f'./data/cleaned/criminal_%s_seg_bert.pkl' % model_setting['train_data'])
    df_pred = pd.read_pickle('./data/cleaned/criminal_%s_seg_bert.pkl' % model_setting['pred_data'])
    for fac in model_setting['factor_lst']:
        model_name = 'bert_%s_%s' % (model_setting['train_data'], fac)
        bw = Bert_Wrapper(model_name=model_name, num_labels = 2)
        trainloader, validloader, testloader = bw.prepare_criminal_judgement_factor_dataloader(df, fac)
        bw.initialize_training()
        bw.train()

        predloader = bw.prepare_criminal_judgement_factor_dataloader(df, fac, for_prediction=True)
        predictions = bw.predict(predloader)

        bw.evaluate(path=f"%s_%s.txt" % (model_setting['train_data'], fac))
        del trainloader, validloader, testloader , bw
        gc.collect()