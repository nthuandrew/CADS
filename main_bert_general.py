# %%
from module.util import *
from class_Segmentation import Segmentation
from class_BertForClassification_Wrapper import Bert_Wrapper
import argparse
# %%
parser = argparse.ArgumentParser(description='Still thinking...')
parser.add_argument("--mode", type=str, default="train_multiclass")
parser.add_argument("--train_data", type=str, default="")
parser.add_argument("--pred_data", type=str, default="")
parser.add_argument("--neutral_data", type=str, default="")
parser.add_argument("--factors", type=str, action='append')
# 要不要斷詞
parser.add_argument("--do_segment", action='store_true') 
# Project name
parser.add_argument("--project_name", type=str, default="Test")
# Model 的版本
parser.add_argument("--version", type=float, default=1.0)
# Model save name
parser.add_argument("--model_dir", type=str, default="")
parser.add_argument("--model_name", type=str, default="")
# Model setting
parser.add_argument("--max_len", type=int, default=128)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epoch", type=int, default=2)
parser.add_argument("--pooling_strategy", type=str, default='reduce_mean')
parser.add_argument("--lr", type=float, default=2e-5)

args = parser.parse_args()
print('>>>> All args: ', args)
model_setting = {'mode': args.mode, \
'train_data': f'{args.train_data}', 'pred_data': f'{args.pred_data}', \
'neutral_data': f'{args.neutral_data}', 'factor_lst': args.factors}
print('>>>>> All model setting: ', model_setting)
# %%
'''
Define factor list for different cases.
'''
factor_lst = args.factors
# %%
'''
斷詞
'''
if args.do_segment:
    bar = progressbar.ProgressBar()
    for i in bar([model_setting['train_data'], model_setting['neutral_data'], model_setting['pred_data']]): # df, df_neu
        if len(i) > 0:
            print(f'>>>>> Segmenting training {i}.xlsx data >>>>>')
            df = pd.read_excel(f'./data/raw/%s.xlsx' % i)
            seg = Segmentation(type="bert")
            categorical_info = model_setting['factor_lst'] # modify col names
            df = seg.segment_wrapper(df, categorical_info, output_filename=f'{model_setting["mode"]}_{i}')
            del df, seg
            gc.collect()
# %%
'''
斷詞之後，進入訓練流程
'''
# %%
# df, df_neu 拿到這邊宣告
df = pd.read_pickle(f'./data/cleaned/{model_setting["mode"]}_{model_setting["train_data"]}_seg_bert.pkl')
if len(args.neutral_data) > 0:
    df_neu = pd.read_pickle(f'./data/cleaned/{model_setting["model"]}_{model_setting["neutral_data"]}_seg_bert.pkl')
else:
    df_neu = None

# 檢查 Project model folder 是否存在
if os.path.isdir(f"{args.model_dir}/{args.project_name}"):
    print(">>>>> Project folder(for placing well-trained model) has exist! >>>>>")
else:
    os.mkdir(f"{args.model_dir}/{args.project_name}")
    print(">>>>> Project folder hasn't exist! Creating the new project folder(for placing well-trained model)! >>>>>")
# 檢查 Model version 是否存在
if os.path.isdir(f"{args.model_dir}/{args.project_name}/version_{args.version}"):
    print(">>>>> Model version has exist! >>>>>")
else:
    os.mkdir(f"{args.model_dir}/{args.project_name}/version_{args.version}")
    print(">>>>> Model version hasn't exist! Creating the new version folder! >>>>>")

# 建立日期
today = str(datetime.date.today())
log_output_path = f"{args.project_name}_{model_setting['mode']}_{model_setting['train_data']}_version{args.version}.txt"

# %%
if model_setting['mode'] == 'train_multiclass':
    log_info(info=f"Training result for {model_setting['factor_lst']}句子{len(model_setting['factor_lst'])}分類 at {datetime.datetime.now()} \n", \
             path="./data/result/"+log_output_path)
    model_name = f"{args.project_name}/version_{args.version}/{model_setting['train_data']}_{model_setting['mode']}_epoch{args.epoch}_seed{args.seed}_{today}"
    bw = Bert_Wrapper(save_model_name=model_name, num_labels = len(model_setting['factor_lst']), seed = args.seed,
                      batch_size = args.batch_size, epoch = args.epoch, max_len = args.max_len,
                      pooling_strategy = args.pooling_strategy,
                      lr = args.lr)
    # trainloader, validloader, testloader = bw.prepare_criminal_sentiment_analysis_dataloader(df, df_neu)
    trainloader, validloader, testloader = bw.prepare_dataloader(
        bw._extract_multiclass_datalst(df=df, df_neu=df_neu, target_feature=model_setting['factor_lst'])
    )
    log_info(info=bw.info_dict['data_preprocess_log'], path="data/result/"+log_output_path)
    bw.initialize_training()
    bw.train()
    bw.evaluate(path=log_output_path)
    log_info(info="================ \n", path="data/result/"+log_output_path)

elif model_setting['mode'] == 'train_bernouli':
    for fac in model_setting['factor_lst']:
        log_info(info=f"Training result for label:{fac} at {datetime.datetime.now()} \n", path="data/result/"+log_output_path)
        model_name = f"{args.project_name}/version_{args.version}/{model_setting['train_data']}_{model_setting['mode']}_{fac}_epoch{args.epoch}_seed{args.seed}_{today}"
        bw = Bert_Wrapper(save_model_name=model_name, num_labels = 2, seed = args.seed,
                          batch_size = args.batch_size, epoch = args.epoch, max_len = args.max_len,
                          pooling_strategy = args.pooling_strategy,
                          lr = args.lr)
        # trainloader, validloader, testloader = bw.prepare_criminal_judgement_factor_dataloader(df, df_neu, fac)
        trainloader, validloader, testloader = bw.prepare_dataloader(
        bw._extract_bernouli_datalst(df, df_neu, target_feature=fac)
        )
        log_info(info=bw.info_dict['data_preprocess_log'], path="data/result/"+log_output_path)
        bw.initialize_training()
        bw.train()
        bw.evaluate(path=log_output_path)
        log_info(info="================ \n", path="data/result/"+log_output_path)
        del trainloader, validloader, testloader , bw
        gc.collect()

else:
    print(">>>>> Mode error! Now we only support options for 'train_multiclass' and 'train_bernouli'! >>>>> ")

# elif model_setting['mode'] == 'pred_sentence':
#     ############# BERT: Classification for criminal sentiment analysis #############
#     #training data
#     df = pd.read_pickle(f'./data/cleaned/criminal_%s_seg_bert.pkl' % model_setting['train_data'])
#     df_neu = pd.read_pickle(f'./data/cleaned/criminal_%s_neutral_seg_bert.pkl' % model_setting['train_data'])
#     # predict data
#     df_final = pd.read_excel(f'data/pred/%s.xlsx' % model_setting['pred_data'])    # word
#     df_pred = pd.read_pickle('./data/cleaned/criminal_%s_seg_bert.pkl' % model_setting['pred_data'])    # vector
    
#     model_name = f"{model_setting['train_data']}_{model_setting['mode']}_epoch2_seed1234_1128"
#     bw = Bert_Wrapper(save_model_name=model_name, num_labels = 3)
#     # trainloader, validloader, testloader = bw.prepare_criminal_sentiment_analysis_dataloader(df, df_neu)
#     trainloader, validloader, testloader = bw.prepare_criminal_dataloader(
#         bw._extract_criminal_sentiment_analysis_datalst(df, df_neu, ['不利', '有利', '中性'])
#     )
#     bw.initialize_training()
#     bw.train()
#     bw.evaluate(path=f"%s.txt" % model_setting['train_data'])

#     # predloader = bw.prepare_criminal_sentiment_analysis_dataloader(df_pred, for_prediction=True)
#     predloader = bw.prepare_criminal_dataloader(df_pred, for_prediction=True)
#     predictions = bw.predict(predloader)
#     df_final['不利'] = predictions[:, 0]
#     df_final['有利'] = predictions[:, 1]
#     df_final['中性'] = predictions[:, 2]
#     df_final.to_excel(f'data/pred/%s.xlsx' % model_setting['pred_data'], index=False)
#     del bw, trainloader, validloader, testloader , predloader, predictions
#     gc.collect()

# elif model_setting['mode'] == 'pred_factor':
#     ############# BERT: Classification for criminal factor classification #############
#     # training data
#     df = pd.read_pickle(f'./data/cleaned/criminal_%s_seg_bert.pkl' % model_setting['train_data'])   # vector
#     df_neu = pd.read_pickle(f'./data/cleaned/criminal_%s_neutral_seg_bert.pkl' % model_setting['train_data'])   # vector
#     # predict data
#     df_final = pd.read_excel(f'data/pred/%s.xlsx' % model_setting['pred_data'])    # word
#     df_pred = pd.read_pickle('./data/cleaned/criminal_%s_seg_bert.pkl' % model_setting['pred_data'])    # vector
    
#     bar = progressbar.ProgressBar()
#     for fac in bar(model_setting['factor_lst']):
#         # m_name = 'bert_%s_%s' % (model_setting['train_data'], fac)
#         # bw = Bert_Wrapper(save_model_name=m_name, num_labels = 2)
#         model_name = f"{model_setting['train_data']}_{model_setting['mode']}_{fac}_epoch2_seed1234_1128"
#         bw = Bert_Wrapper(save_model_name=model_name, num_labels = 2)
#         # trainloader, validloader, testloader = bw.prepare_criminal_judgement_factor_dataloader(df, df_neu, target_feature=fac)
#         trainloader, validloader, testloader = bw.prepare_criminal_dataloader(
#         bw._extract_criminal_judgement_factor_datalst(df, df_neu, target_feature=fac)
#         )
#         bw.initialize_training()
#         bw.train()  # call trained model or train a new model
#         bw.evaluate(path=f"%s_%s.txt" % (model_setting['train_data'], fac))

#         # predloader = bw.prepare_criminal_judgement_factor_dataloader(df_pred, target_feature=fac, for_prediction=True)
#         predloader = bw.prepare_criminal_dataloader(df_pred, for_prediction=True)
#         predictions = bw.predict(predloader)
#         df_final[fac] = predictions[:, 0]
#         del bw, trainloader, validloader, testloader , predloader, predictions
#         gc.collect()
    
#     df_final.to_excel(f'data/pred/%s.xlsx' % model_setting['pred_data'], index=False)
