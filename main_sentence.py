# %%
from module.util import *
from src.class_BertForClassification_Pipeline import Pipeline as PL
import argparse
# %%
def main(args):
    # 讀 data
    train_df = external_df = pred_df = pred_sample = None
     
    # For training
    if args.mode == 'train':
        
        if len(args.model_dir) == 0 or len(args.model_name) == 0:
            print('>>>>>Error: In train mode, we have to input model dir and model name to save the model after training.')
            return
        else:
            # 檢查 Project model folder 是否存在
            if os.path.isdir(f"{args.model_dir}"):
                print(">>>>> Model dir(for placing well-trained model) has exist! >>>>>")
            else:
                os.mkdir(f"{args.model_dir}")
                print(">>>>> Model dir hasn't exist! Creating the new model dir(for placing well-trained model)! >>>>>")
            # 如果有 model version, 檢查 Model version 的 folder 是否存在
            if len(args.model_version) > 0:
                if os.path.isdir(f"{args.model_dir}/version_{args.model_version}"):
                    print(">>>>> Model version has exist! >>>>>")
                else:
                    os.mkdir(f"{args.model_dir}/version_{args.model_version}")
                    print(">>>>> Model version hasn't exist! Creating the new version folder! >>>>>")
                model_path = os.path.join(args.model_dir, f"version_{args.model_version}", args.model_name)
            else:
                model_path = os.path.join(args.model_dir, args.model_name)

        if len(args.train_data_path) == 0:
            print('>>>>>Error: In train mode, we have to input train data path to load data.')
            return
        else: 
            train_df = pd.read_excel(args.train_data_path)

        if len(args.external_data_path) > 0:
            external_df = pd.read_excel(args.external_data_path)
        
        # 如果沒有傳入 logging path，則自動按照日期 create 一個 log_path
        if len(args.log_path) <= 0:
            today = str(datetime.date.today())
            if  len(args.model_version) > 0:
                log_path = f"{args.model_dir}/version_{args.model_version}/log_{today}_train.log"
            else:
                log_path = f"{args.model_dir}/log_{today}.log"
        else:
            log_path = args.log_path

        if len(args.label_column_list) > 1:
            # 有利不利中性的 multi class 分類任務
            log_info(info=f"Training pocedure for {args.label_column_list}{len(args.label_column_list)}分類 at {datetime.datetime.now()} \n", path=log_path)
            pl = PL(save_model_path=model_path, pretrained_model_name=args.pretrained_model_name, load_model_path=args.load_model_path, log_path=log_path, \
                    num_labels = len(args.label_column_list), seed = args.seed, \
                    train_test_split_ratio=args.train_test_split_ratio, device=args.device, batch_size = args.batch_size, epoch = args.epoch, \
                    max_seq_len = args.max_seq_len, pooling_strategy = args.pooling_strategy, lr = args.lr, \
                    opt=args.optimizer, scheduler=args.scheduler)
            
            trainloader, validloader, testloader = pl.prepare_dataloader( \
                pl._extract_multiclass_datalst(df=train_df, df_external=external_df, \
                                               x_column_name=args.text_column_name, y_column_list=args.label_column_list,\
                                               external_column_idx=args.external_column_idx, external_sample_num=args.external_sample_num)
            )
            pl.initialize_training()
            pl.train()
            pl.evaluate(path=log_path)
            log_info(info="=======Train End======= \n", path=log_path)


        else:
            # 針對某個 feature column 的二分類任務
            log_info(info=f"Training pocedure for {args.label_column_list}二元分類 at {datetime.datetime.now()} \n", path=log_path)
            pl = PL(save_model_path=model_path, pretrained_model_name=args.pretrained_model_name, load_model_path=args.load_model_path, log_path=log_path, \
                    num_labels = len(args.label_column_list), seed = args.seed, \
                    train_test_split_ratio=args.train_test_split_ratio, device=args.device, batch_size = args.batch_size, epoch = args.epoch, \
                    max_seq_len = args.max_seq_len, pooling_strategy = args.pooling_strategy, lr = args.lr, \
                    opt=args.optimizer, scheduler=args.scheduler)
            
            trainloader, validloader, testloader = pl.prepare_dataloader( \
                pl._extract_binaryclass_datalst(df=train_df, df_external=external_df, \
                                               x_column_name=args.text_column_name, y_column_list=args.label_column_list,\
                                                external_sample_num=args.external_sample_num)
            )
            pl.initialize_training()
            pl.train()
            pl.evaluate(path=log_path)
            log_info(info="=======Train End======= \n", path=log_path)

    elif args.mode == 'test':
        if len(args.test_data_path) <= 0 and len(args.test_sample) <= 0:
            print('>>>>>Error: In test mode, you have to provide test xlsx file or single test sample(a sentence).')
            return
        if len(args.test_data_path) > 0:
            pred_df = pd.read_excel(args.test_data_path) 
        if len(args.test_sample) > 0:
            pred_sample = args.test_sample
        if len(args.load_model_path) <= 0:
            print('>>>>>Error: In test mode, you have to provide a trained model path to predict.')
            return
        if len(args.log_path) <= 0:
            today = str(datetime.date.today())
            if  len(args.model_version) > 0:
                log_path = f"{args.model_dir}/version_{args.model_version}/log_{today}_test.log"
            else:
                log_path = f"{args.model_dir}/log_{today}.log"
        else:
            log_path = args.log_path
        log_info(info=f"Testing for model_path {args.load_model_path}\n", path=log_path)
        pl = PL(pretrained_model_name=args.pretrained_model_name, load_model_path=args.load_model_path, \
                x_column_name=args.text_column_name, y_column_list=args.label_column_list, \
                num_labels = len(args.label_column_list), \
                device=args.device, batch_size = args.batch_size, \
                max_seq_len = args.max_seq_len, pooling_strategy = args.pooling_strategy)
        if len(args.label_column_list) >= 2:
                label_column_list = args.label_column_list
        else:
            label_column_list = [f"不是{args.label_column_list[0]}"] + args.label_column_list
        if pred_df is not None:
            pred_loader = pl.prepare_dataloader(pred_df, for_prediction=True)
            # pred_loader = pl.prepare_dataloader(                pl._extract_multiclass_datalst(df=pred_df, df_external=external_df, \
            #                                    x_column_name=args.text_column_name, y_column_list=args.label_column_list,\
            #                                    external_column_idx=args.external_column_idx, external_sample_num=args.external_sample_num), for_prediction=True)
            # pl.initialize_training()
            predictions = pl.get_predictions(pl.model, pred_loader)
            ## new
            # predictions, _, y_test = pl.get_predictions(pl.model, pred_loader, compute_acc=True)
            # labels = []
            # for i in range(pl.num_labels):
            #     labels.append(i)
            # acc, pre, rc, f1, cm, micro_pre, macro_pre, weighted_pre, micro_rc, macro_rc, weighted_rc, micro_f1, macro_f1, weighted_f1 = compute_performance(y_test.cpu(), predictions.cpu(), labels=labels)
            # log_performance(acc, pre, rc, f1, cm, micro_pre, macro_pre, weighted_pre, micro_rc, macro_rc, weighted_rc, micro_f1, macro_f1, weighted_f1, labels=labels, path=log_path)
            # log_info(info="=======Test End======= \n", path=log_path)
            ## end of new
            # 把 predictions 寫回 excel
            for i, row in pred_df.iterrows():
                for j, label in enumerate(label_column_list):
                    if int(predictions[i]) == int(j):
                        pred_df.loc[i, label] = 1.0
                    else:
                        pred_df.loc[i, label] = 0.0
            # print(pred_df.head())
            # print(args.test_data_path)
            pred_df.to_excel(args.test_data_path.split(".xlsx")[0]+'_answer.xlsx', index=False)
        if pred_sample is not None:
            pl.initialize_training()
            predict = pl.get_predict(pl.model, pred_sample)
            print(f'>>>>>Predicted result for "{pred_sample}": {[label_column_list[predict]]}')


    else:
        print('>>>>>Error: We only have train or test two modes!')

    
def _parse_args():
    parser = argparse.ArgumentParser(description='>>>>>Please enter argument for classification.')
    # For task:
    parser.add_argument("--mode", type=str, default="train", \
                        help="We have two model: train or test.")
    # For Data Path:
    parser.add_argument("--train_data_path", type=str, default="", \
                        help="Please enter the training data path(xlsx) here.")
    parser.add_argument("--test_data_path", type=str, default="", \
                        help="Please enter the testing data path(xlsx) here.")
    parser.add_argument("--external_data_path", type=str, default="", \
                        help="Please enter the external data path here.")
    parser.add_argument("--external_column_idx", type=int, default=-1, \
                        help="Please enter the external column index if you want to add external data. \
                            For example, if you're doing [不利, 有利, 中性] tenary classification and you have external data, then you need to enter 2 to be the colum index.")
    parser.add_argument("--external_sample_num", type=int, default=3000, \
                        help="Please enter the external sample number you want to add in your dataset.")
    parser.add_argument("--test_sample", type=str, default="", \
                        help="Please enter the testing sample(single paragraph) here.")
    parser.add_argument("--log_path", type=str, default="", \
                        help="Please enter the logging data path you want here. If none, logging file will be saved under model_dir with date.")
    # For Data column:
    parser.add_argument("--text_column_name", type=str, default="Sentence", \
                        help="Please enter the text column name(input x) in your data here.")
    parser.add_argument("--label_column_list", nargs='+', default=[], \
                        help="Please enter the label column list(output y) in your data here.")
    # For model saving path:
    parser.add_argument("--train_test_split_ratio", type=float, default=0.2, \
                       help="Please enter the train/test split ratio.")
    parser.add_argument("--model_dir", type=str, default="", \
                        help="Please enter the data folder which model will be saved here.")
    parser.add_argument("--model_name", type=str, default="", \
                        help="Please enter the model name here. The model path will be saved in model_dir with this model name.")
    parser.add_argument("--model_version", type=str, default="", \
                        help="If you have several model versions, you can add this argument. The version will  be added in model saving path.")
    parser.add_argument("--load_model_path", type=str, default="", \
                        help="Please enter the well-trained model path for testing.")
    parser.add_argument("--pretrained_model_name", type=str, default="bert-base-chinese", \
                        help="Noe we only support bert, roberta, macbert pretrained model.\
                            For example, you can use 'bert-base-chinese' to access the chinese bert model, use 'hfl/chinese-roberta-wwm-ext' to access roberta, use 'hfl/chinese-macbert-base' to access macbert.")
    # For tuning model in train phase:]
    parser.add_argument("--device", type=str, default="", \
                        help="Please choose cup or cuda.")
    parser.add_argument("--seed", type=int, default=1234, \
                        help="Please enter the seed you want. This seed will control data splits and the model init weight.")
    parser.add_argument("--max_seq_len", type=int, default=128, \
                        help="Please enter the max input sentence length. The maximum is 512.")
    parser.add_argument("--batch_size", type=int, default=32, \
                        help="Please enter the batch size for train and test data here.")
    parser.add_argument("--epoch", type=int, default=2, \
                        help="Please enter the training epoch here.")
    parser.add_argument("--optimizer", type=str, default="AdamW", \
                        help="Now we support AdamW, Adafactor.")
    parser.add_argument("--lr", type=float, default=2e-5, \
                        help="Please enter the learning rate for optimizer.")
    parser.add_argument("--scheduler", type=str, default="linear", \
                        help="Now we support linear scheduler and Adafactor scheduler.")
    parser.add_argument("--pooling_strategy", type=str, default='reduce_mean', 
                        help="Now we supppoty 'reduce_mean' and 'cls'. ")

    args = parser.parse_args()
    print('>>>>> All args: ', args)
    return args
if __name__ == '__main__':
    args = _parse_args()
    main(args)