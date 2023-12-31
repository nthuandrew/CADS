from pandas.io.pickle import read_pickle
from torch.utils.data import dataloader
from module.util import *
from class_BertForClassification_Model import BertForClassification
from data.GV import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from pandas import Series,DataFrame
from transformers import AdamW, get_linear_schedule_with_warmup
import time
import math
from matplotlib import pyplot as plt
# %%
class Bert_Wrapper():
    def __init__(self, model_dir="/data/model", save_model_name=None, num_labels=2, seed=1234, device="",
    batch_size = 64, epoch = 2, pooling_strategy='reduce_mean', lr=2e-5, max_len = 128
    ):
        '''
        param save_model_name: None or str. If use None, then will not save model
        '''
        self.device = setup_device(device)
        self.seed = seed
        seed_torch(seed=self.seed)
        self.MAX_LENGTH = max_len
        self.NUM_LABELS = num_labels
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epoch
        self.pooling_strategy = pooling_strategy
        self.lr = lr
        self.PRETRAINED_MODEL_NAME = "bert-base-chinese"
        self.model = None
        self.info_dict = {'save_model_name': save_model_name, 'hyper_param':{},'accuracy':None, 'precision':None, 'recall':None, 'f1':None, 'comfusion_matrix':None, \
            'data_preprocess_log': ""}
        self.model_path = f"{model_dir}/{self.info_dict['save_model_name']}.pkl"
        if self.info_dict['save_model_name'] is not None and os.path.exists(self.model_path):
            print(">>>>>Find an exist trained model in our file system!")
            with open(self.model_path, "rb") as file:
                info_dict, model_state_dict = torch.load(file, map_location=self.device)
                # info_dict, model = pickle.load(file)
            # if info_dict['hyper_param'] == self.info_dict['hyper_param']:   # check if hyper_params are the same
                self.info_dict = info_dict
                # self.model = model
                self.model = BertForClassification.from_pretrained(self.info_dict['hyper_param']['PRETRAINED_MODEL_NAME'], \
                    num_labels=self.info_dict['hyper_param']['NUM_LABELS'], \
                    max_length=self.info_dict['hyper_param']['MAX_LENGTH'], \
                    device=self.device, \
                    pooling_strategy=self.info_dict['hyper_param']['POOLING_STRATEGY'])
                self.model.load_state_dict(model_state_dict)
        else:
            print(">>>>>Initial a new model!")
            self.info_dict['hyper_param']['seed'] = self.seed
            self.info_dict['hyper_param']['MAX_LENGTH'] = self.MAX_LENGTH 
            self.info_dict['hyper_param']['NUM_LABELS'] = self.NUM_LABELS 
            self.info_dict['hyper_param']['BATCH_SIZE'] = self.BATCH_SIZE 
            self.info_dict['hyper_param']['EPOCHS'] = self.EPOCHS 
            self.info_dict['hyper_param']['POOLING_STRATEGY'] = self.pooling_strategy
            self.info_dict['hyper_param']['PRETRAINED_MODEL_NAME'] = self.PRETRAINED_MODEL_NAME
            self.info_dict['data_preprocess_log'] = ""
        
        return
    
    # def add_performance(self, acc, pre, rc, f1, cm):
    #     self.info_dict['Performance']['Acc'].append(acc)
    #     self.info_dict['Performance']['Precision'].append(pre)
    #     self.info_dict['Performance']['Recall'].append(rc)
    #     self.info_dict['Performance']['F1-score'].append(f1)
    #     self.info_dict['Performance']['Confusion-Matrix'].append(cm)

    def _create_labeled_data(self, data_list, num_labels=2):
        assert len(data_list) == num_labels
        y = []
        X = []
        for idx, data in enumerate(data_list):
            y += [idx]*len(data)
            X += data
        return X, y

    def _create_cleaned_dataframe(self, X, y):
        df = pd.DataFrame({'y':y,'X':X})
        df = df[~(df.X.apply(lambda x : len(x)) > self.MAX_LENGTH-2)]
        if len(df) == 0:
            print('>>>>>Error: Sentences was truncated due to all of them were over length!')
        return df

    def _create_dataset(self, X, y, type="train"):
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
        data = {
            'label': y,
            'text': X
        }
        df = DataFrame(data)
        df.to_csv(f'./data/cleaned/{type}.csv', index=False)
        df.to_pickle(f'./data/cleaned/{type}.pkl')
        dataset = SentenceDataset(type)
        return dataset

    '''
    通用版本的 dataloader
    '''
    def _extract_bernouli_datalst(self, df, df_neu=None, target_feature=None):
        target_list = []        # get the vectors of the sentences with the target featrue
        manul_other_list = []   # get the vectors of the sentences w/o the target featrue
        auto_other_list_shuffled  = []

        for index, row in df.iterrows():
            if row[target_feature] == True:
                target_list.append(row['Sentence'])
            else:
                manul_other_list.append(row['Sentence'])

        target_list_shuffled = shuffle(target_list, random_state=self.seed)
        # target 的兩倍, manul:auto=6:4
        manul_other_list_shuffled = shuffle(manul_other_list, random_state=self.seed)[:round(1.2*len(target_list_shuffled))]
        if df_neu is not None:
            auto_other_list = [i for i in df_neu['Sentence']]
            auto_other_list_shuffled = shuffle(auto_other_list, random_state=self.seed)[:round(0.8*len(target_list_shuffled))]

        other_list_shuffled = manul_other_list_shuffled + auto_other_list_shuffled


        print("target feature training number:", len(target_list_shuffled))
        print("manul feature training number:", len(manul_other_list_shuffled))
        print("auto feature training number:", len(auto_other_list_shuffled))
        print("other feature training number:", len(other_list_shuffled))
        self.info_dict['data_preprocess_log'] = f"{target_feature} training number:{len(target_list_shuffled)} \n標注資料中的其他標籤(label) training number:{len(manul_other_list_shuffled)} \n額外加入的中性句 training number: {len(auto_other_list_shuffled)} \n最終的其他標籤(label) training number: {len(other_list_shuffled)} \n"
        data_list = [target_list_shuffled, other_list_shuffled]
        return data_list

    def _extract_multiclass_datalst(self, df, df_neu=None, target_feature=None):
        datas = { feature: [] for feature in target_feature }
        for _, row in df.iterrows():
            for idx, feature in enumerate(target_feature):
                if row[feature] == True:
                    datas[feature].append(row['Sentence'])

        datas_shuffled = { feature: [] for feature in target_feature }
        for idx, feature in enumerate(target_feature):
            datas_shuffled[feature] = shuffle(datas[feature], random_state=self.seed)

        auto_neutral_list_shuffled = []
        if df_neu is not None:
            auto_neutral_list = [i for i in df_neu['Sentence']]
            auto_neutral_list_shuffled = shuffle(auto_neutral_list, random_state=self.seed)[:3000]
            print("額外加入中性句 training number:", len(auto_neutral_list_shuffled))

        neutral_list = datas_shuffled[target_feature[-1]] + auto_neutral_list_shuffled
        datas_shuffled[target_feature[-1]] = shuffle(neutral_list, random_state=self.seed)
        
        for i in datas_shuffled:
            print(f"{i} training numer: {len(datas_shuffled[i])}")
            self.info_dict['data_preprocess_log'] += f"{i} training number:{len(datas_shuffled[i])} \n"


        data_list = [ datas_shuffled[data] for data in datas_shuffled]

        # data_list = [disadvantage_list_shuffled, advantage_list_shuffled, neutral_list_shuffled]
        return data_list
    
    def prepare_dataloader(self, extract_datalist=None, for_prediction=False, ):
        '''
        1. Prepare training data for factor classification
        2. To prepare new data for prediction, we need to set df_neu=None and for_prediction=True
        '''
        if for_prediction is False:
            assert extract_datalist is not None
            data_list = extract_datalist
            X, y = self._create_labeled_data(data_list, self.NUM_LABELS)
            df_clean = self._create_cleaned_dataframe(X=X, y=y)
            X, y = df_clean['X'], df_clean['y']
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.seed)
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=self.seed)
            # Create dataset
            trainset = self._create_dataset(X_train, y_train, type="train")
            validset = self._create_dataset(X_valid, y_valid, type="valid")
            testset = self._create_dataset(X_test, y_test, type="test")

            
            self.trainloader = DataLoader(trainset, batch_size=self.BATCH_SIZE, 
                            collate_fn=create_mini_batch)
            self.validloader = DataLoader(validset, batch_size=self.BATCH_SIZE, 
                                    collate_fn=create_mini_batch)
            self.testloader = DataLoader(testset, batch_size=256, 
                                    collate_fn=create_mini_batch)
            return self.trainloader, self.validloader, self.testloader

        else:   # Prediction
            pred_df = self._create_cleaned_dataframe(X=extract_datalist['Sentence'], y=[0]*len(extract_datalist))
            predset = self._create_dataset(pred_df['X'], pred_df['y'], type="pred")
            self.predloader = DataLoader(predset, batch_size=256, 
                                    collate_fn=create_mini_batch)

            return self.predloader

    
    '''
    針對司法院案子量刑資訊系統的 dataloader
    '''

    def _extract_criminal_judgement_factor_datalst(self, df, df_neu=None, target_feature=None):
        target_list = []        # get the vectors of the sentences with the target featrue
        manul_other_list = []   # get the vectors of the sentences w/o the target featrue
        auto_other_list_shuffled  = []

        for index, row in df.iterrows():
            if row[target_feature] == True:
                target_list.append(row['Sentence'])
            else:
                manul_other_list.append(row['Sentence'])

        target_list_shuffled = shuffle(target_list, random_state=self.seed)
        # target 的兩倍, manul:auto=6:4
        manul_other_list_shuffled = shuffle(manul_other_list, random_state=self.seed)[:round(1.2*len(target_list_shuffled))]
        if df_neu is not None:
            auto_other_list = [i for i in df_neu['Sentence']]
            auto_other_list_shuffled = shuffle(auto_other_list, random_state=self.seed)[:round(0.8*len(target_list_shuffled))]

        other_list_shuffled = manul_other_list_shuffled + auto_other_list_shuffled


        print("target feature training number:", len(target_list_shuffled))
        print("manul feature training number:", len(manul_other_list_shuffled))
        print("auto feature training number:", len(auto_other_list_shuffled))
        print("other feature training number:", len(other_list_shuffled))
        self.info_dict['data_preprocess_log'] = f"{target_feature} training number:{len(target_list_shuffled)} \n標注資料中的其他(量刑因子) training number:{len(manul_other_list_shuffled)} \n自動擷取的中性句 training number: {len(auto_other_list_shuffled)} \n最終的其他(量刑因子) training number: {len(other_list_shuffled)} \n"
        data_list = [target_list_shuffled, other_list_shuffled]
        return data_list

    def _extract_criminal_sentiment_analysis_datalst(self, df, df_neu=None, target_feature=None):
        datas = { feature: [] for feature in target_feature }
        for _, row in df.iterrows():
            for idx, feature in enumerate(target_feature):
                if row[feature] == True:
                    datas[feature].append(row['Sentence'])

        datas_shuffled = { feature: [] for feature in target_feature }
        for idx, feature in enumerate(target_feature):
            datas_shuffled[feature] = shuffle(datas[feature], random_state=self.seed)

        auto_neutral_list_shuffled = []
        if df_neu is not None:
            auto_neutral_list = [i for i in df_neu['Sentence']]
            auto_neutral_list_shuffled = shuffle(auto_neutral_list, random_state=self.seed)[:3000]
            print("自動擷取的中性句 training number:", len(auto_neutral_list_shuffled))

        neutral_list = datas_shuffled[target_feature[-1]] + auto_neutral_list_shuffled
        datas_shuffled[target_feature[-1]] = shuffle(neutral_list, random_state=self.seed)
        
        for i in datas_shuffled:
            print(f"{i} training numer: {len(datas_shuffled[i])}")
            self.info_dict['data_preprocess_log'] += f"{i} training number:{len(datas_shuffled[i])} \n"


        data_list = [ datas_shuffled[data] for data in datas_shuffled]

        # data_list = [disadvantage_list_shuffled, advantage_list_shuffled, neutral_list_shuffled]
        return data_list
    
    def prepare_criminal_dataloader(self, extract_datalist=None, for_prediction=False, ):
        '''
        1. Prepare training data for factor classification
        2. To prepare new data for prediction, we need to set df_neu=None and for_prediction=True
        '''
        if for_prediction is False:
            assert extract_datalist is not None
            data_list = extract_datalist
            X, y = self._create_labeled_data(data_list, self.NUM_LABELS)
            df_clean = self._create_cleaned_dataframe(X=X, y=y)
            X, y = df_clean['X'], df_clean['y']
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.seed)
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=self.seed)
            # Create dataset
            trainset = self._create_dataset(X_train, y_train, type="train")
            validset = self._create_dataset(X_valid, y_valid, type="valid")
            testset = self._create_dataset(X_test, y_test, type="test")

            
            self.trainloader = DataLoader(trainset, batch_size=self.BATCH_SIZE, 
                            collate_fn=create_mini_batch)
            self.validloader = DataLoader(validset, batch_size=self.BATCH_SIZE, 
                                    collate_fn=create_mini_batch)
            self.testloader = DataLoader(testset, batch_size=256, 
                                    collate_fn=create_mini_batch)
            return self.trainloader, self.validloader, self.testloader

        else:   # Prediction
            pred_df = self._create_cleaned_dataframe(X=extract_datalist['Sentence'], y=[0]*len(extract_datalist))
            predset = self._create_dataset(pred_df['X'], pred_df['y'], type="pred")
            self.predloader = DataLoader(predset, batch_size=256, 
                                    collate_fn=create_mini_batch)

            return self.predloader

    def prepare_criminal_judgement_factor_dataloader(self, df, \
        df_neu=None, \
        target_feature=None, for_prediction=False):
        '''
        1. Prepare training data for factor classification
        2. To prepare new data for prediction, we need to set df_neu=None and for_prediction=True
        '''
        if for_prediction is False:
            target_list = []        # get the vectors of the sentences with the target featrue
            manul_other_list = []   # get the vectors of the sentences w/o the target featrue
            auto_other_list_shuffled  = []

            for index, row in df.iterrows():
                if row[target_feature] == True:
                    target_list.append(row['Sentence'])
                else:
                    manul_other_list.append(row['Sentence'])

            target_list_shuffled = shuffle(target_list, random_state=self.seed)
            manul_other_list_shuffled = shuffle(manul_other_list, random_state=self.seed)
            if df_neu:
                auto_other_list = [i for i in df_neu['Sentence']]
                auto_other_list_shuffled = shuffle(auto_other_list, random_state=self.seed)[:3000]

            other_list_shuffled = manul_other_list_shuffled + auto_other_list_shuffled


            print("target feature training number:", len(target_list_shuffled))
            print("manul feature training number:", len(manul_other_list_shuffled))
            print("auto feature training number:", len(auto_other_list_shuffled))
            print("other feature training number:", len(other_list_shuffled))
            # logging
            self.info_dict['data_preprocess_log'] = f"{target_feature} training number:{len(target_list_shuffled)} \n 標注資料中的其他(量刑因子) training number:{len(manul_other_list_shuffled)} \n 自動擷取的中性句 training number: {len(auto_other_list_shuffled)} \n 最終的其他(量刑因子) training number: {len(other_list_shuffled)} \n"
            data_list = [target_list_shuffled, other_list_shuffled]
            X, y = self._create_labeled_data(data_list, self.NUM_LABELS)
            df_clean = self._create_cleaned_dataframe(X=X, y=y)
            X, y = df_clean['X'], df_clean['y']
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.seed)
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=self.seed)
            # Create dataset
            trainset = self._create_dataset(X_train, y_train, type="train")
            validset = self._create_dataset(X_valid, y_valid, type="valid")
            testset = self._create_dataset(X_test, y_test, type="test")

            
            self.trainloader = DataLoader(trainset, batch_size=self.BATCH_SIZE, 
                            collate_fn=create_mini_batch)
            self.validloader = DataLoader(validset, batch_size=self.BATCH_SIZE, 
                                    collate_fn=create_mini_batch)
            self.testloader = DataLoader(testset, batch_size=256, 
                                    collate_fn=create_mini_batch)
            return self.trainloader, self.validloader, self.testloader

        else:   # Prediction
            pred_df = self._create_cleaned_dataframe(X=df['Sentence'], y=[0]*len(df))
            predset = self._create_dataset(pred_df['X'], pred_df['y'], type="pred")
            self.predloader = DataLoader(predset, batch_size=256, 
                                    collate_fn=create_mini_batch)

            return self.predloader

        
    # TODO: Murphy prepare dataloader 這邊重複的 code 太多，需要 refactor
    # TODO: Murphy 修改 dataframe 的切法讓他支援 split for class
    # def prepare_criminal_sentiment_analysis_dataloader(self, df, df_neu=None, for_prediction=False):
    #     if for_prediction==False:
    #         target_features = ['有利', '不利' ,'中性']
    #         advantage_list=[]       # get the vectors of the advantage sentences
    #         disadvantage_list=[]    # get the vectors of the disadvantage sentences
    #         manul_neutral_list=[]   # get the vectors of the neutral sentences

    #         # TODO: Murphy 用 outputToList 來重構
    #         for index, row in df.iterrows():
    #             if row[target_features[0]] == True:
    #                 advantage_list.append(row['Sentence'])
    #             elif row[target_features[1]] == True:
    #                 disadvantage_list.append(row['Sentence'])
    #             elif row[target_features[2]] == True:
    #                 manul_neutral_list.append(row['Sentence'])
    #             else:
    #                 print('Sentiment labeled wrong!')
    #                 return

    #         auto_neutral_list = [i for i in df_neu['Sentence']]

    #         advantage_list_shuffled = shuffle(advantage_list, random_state=self.seed)
    #         disadvantage_list_shuffled = shuffle(disadvantage_list, random_state=self.seed)
    #         manul_neutral_list_shuffled = shuffle(manul_neutral_list, random_state=self.seed)
    #         auto_neutral_list_shuffled = shuffle(auto_neutral_list, random_state=self.seed)[:3000]

    #         neutral_list = manul_neutral_list_shuffled + auto_neutral_list_shuffled
    #         neutral_list_shuffled = shuffle(neutral_list, random_state=self.seed)
            
    #         print("advantage(有利句) training number:", len(advantage_list_shuffled))
    #         print("disadvantage（不利句） training number:", len(disadvantage_list_shuffled))
    #         print("manul neutral（人工標注的中性句） training number:", len(manul_neutral_list_shuffled))
    #         print("auto neutral（自動擷取的中性句） training number:", len(auto_neutral_list_shuffled))
    #         print("neutral（最終的中性句） training number:", len(neutral_list_shuffled))
    #         # logging
    #         self.info_dict['data_preprocess_log'] = f"有利句 training number:{len(advantage_list_shuffled)} \n 不利句 training number:{len(disadvantage_list_shuffled)} \n 人工標注的中性句 training number: {len(manul_neutral_list_shuffled)} \n 自動擷取的中性句 training number: {len(auto_neutral_list_shuffled)} \n 最終中性句 training number: {len(neutral_list_shuffled)} \n"
    #         data_list = [disadvantage_list_shuffled, advantage_list_shuffled, neutral_list_shuffled]
    #         X, y = self._create_labeled_data(data_list, self.NUM_LABELS)

    #         df_clean = self._create_cleaned_dataframe(X=X, y=y)
    #         X, y = df_clean['X'], df_clean['y']
    #         # Split data
    #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.seed)
    #         X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=self.seed)
    #         # Create dataset
    #         trainset = self._create_dataset(X_train, y_train, type="train")
    #         validset = self._create_dataset(X_valid, y_valid, type="valid")
    #         testset = self._create_dataset(X_test, y_test, type="test")

    #         self.trainloader = DataLoader(trainset, batch_size=self.BATCH_SIZE, 
    #                         collate_fn=create_mini_batch)
    #         self.validloader = DataLoader(validset, batch_size=self.BATCH_SIZE, 
    #                                 collate_fn=create_mini_batch)
    #         self.testloader = DataLoader(testset, batch_size=256, 
    #                                 collate_fn=create_mini_batch)

    #         return self.trainloader, self.validloader, self.testloader

    #     else:   # Prediction
    #         pred_df = self._create_cleaned_dataframe(X=df['Sentence'], y=[0]*len(df))
    #         predset = self._create_dataset(pred_df['X'], pred_df['y'], type="pred")

    #         self.predloader = DataLoader(predset, batch_size=256, 
    #                                 collate_fn=create_mini_batch)

    #         return self.predloader

    # def prepare_custody_judgement_factor_dataloader(self, df, df_neu, target_features = ["親子感情", "意願能力", "父母經濟"]):
    #     all_features = ["親子感情", "意願能力", "主要照顧", "父母生活", "子女年齡", "子女意願",  "支持系統", "父母經濟"]
    #     neutral_column = df_neu.columns[df_neu.columns.to_series().str.contains('neutral')].tolist()
    #     neutral_list=[]
    #     # For neutral sentence
    #     df_neu.loc[:,neutral_column].apply(output_to_list, content_list=neutral_list)
    #     # TODO: Murphy 這邊需要更 flexible 一點
    #     class0_list = []
    #     class1_list = []
    #     class2_list = []

    #     # TODO: Murphy 可用學長的 outputToList 來重構
    #     for idx, feature in enumerate(target_features):
    #         for index, row in df.iterrows():
    #             text = row["Text"]
    #             if row[feature] == 1 and row['COUNT'] == 1:
    #                 if idx == 0:
    #                     class0_list.append(text)
    #                 elif idx == 1:
    #                     class1_list.append(text)
    #                 else:
    #                     class2_list.append(text)
    #     class0_list_shuffled = shuffle(class0_list, random_state=self.seed)
    #     class1_list_shuffled = shuffle(class1_list, random_state=self.seed)
    #     class2_list_shuffled = shuffle(class2_list, random_state=self.seed)
    #     # [:len(target_list_shuffled)*2]
    #     # reduced training neutral sentence
    #     # n_neutral_samples = int((len(target_list_shuffled)+len(none_target_list_shuffled))/3)
    #     # neutral_list_shuffled = shuffle(neutral_list, random_state=1234)[:n_neutral_samples]

    #     print("class0 training number:", len(class0_list_shuffled))
    #     print("class1 training number:", len(class1_list_shuffled))
    #     print("class2 training number:", len(class2_list_shuffled))


    #     # Encode label
    #     y0 = [0]*(len(class0_list_shuffled))
    #     y1 = [1]*len(class1_list_shuffled)
    #     y2 = [2]*(len(class2_list_shuffled))

    #     y = y0+y1+y2
    #     X = class0_list_shuffled + class1_list_shuffled + class2_list_shuffled

    #     df_clean = pd.DataFrame({'y':y,'X':X})
    #     df_clean = df_clean[~(df_clean.X.apply(lambda x : len(x)) > self.MAX_LENGTH-2)]
    #     X, y = df_clean['X'], df_clean['y']
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.seed)
    #     X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=self.seed)
    #     X_train.reset_index(drop=True, inplace=True)
    #     X_valid.reset_index(drop=True, inplace=True)
    #     X_test.reset_index(drop=True, inplace=True)
    #     y_train.reset_index(drop=True, inplace=True)
    #     y_valid.reset_index(drop=True, inplace=True)
    #     y_test.reset_index(drop=True, inplace=True)

    #     train_data = {
    #         'label': y_train,
    #         'text': X_train
    #     }
    #     train_df = DataFrame(train_data)
    #     train_df.to_csv('./data/cleaned/train.csv', index=False)
    #     train_df.to_pickle('./data/cleaned/train.pkl')
    #     valid_data = {
    #         'label': y_valid,
    #         'text': X_valid
    #     }
    #     valid_df = DataFrame(valid_data)
    #     valid_df.to_csv('./data/cleaned/valid.csv', index=False)
    #     valid_df.to_pickle('./data/cleaned/valid.pkl')
    #     test_data = {
    #         'label': y_test,
    #         'text': X_test
    #     }
    #     test_df = DataFrame(test_data)
    #     test_df.to_csv('./data/cleaned/test.csv', index=False)
    #     test_df.to_pickle('./data/cleaned/test.pkl')

    #     trainset = SentenceDataset("train")
    #     validset = SentenceDataset("valid")
    #     testset = SentenceDataset("test")

    #     self.trainloader = DataLoader(trainset, batch_size=self.BATCH_SIZE, 
    #                      collate_fn=create_mini_batch)
    #     self.validloader = DataLoader(validset, batch_size=self.BATCH_SIZE, 
    #                             collate_fn=create_mini_batch)
    #     self.testloader = DataLoader(testset, batch_size=256, 
    #                             collate_fn=create_mini_batch)

    #     return self.trainloader, self.validloader, self.testloader

        
    # # TODO: Murphy balance_y
    # def prepare_custody_sentiment_analysis_dataloader(self, df, df_neu):
    #     applicant_advantage_column = df.columns[df.columns.to_series().str.contains('AA')].tolist()
    #     respondent_advantage_column = df.columns[df.columns.to_series().str.contains('RA')].tolist()
    #     applicant_disadvantage_column = df.columns[df.columns.to_series().str.contains('AD')].tolist()
    #     respondent_disadvantage_column = df.columns[df.columns.to_series().str.contains('RD')].tolist()
    #     neutral_column = df_neu.columns[df_neu.columns.to_series().str.contains('neutral')].tolist()

    #     advantage_column = applicant_advantage_column + respondent_advantage_column
    #     disadvantage_column = applicant_disadvantage_column + respondent_disadvantage_column

    #     # training sentence set
    #     advantage_list=[]
    #     disadvantage_list=[]
    #     neutral_list=[]

    #     df.loc[:,advantage_column].apply(output_to_list, content_list=advantage_list)
    #     df.loc[:,disadvantage_column].apply(output_to_list, content_list=disadvantage_list)
    #     df_neu.loc[:,neutral_column].apply(output_to_list, content_list=neutral_list)

    #     advantage_list_shuffled = shuffle(advantage_list, random_state=self.seed)
    #     disadvantage_list_shuffled = shuffle(disadvantage_list, random_state=self.seed)
    #     # reduced training neutral sentence
    #     # TODO: Murphy what is this part about?
    #     n_neutral_samples = int((len(advantage_list_shuffled)+len(disadvantage_list_shuffled))/2)
    #     neutral_list_shuffled = shuffle(neutral_list, random_state=self.seed)[:n_neutral_samples]

    #     print("advantage training number:", len(advantage_list_shuffled))
    #     print("disadvantage training number:", len(disadvantage_list_shuffled))
    #     print("neutral training number:", len(neutral_list_shuffled))

    #     # TODO: Murphy equaly split y & X, i.e. line 241 to 277
    #     # ex. loop through (y0, disadvantage_list), (y1, advantage_list)... with train_test_split()
    #     #     and then append() dfs and reset_index().
    #     if self.NUM_LABELS == 2:
    #         # 不利標為0
    #         y0 = [0]*len(disadvantage_list)
    #         # 有利標為1
    #         y1 = [1]*len(advantage_list)
    #         # 二分類
    #         y = y0+y1
    #         X = disadvantage_list + advantage_list
    #     elif self.NUM_LABELS == 3:
    #          # 不利標為0
    #         y0 = [0]*len(disadvantage_list)
    #         # 有利標為1
    #         y1 = [1]*len(advantage_list)
    #         # 中性標為2
    #         y2 = [2]*len(neutral_list)
    #         # 三分類
    #         y = y0+y1+y2
    #         X = disadvantage_list + advantage_list + neutral_list

    #     else:
    #         print('Number of labels seems wrong!')
    #         return

    #     df_clean = pd.DataFrame({'y':y,'X':X})
    #     df_clean = df_clean[~(df_clean.X.apply(lambda x : len(x)) > self.MAX_LENGTH-2)]
    #     X, y = df_clean['X'], df_clean['y']
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.seed)
    #     X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=self.seed)
    #     X_train.reset_index(drop=True, inplace=True)
    #     X_valid.reset_index(drop=True, inplace=True)
    #     X_test.reset_index(drop=True, inplace=True)
    #     y_train.reset_index(drop=True, inplace=True)
    #     y_valid.reset_index(drop=True, inplace=True)
    #     y_test.reset_index(drop=True, inplace=True)

    #     train_data = {
    #         'label': y_train,
    #         'text': X_train
    #     }
    #     train_df = DataFrame(train_data)
    #     train_df.to_csv('./data/cleaned/train.csv', index=False)
    #     train_df.to_pickle('./data/cleaned/train.pkl')
    #     valid_data = {
    #         'label': y_valid,
    #         'text': X_valid
    #     }
    #     valid_df = DataFrame(valid_data)
    #     valid_df.to_csv('./data/cleaned/valid.csv', index=False)
    #     valid_df.to_pickle('./data/cleaned/valid.pkl')
    #     test_data = {
    #         'label': y_test,
    #         'text': X_test
    #     }
    #     test_df = DataFrame(test_data)
    #     test_df.to_csv('./data/cleaned/test.csv', index=False)
    #     test_df.to_pickle('./data/cleaned/test.pkl')

    #     trainset = SentenceDataset("train")
    #     validset = SentenceDataset("valid")
    #     testset = SentenceDataset("test")

    #     self.trainloader = DataLoader(trainset, batch_size=self.BATCH_SIZE, 
    #                      collate_fn=create_mini_batch)
    #     self.validloader = DataLoader(validset, batch_size=self.BATCH_SIZE, 
    #                             collate_fn=create_mini_batch)
    #     self.testloader = DataLoader(testset, batch_size=256, 
    #                             collate_fn=create_mini_batch)

    #     return self.trainloader, self.validloader, self.testloader


    def predict(self, dataloader, compute_acc=False, output_attention=False):
        assert self.model != None
        model = self.model
        predictions = None
        attentions = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            print("")
            print('Predicting...')
            # 遍巡整個資料集
            for data in dataloader: # batches of data
                # 將所有 tensors 移到 GPU 上
                if next(model.parameters()).is_cuda:
                    data = [t.to("cuda:0") for t in data if t is not None]
                
                
                # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
                # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
                tokens_tensors, segments_tensors, masks_tensors, lengths_tensors  = data[:4]
                outputs = model(input_ids=tokens_tensors, 
                                token_type_ids=segments_tensors, 
                                attention_mask=masks_tensors,
                                lengths = lengths_tensors, 
                                output_attention=output_attention
                                )   # choose output function here: sigmoid -> softmax
                if output_attention:
                    logits = outputs[0][0]
                else:
                    logits = outputs[0]     # get sigmoid probability
                
                pred = F.softmax(logits.data, dim = 1)
                    
                # 將當前 batch 記錄下來
                if predictions is None:
                    predictions = pred
                else:
                    predictions = torch.cat((predictions, pred))

                if output_attention:
                    attention = outputs[1]
                    attentions.append(attention)

        return predictions.cpu().numpy()    # turn tensor into numpy array


    def get_predictions(self, model, dataloader, compute_acc=False, output_attention=False):
        predictions = None
        y_truth = None
        attentions = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            # 遍巡整個資料集
            for data in dataloader: # cv data
                # 將所有 tensors 移到 GPU 上
                if next(model.parameters()).is_cuda:
                    data = [t.to("cuda:0") for t in data if t is not None]
                
                
                # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
                # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
                tokens_tensors, segments_tensors, masks_tensors, lengths_tensors  = data[:4]
                outputs = model(input_ids=tokens_tensors, 
                                token_type_ids=segments_tensors, 
                                attention_mask=masks_tensors,
                                lengths = lengths_tensors, 
                                output_attention=output_attention
                                )   # choose output function here: sigmoid -> softmax
                if output_attention:
                    logits = outputs[0][0]
                else:
                    logits = outputs[0]

                prob = F.softmax(logits.data, dim = 1)
                _, pred = torch.max(prob, 1)

                # _, pred = torch.max(logits.data, 1)
                # 用來計算訓練集的分類準確率
                if compute_acc:
                    labels = data[4]
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()
                    if y_truth is None:
                        y_truth = labels
                    else:
                        y_truth = torch.cat((y_truth, labels))
                    
                    
                # 將當前 batch 記錄下來
                if predictions is None:
                    predictions = pred
                else:
                    predictions = torch.cat((predictions, pred))

                if output_attention:
                    attention = outputs[1]
                    attentions.append(attention)
                
        
        if compute_acc:
            acc = correct / total
            return predictions, acc, y_truth
        if output_attention:
            return predictions, attentions
        else:
            return predictions

    def initialize_training(self):
        '''
        Model hyper-parameter setting? #TODO: csu ask Murphy
        '''
        self.model = BertForClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=self.NUM_LABELS, max_length=self.MAX_LENGTH, device=self.device, pooling_strategy=self.pooling_strategy)
        self.training_stats = []
        self.model.to(self.device)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # optimizer_grouped_parameters
        self.optimizer = AdamW(optimizer_grouped_parameters,
                        lr = self.lr, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                        eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                        )
        ################# Method 1 ######################
        # gradient_accumulation_steps = 1
        # # total_steps = len(self.trainloader) * EPOCHS
        # t_total = int(len(self.trainloader) / gradient_accumulation_steps * EPOCHS)

        # # Create the learning rate scheduler.
        # scheduler = get_linear_schedule_with_warmup(optimizer, 
        #                                             num_warmup_steps = t_total * 0.1, # Default value in run_glue.py
        #                                             num_training_steps = t_total)
        ################# Method 2 #######################
        total_steps = len(self.trainloader) * self.EPOCHS

        # Create the learning rate scheduler.
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                    num_warmup_steps = 0, # Default value in run_glue.py
                                                    num_training_steps = total_steps)
    def train(self):
        '''
        Train model with/without trainning data? # TODO: csu ask Murphy
        '''  # get the exit model path
        # import the exist model
        # Murphy: 這邊即便傳入 save_model_name 但如果 file not exist 的話，就不會跑 train 的流程了
        if self.info_dict['save_model_name'] is not None and os.path.exists(self.model_path):
            with open(self.model_path, "rb") as file:
                info_dict, model_state_dict = torch.load(file, map_location=self.device)
                # info_dict, model = pickle.load(file)
            # if info_dict['hyper_param'] == self.info_dict['hyper_param']:   # check if hyper_params are the same
                self.info_dict = info_dict
                # self.model = model
                self.model.load_state_dict(model_state_dict)
        
        # train a new model
        else:
            for epoch in range(self.EPOCHS):
                print("")
                print('Training...')
                t0 = time.time()
                self.model.train()
                running_train_loss = 0.0
                for data in self.trainloader:
                    
                    tokens_tensors, segments_tensors, \
                    masks_tensors, lengths_tensors, labels = [t.to(self.device) for t in data]

                    # 將參數梯度歸零
                    self.optimizer.zero_grad()
                    
                    # forward pass
                    outputs = self.model(input_ids=tokens_tensors, 
                                    token_type_ids=segments_tensors, 
                                    attention_mask=masks_tensors, 
                                    lengths = lengths_tensors,
                                    labels=labels)
                    
                    loss = outputs[0]
                    # backward
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()


                    # 紀錄當前 batch loss
                    running_train_loss += loss.item()
                    
                # 計算分類準確率
                _, train_acc, _ = self.get_predictions(self.model, self.trainloader, compute_acc=True)
                avg_running_train_loss = running_train_loss / len(self.trainloader)
                training_time = format_time(time.time() - t0)
                print('Train>>>[epoch %d] loss: %.3f, acc: %.3f' %
                    (epoch + 1, avg_running_train_loss, train_acc))
                # ========================================
                #               Validation
                # ========================================
                # After the completion of each training epoch, measure our performance on
                # our validation set.

                print("")
                print("Running Validation...")
                t0 = time.time()
                self.model.eval()
                running_valid_loss = 0.0
                for data in self.validloader:
                    tokens_tensors, segments_tensors, \
                    masks_tensors, lengths_tensors, labels, = [t.to(self.device) for t in data]
                    with torch.no_grad():   
                        outputs = self.model(input_ids=tokens_tensors, 
                                        token_type_ids=segments_tensors, 
                                        attention_mask=masks_tensors, 
                                        lengths=lengths_tensors,
                                        labels=labels)
                        loss = outputs[0]
                        running_valid_loss += loss.item()

                _, valid_acc, _ = self.get_predictions(self.model, self.validloader, compute_acc=True)
                avg_running_valid_loss = running_valid_loss / len(self.validloader)
                validation_time = format_time(time.time() - t0)
                print('Valid>>>[epoch %d] loss: %.3f, acc: %.3f' %
                    (epoch + 1, avg_running_valid_loss, valid_acc))

                # Record all statistics from this epoch.
                self.training_stats.append(
                    {
                        'epoch': epoch + 1,
                        'train_loss': avg_running_train_loss,
                        'valid_loss': avg_running_valid_loss,
                        'train_acc': train_acc,
                        'valid_acc': valid_acc,
                        'train_time': training_time,
                        'valid_time': validation_time
                    }
                )
            
            # save model
            if self.info_dict['save_model_name'] is not None:
                torch.save([self.info_dict, self.model.state_dict()], file=self.model_path)
                print(f'>>>>> Finish training! Save model at {self.model_path} >>>>>')
                # with open(self.model_path, "wb") as file: # save model
                #     pickle.dump([self.info_dict, self.model], file=file)
                #     print(f'>>>>> Finish training! Save model at {self.model_path} >>>>>')
            
        return

    # TODO: Murphy 支援在 terminal 上 plot 或是另存下 learning curve
    def plot_learning_curve(self):
        # prepare to plot learning curve
        train_acc_list = []
        valid_acc_list = []
        train_loss_list = []
        valid_loss_list = []
        for state in self.training_stats:
            train_acc_list.append(state['train_acc'])
            valid_acc_list.append(state['valid_acc'])
            train_loss_list.append(state['train_loss'])
            valid_loss_list.append(state['valid_loss'])
        # Plot learning curve
        plt.style.use('dark_background')
        plt.plot(train_acc_list)
        plt.plot(valid_acc_list)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        plt.plot(train_loss_list)
        plt.plot(valid_loss_list)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        return

    def evaluate(self, path="output.txt"):
        # TODO: Murphy 需要 output data 數量, 模型 setting 等資訊
        # predictions, attention = get_predictions(model, testloader, output_attention=True)
        print('Start evaluate...')
        path_ = "data/result/"+path
        print(path_)
        labels = []
        for i in range(self.NUM_LABELS):
            labels.append(i)
        predictions, _, y_test = self.get_predictions(self.model, self.testloader, compute_acc=True, output_attention=False)
        # y_test = pd.read_csv("data/cleaned/" + 'test' + ".csv", dtype=int).fillna("")['label']
        # y_test = pd.read_pickle("./data/cleaned/" + 'test' + ".pkl").fillna("")['label']
        acc, pre, rc, f1, cm = compute_performance(y_test.cpu(), predictions.cpu(), labels=labels)
        log_performance(acc, pre, rc, f1, cm, labels=labels, path=path)
        return acc, pre, rc, f1, cm






if __name__=='__main__':
    # TODO: Murphy 自動跑 10 次取平均的 script

    print('Bert Classification Wrapper...')

    ############ Classification for custody sentiment analysis #############
    # df = pd.read_pickle('./data/cleaned/judgment_result_seg_bert.pkl')
    # df_neu = pd.read_pickle('./data/cleaned/judgment_result_seg_neu_bert.pkl')
    # bw = Bert_Wrapper(num_labels = 2)
    # trainloader, validloader, testloader = bw.prepare_custody_sentiment_analysis_dataloader(df, df_neu)
    # bw.initialize_training()
    # bw.train()
    # bw.evaluate()
    ############ END #############

    ############ Classification for judgement factor #############
    # df = pd.read_pickle('./data/cleaned/judgment_factor_seg_bert.pkl')
    # df_neu = pd.read_pickle('./data/cleaned/judgment_factor_seg_neu_bert.pkl')
    # bw = Bert_Wrapper(num_labels = 3)
    # trainloader, validloader, testloader = bw.prepare_custody_judgement_factor_dataloader(df, df_neu)
    # bw.initialize_training()
    # bw.train()
    # bw.evaluate()
    ############ END #############

    ############ Classification for criminal sentiment analysis #############
    criminal_type="sex"
    seed_list = [1234, 5678, 7693145, 947, 13, 27, 1, 5, 9, 277]
    df = pd.read_pickle(f'./data/cleaned/criminal_{criminal_type}_seg_bert.pkl')
    df_neu = pd.read_pickle(f'./data/cleaned/criminal_{criminal_type}_neutral_seg_bert.pkl')
    # for i in range(10):
    #     print("Start test:", i )
    #     bw = Bert_Wrapper(num_labels = 3, seed = seed_list[i])
    #     trainloader, validloader, testloader = bw.prepare_criminal_sentiment_analysis_dataloader(df)
    #     bw.initialize_training()
    #     bw.train()
    #     acc, pre, rc, f1, cm = bw.evaluate(path=f"{criminal_type}.txt")
    #     bw.add_performance(acc, pre, rc, f1, cm)

    # get_average_performance(bw.info_dict['Performance'])

    bw = Bert_Wrapper(num_labels = 3)
    trainloader, validloader, testloader = bw.prepare_criminal_sentiment_analysis_dataloader(df, df_neu)
    bw.initialize_training()
    bw.train()
    acc, pre, rc, f1, cm = bw.evaluate(path=f"{criminal_type}.txt")
    bw.add_performance(acc, pre, rc, f1, cm)
    ############ END #############

    ############ Classification for criminal factor classification #############
    # criminal_type="sex"
    # seed_list = [1234, 5678, 7693145, 947, 13, 27, 1, 5, 9, 277]
    # df = pd.read_pickle(f'./data/cleaned/criminal_{criminal_type}_seg_bert.pkl')
    # for i in range(5):
    #     print("Start test:", i )
    #     bw = Bert_Wrapper(num_labels = 2, seed = seed_list[i])
    #     trainloader, validloader, testloader = bw.prepare_criminal_judgement_factor_dataloader(df, '其他審酌事項')
    #     bw.initialize_training()
    #     bw.train()
    #     acc, pre, rc, f1, cm = bw.evaluate(path=f"{criminal_type}.txt")
    #     bw.add_performance(acc, pre, rc, f1, cm)

    # get_average_performance(bw.info_dict['Performance'])

    # bw = Bert_Wrapper(num_labels = 2)
    # trainloader, validloader, testloader = bw.prepare_criminal_judgement_factor_dataloader(df, df_neu, '犯罪後之態度')
    # bw.initialize_training()
    # bw.train()
    # acc, pre, rc, f1, cm = bw.evaluate(path=f"{criminal_type}.txt")
    ############ END #############

    ######## TEST #########
    
    # %%
    




# %%
