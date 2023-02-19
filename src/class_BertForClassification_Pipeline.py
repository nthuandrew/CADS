from pandas.io.pickle import read_pickle
from torch.utils.data import Dataset, dataloader
from module.util import *
from src.class_BertForClassification_Model import BertForClassification
from data.GV import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from pandas import Series,DataFrame
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.optimization import Adafactor, AdafactorSchedule
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer
import time
import math
from matplotlib import pyplot as plt
from transformers import BertTokenizer
from collections import Counter
from zhon.hanzi import punctuation
import string
# %%
def txt_to_clean(input_str):
    # 去除頭尾的中文標點符號
    input_str = input_str.lstrip(punctuation)
    input_str = input_str.rstrip(punctuation)
    # 去除頭尾的英文標點符號
    input_str = input_str.lstrip(string.punctuation)
    input_str = input_str.rstrip(string.punctuation)
    return input_str
# %%
class SentenceDataset(Dataset):
    # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self, X, y, mode, tokenizer, max_seq_len):
        assert mode in ["train", "test", "valid"]  # 一般訓練你會需要 dev set
        self.mode = mode
        self.X = X
        self.y = y
        self.len = len(self.y)
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer
    
    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        if self.mode == "test":
            text = txt_to_clean(self.X[idx])
            label_tensor = None
        else:
            label = self.y[idx]
            text = txt_to_clean(self.X[idx])
            label_id = label
            label_tensor = torch.tensor(label_id)

        tokenized = self.tokenizer(text, padding="max_length", max_length=self.max_seq_len, truncation="longest_first")
        tokens_tensor = tokenized['input_ids']
        segments_tensor = tokenized['token_type_ids']
        mask_tensor = tokenized['attention_mask']
        
        return (tokens_tensor, segments_tensor, mask_tensor, label_tensor)
    
    def __len__(self):
        return self.len
# %%
class Pipeline():
    def __init__(self, save_model_path="", pretrained_model_name="bert-base-chinese", load_model_path="", log_path="", \
                x_column_name="", y_column_list=[], \
                num_labels=2, seed=1234, train_test_split_ratio=0.2, device="", batch_size = 64, epoch = 3, \
                pooling_strategy='reduce_mean', lr=2e-5, max_seq_len = 128, opt="AdamW", scheduler="liner"
    ):
        '''
        param save_model_name: None or str. If use None, then will not save model
        '''
        if len(device) == 0:
            self.device = setup_device(device)
        self.seed = seed
        seed_torch(seed=self.seed)
        self.x_column_name = x_column_name
        self.y_column_list = y_column_list
        self.max_seq_len = max_seq_len
        if num_labels < 2:
            self.num_labels = 2
        else:
            self.num_labels = num_labels
        self.batch_size = batch_size
        self.epochs = epoch
        self.train_test_split_ratio = train_test_split_ratio
        self.pooling_strategy = pooling_strategy
        self.lr = lr
        # Opt, scheduler, model 到 initialize 那邊再詳細宣告
        self.opt = opt
        self.sche = scheduler
        self.pretrained_model_name = pretrained_model_name
        self.model = None
        self.save_model_path = None
        if len(save_model_path)> 0:
            self.save_model_path = f"{save_model_path}.pkl"
        if 'bert' in self.pretrained_model_name:
            self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_name)
        elif 'roberta' in self.pretrained_model_name:
            self.tokenizer = RobertaTokenizerFast.from_pretrained(self.pretrained_model_name)
        elif 'macbert' in self.pretrained_model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        else:
            print('>>>>>Error: We now only support bert, roberta, macbert to do classification.')
            return
        self.load_model_path = load_model_path
        self.log_path = log_path
        self.trainloader = self.validloader = self.testloader = self.predloader = None
    def _create_labeled_data(self, data_list, num_labels=2):
        assert len(data_list) == num_labels
        y = []
        X = []
        for idx, data in enumerate(data_list):
            y += [idx]*len(data)
            X += data
        df = pd.DataFrame({'y':y,'X':X})
        return df, X, y
    
    def _create_dataset(self, X, y, type="train"):
        dataset = SentenceDataset(X, y, type, self.tokenizer, self.max_seq_len)
        return dataset
    
    def _create_mini_batch(self, samples):
        tokens_tensors = torch.LongTensor([s[0] for s in samples])
        segments_tensors = torch.LongTensor([s[1] for s in samples])
        mask_tensors = torch.LongTensor([s[2] for s in samples]) 
        # 測試集有 labels
        if samples[0][3] is not None:
            label_ids = torch.stack([s[3] for s in samples])
        else:
            label_ids = None
        return tokens_tensors, segments_tensors, mask_tensors, label_ids
    
    def _extract_multiclass_datalst(self, df, df_external=None, x_column_name='Sentence', y_column_list=None, \
                                    external_column_idx=-1, external_sample_num=3000):
        dataset = { feature: [] for feature in y_column_list}
        for _, row in df.iterrows():
            for idx, feature in enumerate(y_column_list):
                if row[feature] == True:
                    dataset[feature].append(row[x_column_name])

        dataset_shuffled = { feature: [] for feature in y_column_list }
        for idx, feature in enumerate(y_column_list):
            dataset_shuffled[feature] = shuffle(dataset[feature], random_state=self.seed)

        auto_external_list_shuffled = []
        if df_external is not None:
            auto_external_list = [i for i in df_external[x_column_name]]
            auto_external_list_shuffled = shuffle(auto_external_list, random_state=self.seed)[:external_sample_num]
            print("額外加入中性句 training number:", len(auto_external_list_shuffled))

        external_list = dataset_shuffled[y_column_list[external_column_idx]] + auto_external_list_shuffled
        dataset_shuffled[y_column_list[external_column_idx]] = shuffle(external_list, random_state=self.seed)
        
        for idx, i in enumerate(dataset_shuffled):
            print(f"{i} total sample numer, which label is '{idx}': {len(dataset_shuffled[i])}")
        data_list = [ dataset_shuffled[data] for data in dataset_shuffled]
 
        return data_list
    
    def _extract_binaryclass_datalst(self, df, df_external=None, x_column_name='Sentence', y_column_list=None, \
                                    external_sample_num=3000):
        assert len(y_column_list) == 1
        target_list = []        # get the vectors of the sentences with the target featrue
        manul_other_list = []   # get the vectors of the sentences w/o the target featrue
        auto_other_list_shuffled  = []

        for _, row in df.iterrows():
            if row[y_column_list[0]] == True:
                target_list.append(row[x_column_name])
            else:
                manul_other_list.append(row[x_column_name])

        target_list_shuffled = shuffle(target_list, random_state=self.seed)

        manul_other_list_shuffled = shuffle(manul_other_list, random_state=self.seed)
        if df_external is not None:
            auto_other_list = [i for i in df_external[x_column_name]]
            auto_other_list_shuffled = shuffle(auto_other_list, random_state=self.seed)[:external_sample_num]
        
        other_list_shuffled = manul_other_list_shuffled + auto_other_list_shuffled


        print(f"是 {y_column_list} 的 total sample number:", len(target_list_shuffled))
        print(f"不是 {y_column_list} 的 total sample number:", len(manul_other_list_shuffled))
        print(f"從中性句中找出來的 不是 {y_column_list} total sample number:", len(auto_other_list_shuffled))
        print(f"所有不是 {y_column_list} total sample number:", len(other_list_shuffled))
        print(f'>>>>>不是 {y_column_list} 的資料被 label 為 "0"，是{y_column_list} 的資料被 label 為 "1"。')
        data_list = [other_list_shuffled, target_list_shuffled]
        return data_list

    def prepare_dataloader(self, extract_datalist=None, for_prediction=False):
        '''
        1. Prepare training data for factor classification
        2. To prepare new data for prediction, we need to set df_external=None and for_prediction=True
        '''
        if for_prediction is False:
            assert extract_datalist is not None
            data_list = extract_datalist
            df, X, y = self._create_labeled_data(data_list, self.num_labels)
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.train_test_split_ratio, random_state=self.seed)
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=self.train_test_split_ratio, random_state=self.seed)
            # Log train/valid/test dataset info
            log_info(info=">>>>>Dataset info:", path=self.log_path)
            log_info(info=f"The len of trainset is {len(X_train)}. The label distribution in trainset is {Counter(y_train)}.\nThe len of validset is {len(X_valid)}. The label distribution in validset is {Counter(y_valid)}.\nThe len of testset is {len(X_test)}. The label distribution in testset is {Counter(y_test)}.\n", path=self.log_path)
            # Create dataset
            trainset = self._create_dataset(X_train, y_train, type="train")
            validset = self._create_dataset(X_valid, y_valid, type="valid")
            testset = self._create_dataset(X_test, y_test, type="valid")

            
            self.trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True,
                            collate_fn=self._create_mini_batch)
            self.validloader = DataLoader(validset, batch_size=self.batch_size, shuffle=False,
                                    collate_fn=self._create_mini_batch)
            self.testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False,
                                    collate_fn=self._create_mini_batch)
            return self.trainloader, self.validloader, self.testloader

        else:   # Prediction
            pred_df = extract_datalist
            predset = self._create_dataset(pred_df[self.x_column_name], [0]*len(pred_df), type="test")
            self.predloader = DataLoader(predset, batch_size=self.batch_size, shuffle=False, 
                                    collate_fn=self._create_mini_batch)

            return self.predloader

    def initialize_training(self):
        '''
        Model hyper-parameter setting
        '''
        # Log model information
        log_info(info=">>>>>Model info:", path=self.log_path)
        log_info(info=f"1. Model type is {self.pretrained_model_name}, run for {self.epochs} epochs, optimizer is {self.opt}, scheduler is {self.sche}, learning rate is {self.lr}, batch size is {self.batch_size}, seed is {self.seed}. \n", path=self.log_path)
        
        if 'bert' in self.pretrained_model_name:
            self.model = BertForClassification.from_pretrained(self.pretrained_model_name, num_labels=self.num_labels, \
                                                               max_length=self.max_seq_len, device=self.device, pooling_strategy=self.pooling_strategy)
        elif 'roberta' in self.pretrained_model_name:
            self.model = RobertaForSequenceClassification.from_pretrained(self.pretrained_model_name, num_labels=self.num_labels)
        elif 'macbert' in self.pretrained_model_name:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_model_name, num_labels=self.num_labels)
        else:
            print('>>>>>Error: We now only support bert, roberta, macbert to do text classification.')
            return
        
        if len(self.load_model_path) > 0:
            model_state_dict = torch.load(self.load_model_path, map_location=self.device)
            self.model.load_state_dict(model_state_dict)
            print('>>>>>Load model weight from pretrained model path.')

        self.training_stats = []
        self.model.to(self.device)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # Setup Optimizer
        if self.opt == 'AdamW':
            self.optimizer = AdamW(optimizer_grouped_parameters,
                        lr = self.lr, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                        eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                        )
        elif self.opt == 'Adafactor':
            self.optimizer = Adafactor(scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        else:
            print('>>>>>Error: We now only support two types of oprimizer: AdamW, Adafactor')
            return
        
        if self.trainloader is not None:
            # Setup Scheduler
            total_steps = len(self.trainloader) * self.epochs
            # Create the learning rate scheduler.
            if self.sche == 'linear':
                self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                            num_warmup_steps = 0, # Default value in run_glue.py
                                                            num_training_steps = total_steps)
            elif self.sche == 'adafactor':
                self.scheduler = AdafactorSchedule(self.optimizer)
            else:
                print('>>>>>Error: We now only support linear scheduler and adafactor.')
                return
        
    def get_predictions(self, model, dataloader, compute_acc=False):
        predictions = None
        y_truth = None
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
                tokens_tensors, segments_tensors, masks_tensors = data[:3]
                outputs = model(input_ids=tokens_tensors, 
                                token_type_ids=segments_tensors, 
                                attention_mask=masks_tensors,
                                )   # choose output function here: sigmoid -> softmax
             
                logits = outputs[0]

                prob = F.softmax(logits.data, dim = 1)
                _, pred = torch.max(prob, 1)

                # _, pred = torch.max(logits.data, 1)
                # 用來計算訓練集的分類準確率
                if compute_acc:
                    labels = data[3]
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
                
        
        if compute_acc:
            acc = correct / total
            return predictions, acc, y_truth
        else:
            return predictions
        
    def get_predict(self, model, text):
        tokenized = self.tokenizer(txt_to_clean(text), padding="max_length", max_length=self.max_seq_len, truncation="longest_first")
        with torch.no_grad():
            tokens_tensor = torch.LongTensor([tokenized['input_ids']]).to(self.device)
            segments_tensor = torch.LongTensor([tokenized['token_type_ids']]).to(self.device)
            mask_tensor = torch.LongTensor([tokenized['attention_mask']]).to(self.device)

            outputs = model(input_ids=tokens_tensor, 
                            token_type_ids=segments_tensor, 
                            attention_mask=mask_tensor,
                            )   # choose output function here: sigmoid -> softmax
            
            logits = outputs[0]
            prob = F.softmax(logits.data, dim = 1)
            _, pred = torch.max(prob, 1)
            return pred
    
    def train(self):
        for epoch in range(self.epochs):
            print("")
            print('Training...')
            self.model.train()
            running_train_loss = 0.0
            running_train_correct = 0.0
            train_total = 0.0
            for data in self.trainloader:
                
                tokens_tensors, segments_tensors, \
                masks_tensors, labels = [t.to(self.device) for t in data]

                # 將參數梯度歸零
                self.optimizer.zero_grad()
                
                # forward pass
                outputs = self.model(input_ids=tokens_tensors, 
                                token_type_ids=segments_tensors, 
                                attention_mask=masks_tensors, 
                                labels=labels)
                
                loss = outputs[0]
                # backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                # 紀錄當前 batch loss
                running_train_loss += loss.item()

                # 記錄當前的 batch accuracy
                logits = outputs[1]
                prob = F.softmax(logits.data, dim = 1)
                _, pred = torch.max(prob, 1)
                train_total += labels.size(0)
                running_train_correct += (pred == labels).sum().item()
                
            # 計算分類準確率
            # _, train_acc, _ = self.get_predictions(self.model, self.trainloader, compute_acc=True)
            train_acc = running_train_correct / train_total
            avg_running_train_loss = running_train_loss / len(self.trainloader)
            print('Train>>>[epoch %d] loss: %.3f, acc: %.3f' %
                (epoch + 1, avg_running_train_loss, train_acc))
            
            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")
            self.model.eval()
            running_valid_loss = 0.0
            running_valid_correct = 0.0
            valid_total = 0.0
            for data in self.validloader:
                tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(self.device) for t in data]
                with torch.no_grad():   
                    outputs = self.model(input_ids=tokens_tensors, 
                                    token_type_ids=segments_tensors, 
                                    attention_mask=masks_tensors, 
                                    labels=labels)
                    loss = outputs[0]
                    running_valid_loss += loss.item()

                    # 記錄當前的 batch accuracy
                    logits = outputs[1]
                    prob = F.softmax(logits.data, dim = 1)
                    _, pred = torch.max(prob, 1)
                    valid_total += labels.size(0)
                    running_valid_correct += (pred == labels).sum().item()

            valid_acc = running_valid_correct / valid_total
            avg_running_valid_loss = running_valid_loss / len(self.validloader)
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
                }
            )
        
        # save model
        if self.save_model_path is not None:
            torch.save(self.model.state_dict(), self.save_model_path)
            print(f'>>>>> Finish training! Save model at {self.save_model_path} >>>>>')
            
        return self.model, self.training_stats
    
    def evaluate(self, path="output.log"):
        print('"')
        print('Start evaluate...')
        labels = []
        for i in range(self.num_labels):
            labels.append(i)
        predictions, _, y_test = self.get_predictions(self.model, self.testloader, compute_acc=True)
        acc, pre, rc, f1, cm = compute_performance(y_test.cpu(), predictions.cpu(), labels=labels)
        log_performance(acc, pre, rc, f1, cm, labels=labels, path=path)
        return acc, pre, rc, f1, cm
    


        
    




# %%
