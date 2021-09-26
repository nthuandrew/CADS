from module.util import *
from class_BertForClassification_Model import BertForClassification
from data.GV import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from pandas import Series,DataFrame
from transformers import AdamW, get_linear_schedule_with_warmup
import time

class Bert_Wrapper():
    def __init__(self, num_labels=2):
        self.seed = 1234
        self.device = setup_device()
        seed_torch(seed=self.seed)
        self.MAX_LENGTH = 128
        self.NUM_LABELS = num_labels
        self.BATCH_SIZE = 64
        self.EPOCHS = 3
        self.PRETRAINED_MODEL_NAME = "bert-base-chinese"
        return

    def prepare_custody_sentiment_analysis_dataloader(self, df, df_neu):
        applicant_advantage_column = df.columns[df.columns.to_series().str.contains('AA')].tolist()
        respondent_advantage_column = df.columns[df.columns.to_series().str.contains('RA')].tolist()
        applicant_disadvantage_column = df.columns[df.columns.to_series().str.contains('AD')].tolist()
        respondent_disadvantage_column = df.columns[df.columns.to_series().str.contains('RD')].tolist()
        neutral_column = df_neu.columns[df_neu.columns.to_series().str.contains('neutral')].tolist()

        advantage_column = applicant_advantage_column + respondent_advantage_column
        disadvantage_column = applicant_disadvantage_column + respondent_disadvantage_column

        # training sentence set
        advantage_list=[]
        disadvantage_list=[]
        neutral_list=[]

        df.loc[:,advantage_column].apply(output_to_list, content_list=advantage_list)
        df.loc[:,disadvantage_column].apply(output_to_list, content_list=disadvantage_list)
        df_neu.loc[:,neutral_column].apply(output_to_list, content_list=neutral_list)

        advantage_list_shuffled = shuffle(advantage_list, random_state=self.seed)
        disadvantage_list_shuffled = shuffle(disadvantage_list, random_state=self.seed)
        # reduced training neutral sentence
        n_neutral_samples = int((len(advantage_list_shuffled)+len(disadvantage_list_shuffled))/2)
        neutral_list_shuffled = shuffle(neutral_list, random_state=self.seed)[:n_neutral_samples]

        print("advantage training number:", len(advantage_list_shuffled))
        print("disadvantage training number:", len(disadvantage_list_shuffled))
        print("neutral training number:", len(neutral_list_shuffled))

        df.loc[:,advantage_column].apply(output_to_list, content_list=advantage_list)
        df.loc[:,disadvantage_column].apply(output_to_list, content_list=disadvantage_list)
        df_neu.loc[:,neutral_column].apply(output_to_list, content_list=neutral_list)

        if self.NUM_LABELS == 2:
            # 不利標為0
            y0 = [0]*len(disadvantage_list)
            # 有利標為1
            y1 = [1]*len(advantage_list)
            # 二分類
            y = y0+y1
            X = disadvantage_list + advantage_list
        elif self.NUM_LABELS == 3:
             # 不利標為0
            y0 = [0]*len(disadvantage_list)
            # 有利標為1
            y1 = [1]*len(advantage_list)
            # 中性標為2
            y2 = [2]*len(neutral_list)
            # 三分類
            y = y0+y1+y2
            X = disadvantage_list + advantage_list + neutral_list

        else:
            print('Number of labels seems wrong!')
            return

        df_clean = pd.DataFrame({'y':y,'X':X})
        df_clean = df_clean[~(df_clean.X.apply(lambda x : len(x)) > self.MAX_LENGTH-2)]
        X, y = df_clean['X'], df_clean['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.seed)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=self.seed)
        X_train.reset_index(drop=True, inplace=True)
        X_valid.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        y_valid.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)

        train_data = {
            'label': y_train,
            'text': X_train
        }
        train_df = DataFrame(train_data)
        train_df.to_csv('./data/cleaned/train.csv', index=False)
        train_df.to_pickle('./data/cleaned/train.pkl')
        valid_data = {
            'label': y_valid,
            'text': X_valid
        }
        valid_df = DataFrame(valid_data)
        valid_df.to_csv('./data/cleaned/valid.csv', index=False)
        valid_df.to_pickle('./data/cleaned/valid.pkl')
        test_data = {
            'label': y_test,
            'text': X_test
        }
        test_df = DataFrame(test_data)
        test_df.to_csv('./data/cleaned/test.csv', index=False)
        test_df.to_pickle('./data/cleaned/test.pkl')

        trainset = SentenceDataset("train")
        validset = SentenceDataset("valid")
        testset = SentenceDataset("test")

        self.trainloader = DataLoader(trainset, batch_size=self.BATCH_SIZE, 
                         collate_fn=create_mini_batch)
        self.validloader = DataLoader(validset, batch_size=self.BATCH_SIZE, 
                                collate_fn=create_mini_batch)
        self.testloader = DataLoader(testset, batch_size=256, 
                                collate_fn=create_mini_batch)

        return self.trainloader, self.validloader, self.testloader

    def get_predictions(self, model, dataloader, compute_acc=False, output_attention=False):
        predictions = None
        attentions = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            # 遍巡整個資料集
            for data in dataloader:
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
                                )
                if output_attention:
                    logits = outputs[0][0]
                else:
                    logits = outputs[0]
                _, pred = torch.max(logits.data, 1)
                # 用來計算訓練集的分類準確率
                if compute_acc:
                    labels = data[4]
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()
                    
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
            return predictions, acc
        if output_attention:
            return predictions, attentions
        else:
            return predictions

    def initialize_training(self):
        self.model = BertForClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=self.NUM_LABELS, max_length=self.MAX_LENGTH, device=self.device)
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
                        lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                        eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                        )
        ################# Method 1 ######################
        # gradient_accumulation_steps = 1
        # # total_steps = len(trainloader) * EPOCHS
        # t_total = int(len(trainloader) / gradient_accumulation_steps * EPOCHS)

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
        for epoch in range(self.EPOCHS):
            print("")
            print('Training...')
            t0 = time.time()
            self.model.train()
            running_train_loss = 0.0
            for data in trainloader:
                
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
            _, train_acc = self.get_predictions(self.model, trainloader, compute_acc=True)
            avg_running_train_loss = running_train_loss / len(trainloader)
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
            for data in validloader:
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

            _, valid_acc = self.get_predictions(self.model, validloader, compute_acc=True)
            avg_running_valid_loss = running_valid_loss / len(validloader)
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
        return

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
        # %%
        # Plot learning curve
        from matplotlib import pyplot as plt
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

    def evaluate(self):
        # predictions, attention = get_predictions(model, testloader, output_attention=True)
        predictions = self.get_predictions(self.model, self.testloader, output_attention=False)
        # y_test = pd.read_csv("data/cleaned/" + 'test' + ".csv", dtype=int).fillna("")['label']
        y_test = pd.read_pickle("data/cleaned/" + 'test' + ".pkl").fillna("")['label']
        compute_performance(y_test, predictions.cpu(), labels=[0,1,2])
        return






if __name__=='__main__':
    print('Bert Classification Wrapper...')
    df = pd.read_pickle('./data/cleaned/judgment_result_seg_bert.pkl')
    df_neu = pd.read_pickle('./data/cleaned/judgment_result_seg_neu_bert.pkl')
    bw = Bert_Wrapper()
    trainloader, validloader, testloader = bw.prepare_custody_sentiment_analysis_dataloader(df, df_neu)
    bw.initialize_training()
    bw.train()
    bw.evaluate()


