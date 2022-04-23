# Functions
import os
import re
import gc
import sys
import pickle
import pandas as pd
import numpy as np
import progressbar
import matplotlib
import jieba
import jieba.analyse
from collections import Counter
import json
from data.GV import *
from IPython.display import display
from importlib import import_module, reload
import torch
import torch.nn.functional as F
import random
import datetime
from contextlib import redirect_stdout
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score


########################################################################################################################
def setup_device(device=""):
    if len(device) > 0:
        device = torch.device(device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            device = torch.device("cpu")
    return device

def seed_torch(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

class SentenceDataset(Dataset):
    # 讀取前處理後的 csv 檔並初始化一些參數
    def __init__(self, mode):
        assert mode in ["train", "test", "valid", "pred"]  # 一般訓練你會需要 dev set
        self.mode = mode
        # 大數據你會需要用 iterator=True
        # self.df = pd.read_csv("data/cleaned/" + mode + ".csv").fillna("")
        self.df = pd.read_pickle("data/cleaned/" + mode + ".pkl").fillna("")
        self.len = len(self.df)
    
    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        # if any([self.mode == "test", self.mode == "pred"]):
        if any([self.mode == "pred"]):
            ids = self.df.iloc[idx, 1]  # get X
            label_tensor = None
            length = len(self.df.iloc[idx, 1])
        else:
            label, ids = self.df.iloc[idx, :].values    # get y & X
            # 將 label 文字也轉換成索引方便轉換成 tensor
            label_id = label
            label_tensor = torch.tensor(label_id)
            length = len(self.df.iloc[idx, 1])

        len_a = len(ids)
        tokens_tensor = torch.tensor(ids)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor = torch.tensor([0] * len_a)
        length_tensor = torch.tensor(length)
        return (tokens_tensor, segments_tensor, length_tensor, label_tensor)
    
    def __len__(self):
        return self.len

def my_pad_sequence(sequences, batch_first=True, padding_value=0.0, max_len=128):
    out_dims = (len(sequences), max_len)
    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # length = max_len
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor
    return out_tensor

def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    lengths_tensors = torch.stack([s[2] for s in samples])
    # 測試集有 labels
    if samples[0][3] is not None:
        label_ids = torch.stack([s[3] for s in samples])
    else:
        label_ids = None

    
    
    # zero pad 到同一序列長度
    tokens_tensors = my_pad_sequence(tokens_tensors, 
                                  batch_first=True)
    # print('token', tokens_tensors.shape)
    segments_tensors = my_pad_sequence(segments_tensors, 
                                    batch_first=True)
    
    # attention masks，將 tokens_tensors 裡頭不為 zero padding
    # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
    masks_tensors = torch.zeros(tokens_tensors.shape, 
                                dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(
        tokens_tensors != 0, 1)
    return tokens_tensors, segments_tensors, masks_tensors, lengths_tensors, label_ids

def get_average_performance(performance):
    acc_ = performance['Acc']
    pre_ = performance['Precision']
    rc_ = performance['Recall']
    f1_ = performance['F1-score'] 
    confusion_matrix_  = performance['Confusion-Matrix']
    acc = np.array(acc_)
    pre = np.array(pre_)
    rc = np.array(rc_)
    f1 = np.array(f1_)
    confusion_matrix = np.array(confusion_matrix_)
    acc_av = np.mean(acc)
    pre_av = np.mean(pre, axis = 0)
    rc_av = np.mean(rc, axis = 0)
    f1_av = np.mean(f1, axis = 0)
    cm_av = np.mean(confusion_matrix, axis = 0)
    print(f'Average Acc:{round(acc_av, 3)}')
    print(f'Average Precision:{np.round(pre_av, 3)}')
    print(f'Average Recall:{np.round(rc_av, 3)}')
    print(f'Average F1-score:{np.round(f1_av, 3)}')
    print('Average Confusion Matrix:')
    print(np.round(cm_av, 3))

    return acc_av, pre_av, rc_av, rc_av, f1_av, cm_av

def log_info(info, path=""):
    if len(path) > 0:
        with open(path, 'a') as f:
            with redirect_stdout(f):
                print(info)
    return 
def log_performance(acc, pre, rc, f1, cm, labels=None, path=""):
    if len(path) > 0:
        with open(path, 'a') as f:
            with redirect_stdout(f):
                # ACC
                print("Accuracy: ", '%.4f'%acc)
                # Precision
                result = []
                print("Precision......")
                for label in labels:
                    print(f"Class {label}: {round(pre[label], 3)} ")
                    result += [round(pre[label], 3)]
                print(result)
                result = []
                # Recall
                print("Recall......")
                for label in labels:
                    print(f"Class {label}: {round(rc[label], 3)} ")
                    result += [round(rc[label], 3)]
                print(result)
                # f1
                result = []
                print("F1 score......")
                for label in labels:
                    print(f"Class {label}: {round(f1[label], 3)} ")
                    result += [round(f1[label], 3)]
                print(result)
                # confusion matrix
                print("Confusion matrix......")
                print(cm)

    return
# TODO: Murphy 把計算實驗結果和 logging 拆開來
def compute_performance(y_true, y_pred, labels=None):
    # accuracy
    acc = accuracy_score(y_true, y_pred)
    print("Accuracy: ", '%.4f'%acc)
    # precision
    pre = precision_score(y_true, y_pred, average=None, labels=labels)
    result = []
    print("Precision......")
    for label in labels:
        print(f"Class {label}: {round(pre[label], 3)} ")
        result += [round(pre[label], 3)]
    print(result)
    # recall
    rc = recall_score(y_true, y_pred, average=None, labels=labels)
    result = []
    print("Recall......")
    for label in labels:
        print(f"Class {label}: {round(rc[label], 3)} ")
        result += [round(rc[label], 3)]
    print(result)
    # F1-Score
    print("F1-Score......")
    f1 = f1_score(y_true, y_pred, average=None, labels=labels)
    result = []
    for label in labels:
        print(f"Class {label}: {round(f1[label], 3)} ")
        result += [round(f1[label], 3)]
    print(result)
    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion matrix......")
    print(cm)
    return acc, pre, rc, f1, cm


def output_to_list(content, content_list):
    '''
    Purpose...
    :param content: type. Description.
    :param content_lst: list. 
    :return:
    '''
    #print(type(pd.Series()))
    if type(content) is type(pd.Series()):
        #print(type(pd.Series()))
        content.apply(output_to_list, content_list=content_list)
    elif type(content) is float:
        if not np.isnan(content):
            content_list.append(content)
    elif content is not np.nan:
        content_list.append(content)
    return


########################################################################################################################
def txt_to_clean(input_txt, clean_path=None, textmode=False):
    '''
    Replace money and date with ??
    '''
    if textmode is False:
        txt_path = input_txt
        with open(txt_path) as f:
            data = f.read()
    else:
        data = input_txt

    df=pd.DataFrame([data],columns=['content'])
    #display(df)

    # remove newline, space and Full space
    default_pattern = r'(\r\n|\n|\ |\u3000)'
    patterns_list = [default_pattern]
    pattern=re.compile("|".join(patterns_list))

    df['content'] = df['content'].astype(str).str.replace(pat=pattern,repl='')

    # replace non-printable words with "○"
    default_pattern = r"""(\uf67e|\uf582|\uf57f|\uf67d|\uf5e8|
                            \uf5e7|\uf581|\uf584|\uf5e6|\uf583|
                            \uf57e|\uf580|\uf5e5|\uf5e4|\uf6bb|
                            \uf6bc|\uf6b2|\uf6b3|\uf6b4|\uf67c|
                            \uf67b|\uf67a|\uf679|\uf678|\uf677|
                            \uf676|\uf675|\uf674|\uf673|\uf672|
                            \uf671|\uf670|\uf66f|\uf66e|\uf66d|
                            \uf66c|\uf66b|\uf66a|\uf669|\uf668|
                            \uf667|\ue372|\uf6bd|\uf5e3|\uf5e2|
                            \uf5e1|\ue3cc|\uf6b1|\uf6be|\uf6b0|
                            \uf6af|\uf6ae|\uf6ad|\uf6ac|\uf6ab|
                            \uf6aa|\uee64|\ue38d|\ue450)"""

    patterns_list = [default_pattern]
    pattern=re.compile("|".join(patterns_list),re.VERBOSE)

    df['content'] = df['content'].str.replace(pat=pattern,repl='○')

    # replace box drawing symbols with " "
    pattern_box_drawing_symbols = r"""([┤├│─┼┬┴┐└┘┌])"""

    pattern=re.compile(pattern_box_drawing_symbols,re.VERBOSE)

    df['content'] = df['content'].str.replace(pat=pattern,repl=' ')


    # https://regex101.com/r/XuJpyr/19
    # 法律條文

    pattern_law_articles= r"""((家事事件法|民事訴訟法|民法(親屬編)?|
                                臺灣地區與大陸地區人民關係條例|刑法|
                                家庭暴力防治法|非訟事件法|涉外民事法律適用法|
                                兒童及少年福利與權益保障法|公司法|
                                毒品危害防制條例|家事非訟事件暫時處分類型及方法辦法|
                                家事事件審理細則|其中|準用|或[依有]|[該同前](法?條|法))
                                ((於[\d零一二三四五六七八九十百千、至-]*年修正後，於)?
                                (第?[\d零一二三四五六七八九十百千、至-]*
                                (之[\d零一二三四五六七八九十百千、至-]*)?[編章節條項第])
                                (規定[\d]+)?(之[\d零一二三四五六七八九十百千、至-]+)?
                                (增列)?((第?[\d零一二三四五六七八九十百千、至-]*款)?
                                ((?=[\u4e00-\u9fa5]*罪)(([^及]*罪)))?
                                ([及至或]?[前段]?[、]?(準用)?(但書)?))+)+)+|
                                (家事事件法|民事訴訟法|民法(親屬編)?|
                                臺灣地區與大陸地區人民關係條例|刑法|
                                家庭暴力防治法|非訟事件法|涉外民事法律適用法|
                                兒童及少年福利與權益保障法|公司法|
                                毒品危害防制條例|家事非訟事件暫時處分類型及方法辦法|家事事件審理細則)"""

    pattern=re.compile(pattern_law_articles,re.VERBOSE)
    df['content'] = df['content'].str.replace(pat=pattern,repl='_LawArticles_')


    # https://regex101.com/r/VIfBwt/13
    # 金錢數額
    pattern_amount_of_money= r"""(?!多元)
                                ((([\d,\.零一二三四五六七八九十百千萬壹貳參叄肆伍陸柒捌玖拾佰仟多幾數~、元]+至?)?
                                [\d,\.零一二三四五六七八九十百千萬壹貳參叄肆伍陸柒捌玖拾佰仟多幾~、]+)
                                (?=餘|萬|元)([萬元整]|餘萬?元)+([\d,]+)?)+
                                (([【（(]?計算方?式[：:（]+([\D]*)|[（(]+((?![^計])|(?:[\d])))
                                ([\d,月]*[\.\-－+＋÷*Xx×=＝\/又千萬元（年）個月][\d,人戶元）)】]*)*
                                ([，【]*[\D]+四捨五入[】]*）)?([\D]*[】])?)?"""

    pattern=re.compile(pattern_amount_of_money,re.VERBOSE)
    df['content'] = df['content'].str.replace(pat=pattern,repl='_AmountOfMoney_')


    # https://regex101.com/r/Ly4mvb/9
    # 具體日期(年/月/日 or 月/日 or 農曆大年初一初二...)
    pattern_specific_date= r"""((中華)?(民國)?(
                                    ([\d零一二三四五六七八九十百]+[○]*[\d零一二三四五六七八九十百]*|[○]+|[同去今])
                                    年(農曆)?[\d○零一二三四五六七八九十百]*月
                                    (([\d○零一二三四五六七八九十百、至]+日)+|[\d○零一二三四五六七八九十百]+|農曆春節)
                                )|(農曆)?([\d○零一二三四五六七八九十百]+|[同])月
                                     ([\d○零一二三四五六七八九十百、至]+日)+|(
                                         ([\d零一二三四五六七八九十百]+[○]*[\d零一二三四五六七八九十百]*|[○]+|[同去今])年
                                     )?(農曆)?(春節之?)?(
                                         小?除夕(日|夜|年節?)?|農曆年|過年|(?<![奇偶]數)年?初(?=[\d一二三四五六七八九十])
                                        )(年節)?([\d初零一二三四五六七八九十]+([、](?=初))?)*|
                                 (農曆)?(正月|大年)[\d初零一二三四五六七八九十]+日?
                                )"""


    pattern=re.compile(pattern_specific_date,re.VERBOSE)
    df['content'] = df['content'].str.replace(pat=pattern,repl='_SpecificDate_')


    # https://regex101.com/r/tPQASR/9
    # 具體月份(年/月 or 月份)
    pattern_specific_month= r"""(([\d]+|[同今])年([\d、至及]+月)+(起|間|初|份|底)?)|([\d]+月份)"""

    pattern=re.compile(pattern_specific_month,re.VERBOSE)
    df['content'] = df['content'].str.replace(pat=pattern,repl='_SpecificMonth_')


    # https://regex101.com/r/e6258Q/6
    # 身份證字號
    pattern_security_id= r"""((國民)?身[分份](（[\S]+）)?證字?(統一)?編?號碼?：?[a-zA-Z\d]+(號|（號，?）))"""

    pattern=re.compile(pattern_security_id,re.VERBOSE)
    df['content'] = df['content'].str.replace(pat=pattern,repl='_SecurityID_')

    # https://regex101.com/r/Y8wMJ8/7
    # 地址
    pattern_address= r"""(([\D]{2}(市|縣|鄉|村|(?<!地)區|里))+([\D]{1,10}(路|街))*
                            ([\d○]+(段|巷|弄))*[\d○]+地?號([\d]+樓)?(之[\d]+號?)?(?!函))"""

    pattern=re.compile(pattern_address,re.VERBOSE)
    df['content'] = df['content'].str.replace(pat=pattern,repl='_ADDRESS_')


    # https://regex101.com/r/dAzXw4/7
    # 年紀,年次
    #print("AgesYear")
    pattern_ages_year= r"""((((?=(?P<tmp>([\d零一二三四五六七八九十百]+[及至、]?)+))(?P=tmp)|[○]+)多?歲([\d]+個月)?)|
                            ((?<=年僅)[\d]個多月)|([\d零一二三四五六七八九十百]+年次))"""

    pattern=re.compile(pattern_ages_year,re.VERBOSE)
    df['content'] = df['content'].str.replace(pat=pattern,repl='_AgesYear_')



    # https://regex101.com/r/8dpxhG/9
    # XX年度XX字第XX號 案件、檔案編號
    #print("SpecificCase")
    pattern_specific_case= r"""((臺灣[\w]+([\d○零一二三四五六七八九十百]+年度[\w]+字第[\d○零一二三四五六七八九十百、]+號))|
                                    ((（[\d]+[\w]?）)?(移署資處)?(?=[^以])([\w（）]+)字第[\d]+號函)|
                                    ((?=[^以])([\w○]+)[\d\-]+號函)|((（[\d]+[\w]?）|華總|新北)([\w（）]+)字第[\d]+號)|
                                    ((最高法院)([\d]+年[\w]+字(第[\d]+號[、]?)+[、]?)+)|
                                    (最高法院)?[\d]+年度第[\d]+次[\w]+會議|
                                    ((司法院大法官會議|大法官會議|大法官)[\w]+字?第[\d]+號解釋)|
                                    (臺南地檢|最高法院)[\d]+年[\w]+字[\d]+號|
                                    (臺灣[\w]+地方法院)?
                                    ([\d○零一二三四五六七八九十百]+年[\w]+[\d○零一二三四五六七八九十百]+號))"""

    pattern=re.compile(pattern_specific_case,re.VERBOSE)
    df['content'] = df['content'].str.replace(pat=pattern,repl='_SpecificCase_')



    content=df['content'][0]
    #display(content)

    if textmode is False:
        # save clean file to clean_path
        save_dir = os.path.expanduser(clean_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        path=txt_path
        match = re.split(r'\/',path)
        match = re.split(r'\.',match[-1])
        filename=match[0]

        save_path=save_dir+filename+".clean"
        with open(save_path, "w") as f:
            f.write(content)

        return save_path
    else:
        return content


########################################################################################################################
from transformers import BertTokenizer
PRETRAINED_MODEL_NAME = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
def clean_to_seg_by_tokenizer(input_txt, seg_path=None, textmode=False):
    word_pieces = ["[CLS]"]
    tokens_text = tokenizer.tokenize(input_txt)
    word_pieces += tokens_text + ["[SEP]"]
    ids = np.array(tokenizer.convert_tokens_to_ids(word_pieces))
    return ids


def clean_to_seg_by_jieba(input_txt, seg_path=None, textmode=False):
    '''
    Using "jieba" to segament chinese sentences.
    '''
    if textmode is False:
        clean_path = input_txt
        with open(clean_path) as f:
            data = f.read()
    else:
        data = input_txt

    cutter=Cut2Seg(jieba_cut)

    #content=jieba_cut(data)
    #display(content)

    content=cutter.cut(data)
    content=content.replace(' ','\n')

    # save segmeted file to seg_path
    if textmode is False:
        save_dir = os.path.expanduser(seg_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        path=clean_path
        match = re.split(r'\/',path)
        match = re.split(r'\.',match[-1])
        filename=match[0]

        save_path=save_dir+filename+".seg"
        with open(save_path, "w") as f:
            f.write(content)

        return save_path
    else:
        return content.split()


########################################################################################################################
def init_jieba(stop_words_path, dict_path, idf_path, userdict_path):
    #多核心平行運算，只支援linux
    if os.name == 'posix':
        jieba.enable_parallel()
    jieba.analyse.set_stop_words(stop_words_path)
    jieba.set_dictionary(dict_path)
    jieba.analyse.set_idf_path(idf_path)

    print(stop_words_path)
    print(dict_path)
    print(idf_path)

    for path in userdict_path:
        print(path)
        jieba.load_userdict(path)

    return


########################################################################################################################
def jieba_cut(content):
    if type(content)==float:
        return float('nan')
    #seg_list = jieba.lcut(content, HMM=True, cut_all=False)
    #return " ".join(seg_list)
    return jieba.lcut(content, HMM=True, cut_all=False)


########################################################################################################################
class Cut2Seg(object):
    def __init__(self,cutter):
        self.cutter=cutter

    def cut(self,sentence):
        return ' '.join(list(self.cutter(sentence)))

    def cut_Array(self, sentence_list):
        return np.vectorize(self.cut)(sentence_list)


########################################################################################################################
def json_to_txt(json_path,txt_path):
    with open(json_path) as f:
        data = json.load(f)

    content=data['JFULL']
    #display(content)

    save_dir = os.path.expanduser(txt_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path=json_path
    match = re.split(r'\/',path)
    match = re.split(r'\.',match[-1])
    filename=match[0]

    save_path=save_dir+filename+".txt"
    with open(save_path, "w") as f:
        f.write(content)

    return save_path