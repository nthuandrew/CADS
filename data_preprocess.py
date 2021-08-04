#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /media/allenyl/DATA/Projects/server_setup/repo-ai/NLP-AI-in-Law/samples/data_preprocess.py
# Project: /media/allenyl/DATA/Projects/server_setup/repo-ai/NLP-AI-in-Law/samples
# Created Date: Tuesday, December 11th 2018, 3:10:36 pm
# Author: Allenyl(allen7575@gmail.com>)
# -----
# Last Modified: Tuesday, October 15th 2019, 10:02:18 pm
# Modified By: Allenyl
# -----
# Copyright 2018 - 2018 Allenyl Copyright, Allenyl Company
# -----
# license:
# All shall be well and all shall be well and all manner of things shall be well.
# We're doomed!
# ------------------------------------
# HISTORY:
# Date      	By	Comments
# ----------	---	---------------------------------------------------------
###


## 讀入法律辭典
import pandas as pd

from pathlib import Path
source_dir = Path(__file__).parent

# read keyword file
keyword_list=pd.read_excel(source_dir/'extra_dict/KWIC中英關鍵字-全部法律關鍵字2018-08-22 14-14.xlsx', engine='openpyxl')

import re
# remove some pattern
unused_punctuations = r'([（）「」。，：！？【】#@\/$《》／；\-『』～?+…⋯〈〉:,─x│])'
unused_punctuations2 = r'([、\(\)])'
default_pattern = r'(<script[\s\S]+?/script>|<.*?>|\r|&nbsp;|\u3000|\xa0|\n|[.][.]+)'
patterns_list = [unused_punctuations, default_pattern]
pattern=re.compile("|".join(patterns_list))

keyword_list['KWIC中英關鍵字_mod'] = keyword_list['KWIC中英關鍵字'].str.replace(pat=pattern,repl='')


import csv
base_path = source_dir/'extra_dict/'

keyword_list['KWIC中英關鍵字'][4:].to_csv(base_path/'keyword_list_ori.csv',encoding='utf8', index=False)
keyword_list['KWIC中英關鍵字_mod'][4:].to_csv(base_path/'keyword_list_mod.csv',encoding='utf8', index=False)


# ### [fxsjy/jieba: 结巴中文分词](https://github.com/fxsjy/jieba)
# - [关于idf.txt · Issue #87 · fxsjy/jieba](https://github.com/fxsjy/jieba/issues/87)

#from IPython.lib.deepreload import reload as dreload
#dreload(jieba)

#%load_ext autoreload
#%autoreload 2
import os
import jieba
import jieba.analyse
import numpy as np

# from importlib import reload
# reload(jieba.analyse)
#del jieba.analyse

base_path = source_dir/'extra_dict/'

methods_name = {'dict1':0, 'dict2':1}
stop_words_filenames = ['stop_words.txt','stop_words.txt']
dict_filenames = ['dict.txt.big', 'dict.txt.big']
idf_filenames = ['idf.txt.big', 'idf.txt.big']
userdict_filename = [['keyword_list_mod.csv', 'custom_dict.txt'],['keyword_list_mod.csv', 'custom_dict.txt', 'classical_chinese_dict_zhtw.txt']]

stop_words_path_list = [str(base_path/s) for s in stop_words_filenames]
dict_path_list = [str(base_path/s) for s in dict_filenames]
idf_path_list = [str(base_path/s) for s in idf_filenames]
userdict_list = [[str(base_path/s) for s in userdict_filename[i]] for i in range(len(userdict_filename))]

#// TODO csu moving functions to class_Segmentatin & util.py
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


def jieba_cut(content):
    if type(content)==float:
        return float('nan')
    #seg_list = jieba.lcut(content, HMM=True, cut_all=False)
    #return " ".join(seg_list)
    return jieba.lcut(content, HMM=True, cut_all=False)


init_jieba(*list(zip(stop_words_path_list, dict_path_list, idf_path_list, userdict_list))[methods_name['dict1']])


### 包裝成統一界面
class Cut2Seg(object):
    def __init__(self,cutter):
        self.cutter=cutter

    def cut(self,sentence):
        return ' '.join(list(self.cutter(sentence)))

    def cut_Array(self, sentence_list):
        return np.vectorize(self.cut)(sentence_list)


### 文字清理
import json,re
import pandas as pd

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



def txt_to_clean(input_txt, clean_path=None, textmode=False):
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


def clean_to_seg(input_txt, seg_path=None, textmode=False):
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

