#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /media/allenyl/DATA/Projects/server_setup/repo-ai/NLP-AI-in-Law/samples/Doc2Vec_helper.py
# Project: /media/allenyl/DATA/Projects/server_setup/repo-ai/NLP-AI-in-Law/samples
# Created Date: Tuesday, March 12th 2019, 11:51:04 am
# Author: Allenyl(allen7575@gmail.com>)
# -----
# Last Modified: Thursday, May 23rd 2019, 12:19:37 pm
# Modified By: Allenyl
# -----
# Copyright 2018 - 2019 Allenyl Copyright, Allenyl Company
# -----
# license: 
# All shall be well and all shall be well and all manner of things shall be well.
# We're doomed!
# ------------------------------------
# HISTORY:
# Date      	By	Comments
# ----------	---	---------------------------------------------------------
###

from sklearn import utils
from sklearn.utils import shuffle
import copy
import collections
import random
import numpy as np
import pandas as pd


def model_train(doc2vec_model, tagged_X_train, alpha=0.065, min_alpha=0.005, max_epochs=300, update=False):
    doc2vec_model.alpha = alpha
    delta_alpha = (alpha - min_alpha)/max_epochs
    
    # update vocabulary
    if update:
        doc2vec_model.build_vocab(tagged_X_train,update=True)
    
    for epoch in range(max_epochs):
        if epoch % 10 == 0 :
            print('iteration {}, alpha {}'.format(epoch, doc2vec_model.alpha))
            # need to use deepcopy to avoid model can't be trained problem
            model = copy.deepcopy(doc2vec_model)
            print(sanity_check(model, tagged_X_train, verbose=False))
            
        doc2vec_model.train(utils.shuffle(tagged_X_train),
                    total_examples=len(tagged_X_train),
                    epochs=1)
        # decrease the learning rate
        doc2vec_model.alpha -= delta_alpha
        # fix the learning rate, no decay
        doc2vec_model.min_alpha = doc2vec_model.alpha


def sanity_check(doc2vec_model, tagged_Doc, check_num=500, verbose=True):

    most_similiar_ranks = []
    #second_ranks = []
    most_similiar_doc_id = []
    
    if check_num > len(tagged_Doc):
        check_num = len(tagged_Doc)


    for i in range(check_num):
        #print(tagged_Doc[0].words)
        # 對 doc_id 的文字做 inference, 得到 inferred_vector
        inferred_vector = doc2vec_model.infer_vector(tagged_Doc[0].words)
        #print(tagged_Doc[0].words)
        #print(tagged_Doc[0].tags[0])
        # 對 inferred_vector 跟 所有的tag 做相似度比較，並排序結果
        sims = doc2vec_model.docvecs.most_similar([inferred_vector], topn=len(doc2vec_model.docvecs)) # 文章總數 + label數 = 總tag數
        #print(sims)
        
        #print(doc_id)
        # In the list of total docid which sorted by similarity
        # the more smiliar, the less the list index
        # so index=0 must be the most similar, then index=1 less similar, and so on...
        # use index as similarity rank, search for current docid in the list, and return its index as rank
        # the most similar docid should be its self, so rank=0 should be most case.
        # 對 dbow model 來說，rank 越低越好，代表model 可以提取出足以判斷是否同一篇文章的特徵
        #rank = [docid for docid, sim in sims].index('SENT'+str(doc_id)) # 取出 'SENT'+str(doc_id) 在sims 中的排名
        ranked_docid_list = [docid for docid, sim in sims]
        #print(tag_list[:500])
        #print(tagged_Doc[0].tags[0])
        rank = ranked_docid_list.index(tagged_Doc[0].tags[0]) # 取出 'SENT'+str(doc_id) 在sims 中的排名
        #print(rank)
        
        most_similiar_ranks.append(rank)
#         if rank != 0:
#             print('doc_id:',doc_id, 'rank0:',sims[0], 'rank',rank,':', sims[rank])
        
        # For another check, there may have few docid which similar to many docids
        # this is another feature: find similar group
        # 對 dmm 來說，相似的 doc_id 數量越少越好，代表model 可以提取出該篇文章是否可以歸類到少數幾個主題的特徵
        most_similiar_doc_id.append(sims[0][0]) # 取出最相似的tag
        #second_ranks.append(sims[:rank])
        tagged_Doc = shuffle(tagged_Doc)
        #print(tagged_Doc[:10])
        
    
    rank_count = collections.Counter(most_similiar_ranks)
    id_count = collections.Counter(most_similiar_doc_id)
    
    if verbose:
        print('\n')
        print('number of most similiar doc with rank: ')  
        print(rank_count)  # Results vary between runs due to random seeding and very small corpus
        print('length:', len(rank_count))
        print('\n')
        #return second_ranks
        print('number of most similiar doc with tag: ')
        print(id_count)
        print('length:', len(id_count))
    
    return len(rank_count), len(id_count)

def show_most_least_similar(d2v_model, tagged_X_train, X_test_text):
    from sklearn.utils import shuffle

    inferred_vector = d2v_model.infer_vector(X_test_text)
    sims = d2v_model.docvecs.most_similar([inferred_vector], topn=len(d2v_model.docvecs))

    # Compare and print the most/median/least similar documents from the train corpus
    print('Test Document: «{}»\n'.format(' '.join(X_test_text)))

    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % d2v_model)
    
    for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
        # 從 training data 中取出 tag
        df_tmp = pd.DataFrame(pd.DataFrame(tagged_X_train).tags.tolist())
        
        # 從 most similar 中取出 tag
        most_similar_tag = sims[index][0]
        print('most similar label:', most_similar_tag)
        
        # 找出"所有" 對應 most similar tag 的 index
        df_tmp2 = (df_tmp==most_similar_tag) # 找出符合條件的欄位        
        list_of_df = [] # 新增空的暫存list
        ## 將每個 column 中符合條件的 dataframe 暫存到 list
        for i_column in df_tmp2.columns:
            list_of_df.append(df_tmp[df_tmp2[i_column]])
        
        df_tmp = pd.concat(list_of_df) # 將暫存的 dataframe 重新 concat 在一起
        #rint(df_tmp)
        most_similar_index_list = df_tmp.index # 取出所有 index

        # 可能有多個 index, 每次都亂數取一個
        doc_index = shuffle(most_similar_index_list)[0]
        #print(doc_index)
        
        #doc_index = pd.DataFrame(pd.DataFrame(tagged_X_train).tags.tolist())[0].tolist().index(sims[index][0])
        print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(tagged_X_train[doc_index].words)))


def sample_most_least_similar(model, tagged_X_train, X_test, y_test, n_sample=5):
    count_dict = {}
    n_category = len(np.unique(y_test))
    index = 0

    for i in range(n_category):
        count_dict[i] = 0

    while index < n_sample:
        i = np.random.randint(0, len(y_test))
        #print(i)
        # evenly sample each category
        if count_dict[y_test[i]] < n_sample/n_category:
            print('sample:', index)
            print('id:', i)
            count_dict[y_test[i]] += 1
            index += 1
            print('label: ', y_test[i])
            show_most_least_similar(model, tagged_X_train, X_test[i])
            print('--------------------')
            #print(i)
