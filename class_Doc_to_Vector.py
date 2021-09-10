from module.util import *
from data.GV import *
from gensim.models.doc2vec import Doc2Vec
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
import copy
from itertools import count
from collections import defaultdict 

class Doc_to_Vector():
    def __init__(self):
        self.categorical = ['Result','Willingness','AK','RK','AN','RN', 'Type']
        self.meta_info = ['filename', 'ID', 'Others']
        
        return

    # //TODO 弘祥：把 train_doc2vec.py 放進來
    def train_model(self, advantage_list, disadvantage_list, neutral_list):
        '''
        Purpose...
        :param 
        :param 
        :return:
        '''
        return
    
    def _seg_to_DocVec(self, input_txt, model):
    #     # Doc2Vec infer_vector() could (as option?) offer deterministic result · Issue #447 · RaRe-Technologies/gensim
    #     # https://github.com/RaRe-Technologies/gensim/issues/447
    #     for seed_func in seed_funcs:
    #         seed_func(0)
        avgDoc2Vec = []
        for _ in range(1):
            avgDoc2Vec.append(np.array(model.infer_vector([input_txt])))
            
        avgDoc2Vec = np.mean(avgDoc2Vec, axis=0)
        
        # assert if avgDoc2Vec is nan
        #assert ~np.isnan(avgDoc2Vec).any()
        
        return avgDoc2Vec

    def _load_Doc2Vec(self, dbow='dbow_100_ADV_DIS_model', dmm='dmm_100_ADV_DIS_model'):
        model_dbow = Doc2Vec.load(f'./data/model/{dbow}.bin')
        model_dmm = Doc2Vec.load(f'./data/model/{dmm}.bin')
        # %%
        # Freeze the random seed and concatenate doc2vec models
        model_dbow.random.seed(0)
        model_dmm.random.seed(0)
        concate_model = ConcatenatedDoc2Vec([model_dbow, model_dmm])
        seed_funcs = [model_dbow.random.seed, model_dmm.random.seed]

        return concate_model

    def _transform(self, df, data_columns, meta_columns, model, augmentation_ratio=1, output_prefix=''):
        debug = False
        tmp_df = pd.DataFrame(columns=df.columns)
        for _ in range(augmentation_ratio):
            tmp_df2 = pd.DataFrame(columns=df.columns)
            print('===', meta_columns)
            tmp_df2[meta_columns] = df[meta_columns]
            for i_column in data_columns:
                # select none empty entries, and apply txt_to_clean, clean_to_seg, seg_to_DocVec
                if debug:
                    tmp_df2[i_column] = df[i_column].apply(lambda x: i_column)
                else:
                    tmp_df2[i_column] = df.loc[~df.loc[:,i_column].isnull()][i_column].apply(self._seg_to_DocVec, model=model)
            tmp_df = pd.concat([tmp_df, tmp_df2], ignore_index=True)
        
        if len(output_prefix) > 0:
            tmp_df.to_csv(f"./data/cleaned/judgment_result_doc2vec_{output_prefix}.csv", index=False)
        else:
            tmp_df.to_csv(f"./data/cleaned/judgment_result_doc2vec.csv", index=False)

        return tmp_df

    # //TODO Murphy：把 transform_by_doc2vec.py 放進來
    def transform_wrapper(self, df, df_neu, dbow='dbow_100_ADV_DIS_model', dmm='dmm_100_ADV_DIS_model'):
        '''
        Purpose...
        :param 
        :param 
        :return:
        '''

        df_list = [df]
        df_list_neu = [df_neu]

        for df, df_neu in zip(df_list, df_list_neu):
            all_neutral_columns = df_neu.columns[df_neu.columns.to_series().str.contains('neutral')].tolist()

            non_neutral_columns = sorted(list( \
                                    set(list(matplotlib.cbook.flatten(df.columns.tolist()))) - \
                                    set(all_neutral_columns) - \
                                    set(self.categorical) - \
                                    set(self.meta_info)))

            print('neutral columns: \n %s \n' % all_neutral_columns)
            print('non neutral columns: \n %s \n' % non_neutral_columns)

        concate_model = self._load_Doc2Vec(dbow, dmm)
        # %%
        # Transform  non neutral data
        df_list3 = []
        for df in df_list:
            non_neutral_meta_columns = self.categorical+self.meta_info
            df_list3.append(self._transform(df, \
                data_columns=non_neutral_columns, meta_columns=non_neutral_meta_columns, \
                model=concate_model, augmentation_ratio=1, output_prefix="non_neutral"))
        
        for df in df_list3:
            display(df)



        # Transform neutral data
        df_list3_neu = []
        for df in df_list_neu:
            neutral_meta_columns = 'ID'
            df_list3_neu.append(self._transform(df, \
                data_columns=all_neutral_columns, meta_columns=neutral_meta_columns, \
                model=concate_model, augmentation_ratio=1, output_prefix="neutral"))
        for df in df_list3_neu:
            display(df)
        
        return





if __name__=='__main__':
    d2v = Doc_to_Vector()
    advantage_list = pd.read_csv("./data/cleaned/sentence_advantage.csv")['0'].tolist()
    disadvantage_list = pd.read_csv("./data/cleaned/sentence_disadvantage.csv")['0'].tolist()
    neutral_list = pd.read_csv("./data/cleaned/sentence_neutral.csv")['0'].tolist()
    d2v.train_model(advantage_list, disadvantage_list, neutral_list)

    df = pd.read_csv("./data/cleaned/judgment_result_seg_jieba.csv")
    df_neu = pd.read_csv("./data/cleaned/judgment_result_seg_neu_jieba.csv")
    d2v.transform_wrapper(df, df_neu)

