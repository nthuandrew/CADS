from module.util import *
from data.GV import *
# %%
class Segmentation():
    def __init__(self, type='jieba'):
        if type == 'jieba':
            self.clean2seg = clean_to_seg_by_jieba
        elif type == "bert":
            self.clean2seg = clean_to_seg_by_tokenizer
        self.txt2clean = txt_to_clean
        self.type=type
        return

    # //TODO: Murphy 調整 clean_to_seg, segment_articles() 使 bert 可用
    def _segment_articles(self, df_list, df_list2, columns):
        '''
        Purpose...
        :param 
        :param 
        :return:
        '''
        debug = False

        for df, df2 in zip(df_list, df_list2):
            for i_column in columns:
                # select none empty entries, and apply txt_to_clean, clean_to_seg, seg_to_DocVec
                if debug:
                    df2[i_column] = df[i_column].apply(lambda x: i_column)
                else:
                    df2[i_column] = df.loc[~df.loc[:,i_column].isnull()][i_column].apply(self.txt2clean, textmode=True).apply(self.clean2seg, textmode=True)
        return df_list2  


    def _initialize_jieba(self):
        '''
        Initialize jieba.
        Ref:
        # [fxsjy/jieba: 结巴中文分词](https://github.com/fxsjy/jieba)
        # [关于idf.txt · Issue #87 · fxsjy/jieba](https://github.com/fxsjy/jieba/issues/87)
        '''
        #//TODO: csu skip 1. if data has been created
        # 1. Clean 法律辭典
        # read keyword file
        keyword_list=pd.read_excel('data/extra_dict/KWIC中英關鍵字-全部法律關鍵字2018-08-22 14-14.xlsx', engine='openpyxl')

        # remove some pattern
        unused_punctuations = r'([（）「」。，：！？【】#@\/$《》／；\-『』～?+…⋯〈〉:,─x│])'
        unused_punctuations2 = r'([、\(\)])'
        default_pattern = r'(<script[\s\S]+?/script>|<.*?>|\r|&nbsp;|\u3000|\xa0|\n|[.][.]+)'
        patterns_list = [unused_punctuations, default_pattern]
        pattern=re.compile("|".join(patterns_list))

        keyword_list['KWIC中英關鍵字_mod'] = keyword_list['KWIC中英關鍵字'].str.replace(pat=pattern,repl='')

        # Save
        source_dir = 'data/'
        base_path = source_dir+'extra_dict/'
        keyword_list['KWIC中英關鍵字'][4:].to_csv(base_path+'keyword_list_ori.csv',encoding='utf8', index=False)
        keyword_list['KWIC中英關鍵字_mod'][4:].to_csv(base_path+'keyword_list_mod.csv',encoding='utf8', index=False)
        

        # 2. Initialize jieba
        base_path = 'data/extra_dict/'
        methods_name = {'dict1':0, 'dict2':1}
        stop_words_filenames = ['stop_words.txt','stop_words.txt']
        dict_filenames = ['dict.txt.big', 'dict.txt.big']
        idf_filenames = ['idf.txt.big', 'idf.txt.big']
        userdict_filename = [['keyword_list_mod.csv', 'custom_dict.txt'],['keyword_list_mod.csv', 'custom_dict.txt', 'classical_chinese_dict_zhtw.txt']]

        stop_words_path_list = [str(base_path+s) for s in stop_words_filenames]
        dict_path_list = [str(base_path+s) for s in dict_filenames]
        idf_path_list = [str(base_path+s) for s in idf_filenames]
        userdict_list = [[str(base_path+s) for s in userdict_filename[i]] for i in range(len(userdict_filename))]

        init_jieba(*list(zip(stop_words_path_list, dict_path_list, idf_path_list, userdict_list))[methods_name['dict1']])
        return
                                     
    def segment_criminal_sentiment_analysis_articles_wrapper(self, \
        df, \
        categorical_info, meta_info=[], target_columns=["Sentence"], criminal_type="sex"):
        df_list = [df]
        categorical = categorical_info
        #meta_info = ['TextID']
        # meta_info=[meta_info]

        df_list2 = []

        for df in df_list:
            df2 = pd.DataFrame(columns=df.columns)
            df2[meta_info+categorical] = df[meta_info+categorical]
            df_list2.append(df2)

        # target_columns = [target_columns]
        df_output = self._segment_articles(df_list, df_list2, target_columns)[0]
        display(df_output)
        df_output.to_csv(f"./data/cleaned/criminal_{criminal_type}_seg_{self.type}.csv", index=False)
        df_output.to_pickle(f"./data/cleaned/criminal_{criminal_type}_seg_{self.type}.pkl")
        return df_output

    def segment_custody_judgement_factor_articles_wrapper(self, df, df_neu):
        df_list = [df]
        df_list_neu = [df_neu]

        categorical = ['Result','Win','Ans', 'Label', 'COUNT', '親子感情', '意願能力', '父母經濟', '支持系統', '父母生活', '主要照顧',
       '子女年齡', '人格發展', '父母健康', '父母職業', '子女意願', '友善父母', '父母品行', 'AK','RK','AN','RN']
        meta_info = ['Index', 'filename', 'ID']

        for df, df2 in zip(df_list,df_list_neu):
            all_neutral_columns = df2.columns[df2.columns.to_series().str.contains('neutral')].tolist()
            # TODO: 要改
            non_neutral_columns = sorted(list( \
                                    set(list(matplotlib.cbook.flatten(df.columns.tolist()))) - \
                                    set(all_neutral_columns) - \
                                    set(categorical) - \
                                    set(meta_info)))

            print('neutral columns: \n %s \n' % all_neutral_columns)
            print('non neutral columns: \n %s \n' % non_neutral_columns)

        df_list2 = []

        for df in df_list:
            df2 = pd.DataFrame(columns=df.columns)
            df2[meta_info+categorical] = df[meta_info+categorical]
            df_list2.append(df2)

        df_output = self._segment_articles(df_list, df_list2, non_neutral_columns)[0]
        print(df_output[non_neutral_columns])
        
        df_list2_neu = []

        for df in df_list_neu:
            df2 = pd.DataFrame(columns=df.columns)
            df2['ID'] = df['ID']
            df_list2_neu.append(df2)
        df_neu_output = self._segment_articles(df_list_neu, df_list2_neu, all_neutral_columns)[0]

        display(df_output)
        display(df_neu_output)
        df_output.to_csv(f"./data/cleaned/judgment_factor_seg_{self.type}.csv", index=False)
        df_neu_output.to_csv(f"./data/cleaned/judgment_factor_seg_neu_{self.type}.csv", index=False)
        df_output.to_pickle(f"./data/cleaned/judgment_factor_seg_{self.type}.pkl")
        df_neu_output.to_pickle(f"./data/cleaned/judgment_factor_seg_neu_{self.type}.pkl")
        return df_output, df_neu_output
        

    def segment_custody_sentiment_analysis_articles_wrapper(self, df, df_neu):
        '''
        Purpose...
        :param 
        :param 
        :return:
        '''

        # //TODO csu check if this work
        if self.type == 'jieba':
            self._initialize_jieba()
        
         
        # 將 neutral 與 非neutral 分開
        categorical = ['Result','Willingness','AK','RK','AN','RN', 'Type']
        meta_info = ['filename', 'ID', 'Others']
        # column_prefixes = ['AA', 'RA', 'AD', 'RD', 'neutral']

        df_list = [df]
        df_list_neu = [df_neu]

        for df, df2 in zip(df_list,df_list_neu):
            all_neutral_columns = df2.columns[df2.columns.to_series().str.contains('neutral')].tolist()

            non_neutral_columns = sorted(list( \
                                    set(list(matplotlib.cbook.flatten(df.columns.tolist()))) - \
                                    set(all_neutral_columns) - \
                                    set(categorical) - \
                                    set(meta_info)))

            print('neutral columns: \n %s \n' % all_neutral_columns)
            print('non neutral columns: \n %s \n' % non_neutral_columns)

        # 
        meta_info = ['filename', 'ID', 'Others']
        categorical = ['Result','Willingness','Type', 'AK','RK','AN','RN']

        df_list2 = []

        for df in df_list:
            df2 = pd.DataFrame(columns=df.columns)
            df2[meta_info+categorical] = df[meta_info+categorical]
            df_list2.append(df2)

        df_output = self._segment_articles(df_list, df_list2, non_neutral_columns)[0]
        
        print(df_output[non_neutral_columns])
        
        df_list2_neu = []

        for df in df_list_neu:
            df2 = pd.DataFrame(columns=df.columns)
            df2['ID'] = df['ID']
            df_list2_neu.append(df2)

        df_neu_output = self._segment_articles(df_list_neu, df_list2_neu, all_neutral_columns)[0]
        # Debug
        display(df_output)
        display(df_neu_output)
        df_output.to_csv(f"./data/cleaned/judgment_result_seg_{self.type}.csv", index=False)
        df_neu_output.to_csv(f"./data/cleaned/judgment_result_seg_neu_{self.type}.csv", index=False)
        df_output.to_pickle(f"./data/cleaned/judgment_result_seg_{self.type}.pkl")
        df_neu_output.to_pickle(f"./data/cleaned/judgment_result_seg_neu_{self.type}.pkl")
        
        #//TODO: csu remove extra_dict folder from the root after finish this class.

        return df_output, df_neu_output
    
    




if __name__=='__main__':
    ############ Segment judgement for sentiment analysis #############
    # df = pd.read_csv('./data/cleaned/judgement_result_onehot.csv')
    # df_neu = pd.read_csv('./data/cleaned/judgement_result_neu.csv')
    # seg = Segmentation(type="bert")
    # df_output, df_neu_output = seg.segment_custody_sentiment_analysis_articles_wrapper(df, df_neu)
    ############ END #############

    ############ Segment jugement factor for factor classification #############
    # df = pd.read_excel('data/raw/data_features.xlsx')
    # df_neu = pd.read_csv('./data/cleaned/judgement_result_neu.csv')
    # seg = Segmentation(type="bert")
    # df_output, df_neu_output = seg.segment_custody_judgement_factor_articles_wrapper(df, df_neu)
    ############ END #############

    ############ Segment criminal for sentiment analysis #############
    criminal_type="drug"
    df = pd.read_excel(f'data/raw/data_criminal_{criminal_type}.xlsx')
    # df = pd.read_excel(f'data/raw/data_criminal_{criminal_type}_neutral.xlsx')
    seg = Segmentation(type="bert")
    # For sex
    # categorical_info = ['犯罪後之態度', '犯罪之手段與所生損害', '被害人的態度',
    #    '被告之品行', '其他審酌事項', '有利', '中性', '不利']
    # For gun
    # categorical_info = ['犯罪後之態度', '犯罪所生之危險或違反義務之程度', '被告之品行',
    #     '其他審酌事項', '有利', '中性', '不利']
    # For drug
    categorical_info = ['犯罪後之態度', '犯罪所生之危險或損害或違反義務之程度', '被告之品行',
        '其他審酌事項', '有利', '中性', '不利']

    df = seg.segment_criminal_sentiment_analysis_articles_wrapper(df, \
        categorical_info, meta_info=None, target_columns="Sentence", criminal_type=criminal_type)
     ############ END #############
# %%


# %%
