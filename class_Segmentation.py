from module.util import *
from data.GV import *

class Segmentation():
    def __init__(self, df):
        self.df=df
        return

    def run_jieba(self):
        '''

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


        # //TODO: csu go through the pipeline
        # 3. Segement articles
        # Read in dfs
        df = pd.read_csv('./data/cleaned/judgement_result_onehot.csv')
        df_neu = pd.read_csv('./data/cleaned/judgement_result_neu.csv')
        
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
        
        # 
        # set True for debug
        debug = False

        for index, df, df2 in zip(count(), df_list, df_list2):
            for i_column in non_neutral_columns:
                # select none empty entries, and apply txt_to_clean, clean_to_seg, seg_to_DocVec
                if debug:
                    df2[i_column] = df[i_column].apply(lambda x: i_column)
                else:
                    #df2[i_column] = df[i_column].apply(lambda x: np.zeros(200))
                    #df2[i_column] = df[i_column].apply(txt_to_clean, textmode=True).apply(clean_to_seg, textmode=True)
                    df2[i_column] = df.loc[~df.loc[:,i_column].isnull()][i_column].apply(txt_to_clean, textmode=True).apply(clean_to_seg, textmode=True)
                    #display(df.loc[~df.loc[:,i_column].isnull()][i_column])
                    # print("debug")
            # display(df.loc[df.loc[:,i_column].isnull()][i_column])

        #
        df_list2_neu = []

        for df in df_list_neu:
            df2 = pd.DataFrame(columns=df.columns)
            df2['ID'] = df['ID']
            df_list2_neu.append(df2)
        # 
        # set True for debug
        debug = False

        for index, df, df2 in zip(count(), df_list_neu, df_list2_neu):
            for i_column in all_neutral_columns:
                # select none empty entries, and apply txt_to_clean, clean_to_seg, seg_to_DocVec
                if debug:
                    #df2[i_column] = df[i_column].apply(lambda x: i_column)
                    df[i_column] = df[i_column].apply(lambda x: np.nan if type(x)==int else x)
                    df2[i_column] = df[i_column].apply(lambda x: print(x) if type(x) != str else None)
                else:
                    #df2[i_column] = df[i_column].apply(lambda x: np.zeros(200))
                    #df2[i_column] = df[i_column].apply(txt_to_clean, textmode=True).apply(clean_to_seg, textmode=True)
                    df[i_column] = df[i_column].apply(lambda x: np.nan if type(x)==int else x)
                    df2[i_column] = df.loc[~df.loc[:,i_column].isnull()][i_column].apply(txt_to_clean, textmode=True).apply(clean_to_seg, textmode=True)
                    #display(df.loc[~df.loc[:,i_column].isnull()][i_column])
                    # print("debug")
            # display(df.loc[df.loc[:,i_column].isnull()][i_column])

        #
        df_output = df_list2[0]
        df_neu_output = df_list2_neu[0]

        # 
        df_output.to_csv("./data/cleaned/judgment_result_seg.csv")
        df_neu_output.to_csv("./data/cleaned/judgment_result_seg_neu.csv")
        
        #//TODO: csu remove extra_dict folder from the root after finish this class.

        return df_output, df_neu_output
    
    def run_bert(self, language):
        
        return

    # TODO: Murphy -> def _wrap_data()



if __name__=='__main__':
    df = pd.read_csv('./data/cleaned/judgement_result_onehot.csv')
    df_neu = pd.read_csv('./data/cleaned/judgement_result_neu.csv')
    seg = Segmentation()
    df, df_neu = seg.run_jieba(df, df_neu)
