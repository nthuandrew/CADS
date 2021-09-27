from module.util import *
from data.GV import *

#//TODO: 弘祥 solid
class Data_Cleaner():
    def __init__(self):
        return

    
    def show_info(self):

        return


    def nlp_custody_judgment(self, df, df_neu, keep_result=['a', 'b', 'c'], keep_willingness=['a']):
        '''
        '''
        self.info_dict={}

        # 1.1 Data Selection
        # Select result
        # a: 提出方勝, b: 相對方勝, c: 雙方共有, d: 其他人勝, e: 非親權（本次不探討）, f: 子女判給不同方
        df = df.loc[df['Result'].isin(keep_result)]

        print("Result count:", Counter(df['Result']))
        print("Willingness count:", Counter(df['Willingness']))
        
        # Select Willingness
        # a: 雙方有意願, b: 提出方有意願, c: 相對方有意願, d: 雙方無意願, e: 其他
        df = df.loc[ df['Willingness'].isin(keep_willingness)]

        print("Willingness count:", Counter(df['Willingness']))
        print("Result count:", Counter(df['Result']))
        
        # 1.2 show info
        # show df info
        print("total conut: ", len(df))
        categorical = ['Result','Willingness','Type', 'AK','RK','AN','RN']
        for i_column in categorical:
            print('%s count: %s' % (i_column, Counter(df[i_column])))
        
        # show type=a info
        df_tmp2 = df[df['Type']=='a']
        print("total conut: ", len(df_tmp2))
        for i_column in categorical:
            print('%s count: %s' % (i_column, Counter(df_tmp2[i_column])))
        
        # show type=b info
        df_tmp2 = df[df['Type']=='b']
        print("total conut: ", len(df_tmp2))
        for i_column in categorical:
            print('%s count: %s' % (i_column, Counter(df_tmp2[i_column])))
        
        # 1.3 Subset neutrol df
        df_neu = df_neu[df_neu['ID'].isin(df['ID'])]
        
        # Save subsets of dfs
        df.to_csv("./data/cleaned/judgement_result.csv", index=False)
        df_neu.to_csv("./data/cleaned/judgement_result_neu.csv", index=False)
        
        
        # 2. onehot encoding
        '''
        AK: 提出方身分
        (options)a:父, b:母, c:男性親屬, d:女性親屬, e:其他
        RK: 相對方身分
        (options)a:父, b:母, c:男性親屬, d:女性親屬, e:其他
        AN: 提出方國籍
        (options)a:本國, b:大陸籍, c:其他, d:雙重國籍
        RN: 相對方國籍
        (options)a:本國, b:大陸籍, c:其他, d:雙重國籍
        Type: a.一般親權（酌定） b.改定親權（改訂）
        '''
        categorical = ['Result','Willingness','AK','RK','AN','RN', 'Type']
        df2 = df

        for i_column in categorical:
            print(i_column)

            # split labels, expand and stack it as new index levels
            df_tmp2 = df[i_column].str.split('|', expand=True).stack()
            print(Counter(df_tmp2))
            # one-hot encode, groupby level 0 and sum it
            df_tmp2 = pd.get_dummies(df_tmp2).groupby(level=0).sum()

            # search for multiple labels (mutiple ones)
            df_tmp2.apply(lambda x: print(x) if sum(x) != 1 else x, axis=1)

            # apply to np.array
            df2[i_column] = df_tmp2.apply(lambda x: tuple(x), axis=1).apply(np.array)

        # Save onehot
        df2.to_csv("./data/cleaned/judgement_result_onehot.csv", index=False)
        

        # Fina out the column names of adv/dis/neutral sentences. 
        # Find out the column names of adv/dis/neutral sentences. 
        # The following example shows the naming rule of advantage sentences columns: AA1, AA2.
        # 3. Seperate sentences into advantage, disadvantage, and neutral
        applicant_advantage_column = df.columns[df.columns.to_series().str.contains('AA')].tolist()
        respondent_advantage_column = df.columns[df.columns.to_series().str.contains('RA')].tolist()
        applicant_disadvantage_column = df.columns[df.columns.to_series().str.contains('AD')].tolist()
        respondent_disadvantage_column = df.columns[df.columns.to_series().str.contains('RD')].tolist()
        neutral_column = df_neu.columns[df_neu.columns.to_series().str.contains('neutral')].tolist()


        advantage_column = applicant_advantage_column + respondent_advantage_column
        disadvantage_column = applicant_disadvantage_column + respondent_disadvantage_column

        print(advantage_column)
        print(disadvantage_column)
        print(neutral_column)
        
        # 
        advantage_list=[]
        disadvantage_list=[]
        neutral_list=[]

        
        # Exracting advantage/disadvantage/neutral sentence according to their column names by output_to_list(func)
        df2.loc[:,advantage_column].apply(output_to_list, content_list=advantage_list)
        df2.loc[:,disadvantage_column].apply(output_to_list, content_list=disadvantage_list)
        df_neu.loc[:,neutral_column].apply(output_to_list, content_list=neutral_list)
        
        # Save sentences
        pd.DataFrame(advantage_list).to_csv("data/cleaned/sentence_advantage.csv", index=False)
        pd.DataFrame(disadvantage_list).to_csv("data/cleaned/sentence_disadvantage.csv", index=False)
        pd.DataFrame(neutral_list).to_csv("data/cleaned/sentence_neutral.csv", index=False)
        

        #//TODO: csu Add txt_to_clean here
        
        


        return
    
    def test(self, df):
        df = sorted(df)
        
        return df



if __name__=='__main__':
    # clean custody judgements for sentiment analysis
    df = pd.read_csv('./data/raw/labels_full.csv')
    df_neu = pd.read_csv('./data/raw/neutral_sentences.csv')
    clean = Data_Cleaner()
    df = clean.nlp_custody_judgment(df, df_neu)

    # clean custody judgements factor for factor classification
    # None
