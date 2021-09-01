from module.util import *
from data.GV import *

class Word_to_Vector():
    def __init__(self):
        
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
    

    # //TODO Murphy：把 transform_by_doc2vec.py 放進來
    def transform(self, df, df_neu):
        '''
        Purpose...
        :param 
        :param 
        :return:
        '''
        return





if __name__=='__main__':
    w2v = Word_to_Vector()
    advantage_list = pd.read_csv("./data/cleaned/sentence_advantage.csv")['0'].tolist()
    disadvantage_list = pd.read_csv("./data/cleaned/sentence_disadvantage.csv")['0'].tolist()
    neutral_list = pd.read_csv("./data/cleaned/sentence_neutral.csv")['0'].tolist()
    w2v.train_model(advantage_list, disadvantage_list, neutral_list)

    df = pd.read_csv("./data/cleaned/judgment_result_seg.csv")
    df_neu = pd.read_csv("./data/cleaned/judgment_result_seg_neu.csv")
    w2v.transform(df, df_neu)

