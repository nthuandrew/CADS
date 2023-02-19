from module.util import *
from class_Data_Cleaner import Data_Cleaner
from class_Segmentation import Segmentation
from class_BertForClassification_Wrapper import Bert_Wrapper


# Data_Cleaner()
df = pd.read_csv('./data/raw/labels_full.csv')
df_neu = pd.read_csv('./data/raw/neutral_sentences.csv')
clean = Data_Cleaner()
df = clean.nlp_custody_judgment(df, df_neu)
del df, df_neu, clean
gc.collect()

# Segmentation
df = pd.read_csv('./data/cleaned/judgement_result_onehot.csv')
df_neu = pd.read_csv('./data/cleaned/judgement_result_neu.csv')
seg = Segmentation(type='bert')
df_output, df_neu_output = seg.segment_custody_sentiment_analysis_articles_wrapper(df, df_neu)
del df, df_neu, df_output, df_neu_output, seg
gc.collect()

# BERT: Classification for custody sentiment analysis
df = pd.read_pickle('./data/cleaned/judgment_result_seg_bert.pkl')
df_neu = pd.read_pickle('./data/cleaned/judgment_result_seg_neu_bert.pkl')
bw = Bert_Wrapper(num_labels = 2)
trainloader, validloader, testloader = bw.prepare_custody_sentiment_analysis_dataloader(df, df_neu)
bw.initialize_training()
bw.train()
bw.evaluate()