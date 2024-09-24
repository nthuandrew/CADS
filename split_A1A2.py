import re
import pandas as pd
from src.article_predict import split_sentence
    
df = pd.read_excel("./data/final/article/A1A2_article.xlsx")
df['split_sentences'] = df.apply(lambda row: [sent for sent in split_sentence(row['Content']) if sent not in ['', ' ', '。', '，', '\n']], axis=1)
#  = [sent for sent in split_sentence(row['Content(remove_tag)']) if sent not in ['', ' ', '。', '，', '\n']]
# df.to_excel('split_output.xlsx')
# output_df = pd.DataFrame(columns=['TextID', 'Title', 'Crisis_Level','Sentence', '無標註', '自殺與憂鬱', '自殺行為', '其他類型'])
output_df = pd.DataFrame(columns=['TextID', 'Title', 'Crisis_Level','Sentence'])
k = 0
for index, row in df.iterrows():
    if k % 100 == 0:
        print(k)
    k += 1
    textID = row['TextID']
    title = row['Title']
    crisis_level = row['Crisis_Level']
    for sent in row['split_sentences']:
        # small_df = pd.DataFrame({'TextID':[textID], 'Title':[title], 'Crisis_Level': [crisis_level],'Sentence':[sent],'無標註':[1], '自殺與憂鬱':[0], '自殺行為':[0], '其他類型':[0] })
        small_df = pd.DataFrame({'TextID':[textID], 'Title':[title], 'Crisis_Level': [crisis_level],'Sentence':[sent]})
        output_df = pd.concat([output_df, small_df],ignore_index=True)
# print(output_df.append({'TextID':'1', 'Title':"2", 'Sentence':"3"},ignore_index=True))
print(len(output_df))
print(output_df.shape)
# output_df['無標註'] = [0]*len(output_df)
# output_df['自殺與憂鬱'] = [0]*len(output_df)
# output_df['自殺行為'] = [0]*len(output_df)
# output_df['其他類型'] = [0]*len(output_df)
output_df.to_excel('./data/final/article/split_A1A2_article.xlsx')
