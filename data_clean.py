# %%
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import json
# %%
df = pd.read_csv('./data/raw/labels_full.csv')
df_neu = pd.read_csv('./data/raw/neutral_sentences.csv')
# %%
keep_result = ['a', 'b', 'c']
df = df.loc[df['Result'].isin(keep_result)]

print("Result count:", Counter(df['Result']))
print("Willingness count:", Counter(df['Willingness']))
# %%
# select only Willingness == 'a'
df = df.loc[ df['Willingness'] == 'a']
# exclude Result == 'e'
#df = df[~(df['Result'] == 'e')]

print("Willingness count:", Counter(df['Willingness']))
print("Result count:", Counter(df['Result']))
# %%
# clean labels
import numpy as np
from collections import Counter

print("total conut: ", len(df))

categorical = ['Result','Willingness','Type', 'AK','RK','AN','RN']

for i_column in categorical:
    print('%s count: %s' % (i_column, Counter(df[i_column])))
# %%
df_tmp2 = df[df['Type']=='a']

print("total conut: ", len(df_tmp2))

categorical = ['Result','Willingness','Type', 'AK','RK','AN','RN']

for i_column in categorical:
    print('%s count: %s' % (i_column, Counter(df_tmp2[i_column])))
# %%
df_tmp2 = df[df['Type']=='b']

print("total conut: ", len(df_tmp2))

categorical = ['Result','Willingness','Type', 'AK','RK','AN','RN']

for i_column in categorical:
    print('%s count: %s' % (i_column, Counter(df_tmp2[i_column])))
# %%
df_neu = df_neu[df_neu['ID'].isin(df['ID'])]
# %%
df.to_csv("./data/cleaned/judgement_result.csv")
df_neu.to_csv("./data/cleaned/judgement_result_neu.csv")
# %%
from collections import Counter

# categorical = ['Result','Willingness','AK','RK','AN','RN']
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

# %%
df2.to_csv("./data/cleaned/judgement_result_onehot.csv")
# %%
import numpy as np

def output_to_list(content, content_list):
    #print(type(pd.Series()))
    if type(content) is type(pd.Series()):
        #print(type(pd.Series()))
        content.apply(output_to_list, content_list=content_list)
    elif type(content) is float:
        if not np.isnan(content):
            content_list.append(content)
    elif content is not np.nan:
        content_list.append(content)
# %%
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
# %%
advantage_list=[]
disadvantage_list=[]
neutral_list=[]

df2.loc[:,advantage_column].apply(output_to_list, content_list=advantage_list)
df2.loc[:,disadvantage_column].apply(output_to_list, content_list=disadvantage_list)
df_neu.loc[:,neutral_column].apply(output_to_list, content_list=neutral_list)
# %%
pd.DataFrame(advantage_list).to_csv("data/cleaned/sentence_advantage.csv")
pd.DataFrame(disadvantage_list).to_csv("data/cleaned/sentence_disadvantage.csv")
pd.DataFrame(neutral_list).to_csv("data/cleaned/sentence_neutral.csv")
# %%
