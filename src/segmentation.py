# %%
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.utils import shuffle
from itertools import count
import data_preprocess as dp
from IPython.display import display

txt_to_clean = dp.txt_to_clean
clean_to_seg = dp.clean_to_seg



# 1. Import data
df = pd.read_csv('./data/cleaned/judgement_result_onehot.csv')
df_neu = pd.read_csv('./data/cleaned/judgement_result_neu.csv')
# %%
# 2. Seperating the neutral and non-neutral column name
import matplotlib

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
# %%
# 3. Extract the non-neutral sentence data: copy a new dataframe which only includes the non-neutral column in the original dataframe.
meta_info = ['filename', 'ID', 'Others']
categorical = ['Result','Willingness','Type', 'AK','RK','AN','RN']

df_list2 = []

for df in df_list:
    df2 = pd.DataFrame(columns=df.columns)
    df2[meta_info+categorical] = df[meta_info+categorical]
    df_list2.append(df2)
# %%
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


for df2 in df_list2:
    print(df2[non_neutral_columns])
# %%
# 4. Extract the neutral sentence data: copy a new dataframe which only includes the neutral column in the original dataframe.

df_list2_neu = []

for df in df_list_neu:
    df2 = pd.DataFrame(columns=df.columns)
    df2['ID'] = df['ID']
    df_list2_neu.append(df2)
# %%
# set True for debug
debug = False

for index, df, df2 in zip(count(), df_list_neu, df_list2_neu):
    for i_column in all_neutral_columns:
        # select none empty entries, and apply txt_to_clean, clean_to_seg, seg_to_DocVec
        if debug:
#             df2[i_column] = df[i_column].apply(lambda x: i_column)
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

# %%
df_output = df_list2[0]
df_neu_output = df_list2_neu[0]

# Debug

display(df_output)
display(df_neu_output)
# %%
# 5. Ouptut the seperated data.
df_output.to_csv("./data/cleaned/judgment_result_seg.csv", index=False)
df_neu_output.to_csv("./data/cleaned/judgment_result_seg_neu.csv", index=False)
# %%
