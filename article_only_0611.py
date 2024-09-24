import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
seed = 1

df = pd.read_excel('data/final/article/A1A2_article.xlsx')
# target_columns = ['Content' , 'Crisis_Level', 'type1_1', 'type1_0']
target_columns = ['Content' , 'Crisis_Level', 'a:A', 'b:B', 'c:C', 'd:0']
df = df[target_columns]
all_labelize_df = []
new_column=['d:0','c:C','b:B', 'a:A' ]
# new_column=['type1_0','type1_1' ]
for label in new_column:
    all_labelize_df.append(df[df[label] == 1].loc[:])
for i in all_labelize_df:
    print(len(i))
    print(i.columns)

all_train_df = [[] for _ in range(5)]
all_test_df = [[] for _ in range(5)]
skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
for i in range(len(all_labelize_df)):
    for idx, (train_index, test_index) in enumerate(skf.split(all_labelize_df[i], [0]*len(all_labelize_df[i]))):
        train_df = all_labelize_df[i].iloc[train_index]
        test_df = all_labelize_df[i].iloc[test_index]
        all_train_df[idx].append(train_df)
        all_test_df[idx].append(test_df)

for i in range(5):
    train_df = pd.concat(all_train_df[i], axis=0)
    test_df = pd.concat(all_test_df[i], axis=0)
    save_folder = f"data/final/sentence/article/seed_{seed}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    train_df.to_excel(f"{save_folder}/article_type3_train_fold_{i+1}.xlsx".format(i))
    test_df.to_excel(f"{save_folder}/article_type3_test_fold_{i+1}.xlsx".format(i))
    A1A2_df = pd.read_excel(f"{save_folder}/article_type3_train_fold_{i+1}.xlsx".format(i))
    type_A_num = sum(A1A2_df['a:A'])
    type_num = [ 150, 150, 150, type_A_num]
    all_kind_of_df = []
    to_save_df = []
    all_kind_of_df.append(A1A2_df[A1A2_df['d:0'] == 1])
    all_kind_of_df.append(A1A2_df[A1A2_df['c:C'] == 1])
    all_kind_of_df.append(A1A2_df[A1A2_df['b:B'] == 1])
    all_kind_of_df.append(A1A2_df[A1A2_df['a:A'] == 1])


    for j, a_df in enumerate(all_kind_of_df):
        # randomly choose type_A_num data from the dataframe
        a_df = a_df.sample(n=type_num[j], random_state=seed)
        to_save_df.append(a_df)
    final_df = pd.concat(to_save_df)
    # shuffle the dataframe
    final_df = final_df.sample(frac=1, random_state=seed)
    # show statistics
    print(final_df.shape)
    print(final_df['a:A'].sum())
    print(final_df['b:B'].sum())
    print(final_df['c:C'].sum())
    print(final_df['d:0'].sum())
    final_df.to_excel(f"{save_folder}/article_type3_train_raw_fold_{i+1}.xlsx".format(i))