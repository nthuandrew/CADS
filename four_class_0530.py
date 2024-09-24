import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
seed = 1

df = pd.read_excel('data/final/sentence/four_class_0530.xlsx')
target_columns = ['TextID' , 'Title' ,'Sentence','無標註','自殺與憂鬱','自殺行為','其他類型']
df = df[target_columns]
all_labelize_df = []
new_column=['無標註','自殺與憂鬱','自殺行為','其他類型']
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
    save_folder = f"data/final/sentence/four_class/seed_{seed}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    train_df.to_excel(f"{save_folder}/four_class_0530_train_fold_{i+1}.xlsx".format(i))
    test_df.to_excel(f"{save_folder}/four_class_0530_test_fold_{i+1}.xlsx".format(i))