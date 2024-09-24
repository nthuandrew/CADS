import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
seed = 1

df = pd.read_excel('data/final/sentence/four_class_0530.xlsx')
target_columns = ['TextID' , 'Title' ,'Sentence','無標註','自殺與憂鬱','無助或無望','正向文字','其他負向文字','生理反應或醫療狀況','自殺行為']
df = df[target_columns]
all_labelize_df = []
new_column=['無標註','自殺與憂鬱','無助或無望','正向文字','其他負向文字','生理反應或醫療狀況','自殺行為']
for label in new_column:
    if label == '無標註':
        temp = df[df[label] == 1].loc[:]
        all_labelize_df.append(temp[temp['正向文字'] == 0].loc[:])
    elif label == '正向文字':
        temp = df[df[label] == 1].loc[:]
        # set the label of "無標註" to 0
        temp["無標註"] = [0]*len(temp)
        all_labelize_df.append(temp)

    else:
        all_labelize_df.append(df[df[label] == 1].loc[:])
sum_ = 0
for i in all_labelize_df:
    print(len(i))
    # print(i.columns)
    sum_ += len(i)
print("Sum: ", sum_)

all_train_df = [[] for _ in range(5)]
all_test_df = [[] for _ in range(5)]
skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
for i in range(len(all_labelize_df)):
    for idx, (train_index, test_index) in enumerate(skf.split(all_labelize_df[i], [0]*len(all_labelize_df[i]))):
        train_df = all_labelize_df[i].iloc[train_index]
        test_df = all_labelize_df[i].iloc[test_index]
        all_train_df[idx].append(train_df)
        all_test_df[idx].append(test_df)

to_save_num = 300
for i in range(5):
    train_df = pd.concat(all_train_df[i], axis=0)
    test_df = pd.concat(all_test_df[i], axis=0)
    save_folder = f"data/final/sentence/seven_class/seed_{seed}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    train_df.to_excel(f"{save_folder}/seven_class_0530_train_fold_{i+1}.xlsx".format(i))
    test_df.to_excel(f"{save_folder}/seven_class_0530_test_fold_{i+1}.xlsx".format(i))
    # check the distribution of each label, if the number of label is less than to_save_num, then save it; else, sample it
    samples_df = []
    for idx, label in enumerate(new_column):
        if label == '無標註':
            temp = train_df[train_df[label] == 1].loc[:]
            temp = temp[temp['正向文字'] == 0].loc[:]
        elif label == '正向文字':
            temp = df[df[label] == 1].loc[:]
            # set the label of "無標註" to 0
            temp["無標註"] = [0]*len(temp)
        else:
            temp = train_df[train_df[label] == 1].loc[:]
        if len(temp) > to_save_num:
            temp = temp.sample(n=to_save_num, random_state=seed)
        samples_df.append(temp)
    sample_df = pd.concat(samples_df, axis=0)
    # random shuffle
    sample_df = sample_df.sample(frac=1, random_state=seed)
    # show statistics of each label
    for idx, label in enumerate(new_column):
        print(f"Fold {i+1} {label}: {len(sample_df[sample_df[label] == 1])}")
    sample_df.to_excel(f"{save_folder}/seven_class_0530_train_raw_fold_{i+1}.xlsx".format(i))
    