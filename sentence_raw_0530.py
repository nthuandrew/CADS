
import pandas as pd

seed = 1
# fold = 5
for fold in range(1, 6):
    df = pd.read_excel(f'data/final/sentence/four_class/seed_{seed}/four_class_0530_train_fold_{fold}.xlsx')
    target_columns = ['TextID' , 'Title' ,'Sentence','無標註','自殺與憂鬱','自殺行為','其他類型']
    df = df[target_columns]
    labelize_df = []
    new_column=['無標註','自殺與憂鬱','自殺行為','其他類型']
    for label in new_column:
        labelize_df.append(df[df[label] == 1].loc[:])
    for i in labelize_df:
        print(len(i))
   
    to_train = [1000, 1000, len(labelize_df[2]), 1000]
    training_df = []

    augment_id = 0
    augment_num = 3
    for idx, a_df in enumerate(labelize_df):
        training_df.append(a_df.sample(n=to_train[idx], random_state=seed))
    for df in training_df:
        print(df.shape)


    augmented_df = pd.concat(training_df,axis=0, ignore_index=True)
    augmented_df.to_excel(f'data/final/sentence/four_class/seed_{seed}/four_class_0530_train_raw_fold_{fold}.xlsx')



