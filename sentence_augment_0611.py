
import pandas as pd

seed = 1
# fold = 5
for fold in range(1, 6):
    df = pd.read_excel(f'data/final/sentence/seven_class/seed_{seed}/seven_class_0530_train_fold_{fold}.xlsx')
    target_columns = ['TextID' , 'Title' ,'Sentence', '無標註','自殺與憂鬱','無助或無望','正向文字','其他負向文字','生理反應或醫療狀況','自殺行為']
    df = df[target_columns]
    labelize_df = []
    new_column=['無標註','自殺與憂鬱','無助或無望','正向文字','其他負向文字','生理反應或醫療狀況','自殺行為']
    for label in new_column:
        labelize_df.append(df[df[label] == 1].loc[:])
    for i in labelize_df:
        print(len(i))
        # print(i.columns)
    # break

    # In[60]:


    # split Augmentation Data
    all_middle_df = labelize_df[0]
    print(all_middle_df.shape)
    to_add = [2000, 2000, 2000, 2000, 2000]
    Augment_df = []
    for i in range(5):
        Augment_df.append(all_middle_df.sample(n=to_add[i], random_state=seed))
        all_middle_df = all_middle_df.drop(Augment_df[-1].index)
    print(all_middle_df.shape)
    labelize_df[0] = all_middle_df

    for i in range(len(labelize_df)):
        labelize_df[i] = labelize_df[i].reset_index(drop=True)
    # for df in labelize_df:
    #     print(df)
    for i in range(len(Augment_df)):
        Augment_df[i] = Augment_df[i].reset_index(drop=True)
    for df in Augment_df:
        print(df.shape)


    to_train = [3000, len(labelize_df[1]),0, 0, 3000,0,0]
    target_num = 2500
    training_df = []

    augment_id = 0
    augment_num = 3
    for idx, a_df in enumerate(labelize_df):
        print("labelize: ",a_df.shape)
        # augment_label = new_column[idx]
        if a_df.shape[0] > target_num:
            training_df.append(a_df.sample(n=to_train[idx], random_state=seed))
            print("augment: ",training_df[-1].shape)
        else:
            to_augment_df = a_df
            boundary = a_df.shape[0]
            for i in range(len(Augment_df[augment_id])):
                # try: 
                    # print(idx, i)
                    if len( str(Augment_df[augment_id].at[i, 'Sentence'])) < augment_num:
                        augment_sentence = str(a_df.at[i % boundary, 'Sentence']) + str(Augment_df[augment_id].at[i, 'Sentence'])
                    else:
                        augment_sentence = str(a_df.at[i % boundary, 'Sentence']) + str(Augment_df[augment_id].at[i, 'Sentence'])[:augment_num]
                    to_augment_df = pd.concat([to_augment_df, a_df.iloc[i % boundary:(i % boundary)+1]], ignore_index=True)
                    to_augment_df.at[boundary+i, 'Sentence'] = augment_sentence
                # except:
                #     print(i % boundary)
            training_df.append(to_augment_df)
            print("augment: ",to_augment_df.shape)
            augment_id += 1
            to_add = to_train
    for df in training_df:
        print(df.shape)
    # break

    # In[67]:



    augmented_df = pd.concat(training_df,axis=0, ignore_index=True)
    augmented_df.to_excel(f'data/final/sentence/seven_class/seed_{seed}/seven_class_0530_train_augmented_fold_{fold}.xlsx')



