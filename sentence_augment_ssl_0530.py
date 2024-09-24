
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
        # print(i.columns)


    # In[60]:


    # split Augmentation Data
    all_middle_df = labelize_df[0]
    print(all_middle_df.shape)
    to_add = [4000, 6000]
    Augment_df = []
    for i in range(2):
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

    ### new block at 11/13!!! Goal: take A2 sentence into training data
    df_A2 = pd.read_excel('data/article/split_output_A2_v2_answer.xlsx')
    df_A2 = df_A2[target_columns]
    # DF = pd.concat([df, df_A2], axis=0, ignore_index=True)
    all_labelize_df_A2 = []
    for label in new_column:
        all_labelize_df_A2.append(df_A2[df_A2[label] == 1].loc[:])
    for i in all_labelize_df_A2:
        print(len(i))

    # ### new block at 11/13!!! Goal: combine A1 training data and A2 training data
    for i in range(len(labelize_df)):
        labelize_df[i] = pd.concat([labelize_df[i], all_labelize_df_A2[i]], axis=0, ignore_index=True)
        print(labelize_df[i].shape)

   
    to_train = [17000, 0, 0, len(labelize_df[3])]
    target_num = 16000
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


    # In[67]:



    augmented_df = pd.concat(training_df,axis=0, ignore_index=True)
    augmented_df.to_excel(f'data/final/sentence/four_class/seed_{seed}/four_class_0530_train_augmented_ssl_fold_{fold}.xlsx')



