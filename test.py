# %%
from module.util import *
from class_Segmentation import Segmentation
from class_BertForClassification_Wrapper import Bert_Wrapper
# %%
'''
這邊所有句子可以共用這些 segmentation 和 predict 的 model，只要 load model 一次就好
'''
# Init model
seg = Segmentation(type="bert")
sentence_model_name = "sex_pred_sentence_epoch2_seed1234_1128"
s_bw = Bert_Wrapper(save_model_name=sentence_model_name)
factor1_model_name = "test"
f1_bw = Bert_Wrapper(save_model_name=factor1_model_name)
# %%
'''
這邊同一句話要做不同的分類(有利不利或是量刑因子，只要製作 dataloader 一次就好)
'''
# 包成 dataloader(用 s_bw 或是 f1_bw 這邊回傳的 dataloader 都一樣)
pred_sentence = seg.clean2seg("更遑論有何積極修復甲女損害之意願及具體作為存在，從而可認被告犯後態度甚差")
pred_df = s_bw._create_cleaned_dataframe(X=[pred_sentence], y=[0])
predset = s_bw._create_dataset(pred_df['X'], pred_df['y'], type="pred")
predloader = DataLoader(predset, batch_size=256, 
                        collate_fn=create_mini_batch)
# %%
# Predict
s_predictions = s_bw.predict(predloader)
f1_predictions = f1_bw.predict(predloader)
# %%
# Output 結果
print("不利:", s_predictions[:, 0])
print("有利:", s_predictions[:, 1])
print("中性:", s_predictions[:, 2])

print("是這個量型因子", f1_predictions[:, 1])
print("不是這個量型因子:", f1_predictions[:, 0])


# %%
