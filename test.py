# %%
from module.util import *
from class_Segmentation import Segmentation
from class_BertForClassification_Wrapper import Bert_Wrapper

# %%

# %%
'''
這邊所有句子可以共用這些 segmentation 和 predict 的 model，只要 load model 一次就好
'''
# Init model
seg = Segmentation(type="bert")
sentence_model_name = "sex_train_sentence_epoch4_seed1234_2022-02-14"
s_bw = Bert_Wrapper(model_dir="/data/model/criminal/version_2.1", save_model_name=sentence_model_name, device="cpu")
factor1_model_name = "criminal/version_2.1/sex_train_factor_被害人的態度_epoch2_seed1234_2022-02-14"
f1_bw = Bert_Wrapper(model_dir="/data/model/criminal/version_2.1", save_model_name=factor1_model_name, device="cpu")
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

print("是這個量型因子", f1_predictions[:, 0])
print("不是這個量型因子:", f1_predictions[:, 1])


# %%
# 補救計畫(重新存 model)
# version = "version_2.1"
# pooling = 'reduce_mean'
# for file in os.listdir(f"/data/model/criminal/{version}_old"):
#     new_file = file.replace('.pkl', '')
#     sentence_model_name = f"criminal/{version}_old/{new_file}"
#     print(sentence_model_name)
#     s_bw = Bert_Wrapper(save_model_name=sentence_model_name, device="cpu")
#     model = s_bw.model
#     info_dict = s_bw.info_dict
#     # 需要視情況修改
#     # cls, reduce_mean
#     info_dict['hyper_param']['POOLING_STRATEGY'] = pooling
#     target = [info_dict, model.state_dict()]
#     print(f"/data/model/criminal/{version}/{file}")
#     torch.save(target, f"/data/model/criminal/{version}/{file}")
    
