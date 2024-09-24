import streamlit as st
from load_css import local_css
import pandas as pd
from module.util import *
from src.class_BertForClassification_Pipeline import Pipeline as PL
from src.article_predict import *
from transformers import BertTokenizer, BertConfig
import csv
local_css("style.css")

sentence_model_path = 'sentence_model/four_class_augmented_0526_v2.pkl'
article_model_path = 'article_model/0819_400limit_3epoch_binary_short_5.pkl'
pretrained_model_name = 'hfl/chinese-bert-wwm-ext'
label_column_list = ['中性句','自殺與憂鬱', '自殺行為', '其他類型']
article_label_list = ['無危機文章', '高危機文章']
seed = 1234

@st.cache(allow_output_mutation=True)
def load_sentence_model(model_path):
    sentence_model = PL(pretrained_model_name=pretrained_model_name, load_model_path=model_path, \
        num_labels = len(label_column_list), seed = seed)
    sentence_model.initialize_training()
    return sentence_model

pl = load_sentence_model(sentence_model_path)

@st.cache(allow_output_mutation=True)
def load_article_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = BertConfig.from_pretrained(pretrained_model_name)
    config.max_length = 512
    model = MyModel(config, num_labels=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
article_model = load_article_model(article_model_path)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# config = BertConfig.from_pretrained(pretrained_model_name)
# config.max_length = 512
# article_model = MyModel(config, num_labels=2).to(device)
# article_model.load_state_dict(torch.load(article_model_path, map_location='cpu'))
# article_model.eval()


Title = "範例文章"
TextIDs = '81703ac14e'
TextTimes = '2021-09-08 17:44:13'
Authors = '匿名/男'

# original_contents = excel_data_df1['Content(remove_tag)'][:].tolist()
content = '''妳總是反覆的給我希望
而我一次次的害怕失望
總有一天會把力氣用光
卻始終看不見妳我相望

我會覺得很可惜 在這種時候又遇見
每次都是這樣 又以相同的方式收場
如果哪一天我突然消失了
就讓我好好休息一會兒吧
我累了 這樣的我們'''

sample_sentences = [sent for sent in split_sentence(content) if sent not in ['', ' ', '。', '，', '\n']]
predict_labels = [0, 3, 3, 3, 0, 0, 1, 0, 1]
# sentences = excel_data_df2['Sentence'][:].tolist()
# labels = excel_data_df2['標註代碼'][:].tolist()


# article = st.selectbox('Choose an article', Titles)
# key_i = int(excel_data_df3["index"][0])


title_col1, title_col2, title_col3 = st.columns([18,2,2])

with title_col1:
    st.title("網路危機訊息偵測系統")

with title_col2:
    st.markdown("\n")    
    st.markdown("\n")
    button_previous = st.button("上一篇")
    # if(button_previous):
    #     if key_i > 0:
    #         key_i -= 1
    #         article = Titles[key_i]
    #         button = False
with title_col3:
    st.markdown("\n")    
    st.markdown("\n")
    button_next = st.button("下一篇")

to_show = 0
category = ""


    

col2, col1 = st.columns([30, 30])
with col1:
    st.header("原始網路文章")
    input_text = st.text_area('請輸入欲預測的文章',content, height=300)
    sample_sentences = [sent for sent in split_sentence(input_text) if sent not in ['', ' ', '。', '，', '\n']]
    paragraphs = ['','','','']
    pred_loader = pl.prepare_dataloader(sample_sentences, for_prediction=True, not_df=True)
    predicts = pl.get_predictions(pl.model, pred_loader)
    print("文句分類完成")
    predict_labels = predicts
    predict_labels_not_tensor = [label_column_list[int(j)] for j in predict_labels]
    data = {}
    data['Sentence'] = sample_sentences
    data['label'] = predict_labels_not_tensor
    with open('sentence_classification.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # 寫入 header (也就是 key)
        writer.writerow(data.keys())

        # 寫入每一行資料
        for row in zip(*data.values()):
            writer.writerow(row)

    for idx, pred in enumerate(predicts):
        paragraphs[pred] += '_' + sample_sentences[idx]
    paragraphs = np.array([paragraphs])
    test_dataset = TextClassificationDataset(paragraphs, [[0.0, 0.0]], tokenizer, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    for batch_data in tqdm(test_dataloader):
        tokens_tensors, segments_tensors, masks_tensors,labels  = [t.to(device) for t in batch_data]
        with torch.no_grad(): 
            loss, logits = article_model(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors,labels=labels)
        # print(loss)
            prob = logits.data
            print(prob)
            _, crisis_level = torch.max(prob, 1)
    print("文章分類完成")
    if crisis_level == 1:
        print("YES")
        category = "A (危機程度高：有自傷自殺行為)"
    else:
        category = "0 (無危機狀況)"
    level = "<div><span class='category'>" + "危機程度: " + category + "</span></div>" 

with st.sidebar:
    title = "文章標題: " + Title
    level = "<div><span class='category'>" + "危機程度: " + category + "</span></div>" 
    textid = "<div><span class='highlight'>" + "文章ID: " + str(TextIDs) + "</span></div>" 
    texttime = "<div><span class='highlight'>" + "發佈時間: " + str(TextTimes) + "</span></div>" 
    author = "文章作者: " + Authors
    st.header(title)
    st.markdown(level, unsafe_allow_html=1)
    st.markdown(textid, unsafe_allow_html=1)
    st.markdown(texttime, unsafe_allow_html=1)
    st.markdown(author)
    # to_show = st.checkbox('是否呈現標註結果分類文字')
    st.subheader("標註分類與統計： ")
    container_total = st.container()
    for i in range(7):
        if i == 0:
            container0 = st.container()
        elif i == 1:
            container1 = st.container()
        elif i == 2:
            container2 = st.container()
        elif i == 3:
            container3 = st.container()
        elif i == 4:
            container4 = st.container() 
        elif i == 5:
            container5 = st.container()
        else:
            container6 = st.container()
            for j in range(3):
                st.markdown("\n")     
with col2:
    st.header("標註後網路文章")
    now_sentences = sample_sentences
    now_labels = predict_labels
    # print(len(now_sentences))
    tem_stastic = [0]*4
    for i in range(len(now_sentences)):
        lab = now_labels[i]
        sen = str(now_sentences[i])
        if lab == 0:
            s = "<div><span class='highlight zero'>" + sen + "</span></div>"
            st.markdown(s, unsafe_allow_html=1)
            if to_show:
                st.caption("中性句")
            tem_stastic[0] += 1 
        elif lab == 1:
            s = "<div><span class='highlight one'>" + sen + "</span></div>"
            st.markdown(s, unsafe_allow_html=1)
            if to_show:
                st.caption("自殺與憂鬱(認知或情緒)")
            tem_stastic[1] += 1  
        elif lab == 2:
            s = "<div><span class='highlight six'>" + sen + "</span></div>"
            st.markdown(s, unsafe_allow_html=1)
            if to_show:
                st.caption("自殺行為(行為)")
            tem_stastic[2] += 1
        else:
            s = "<div><span class='highlight three'>" + sen + "</span></div>"
            st.markdown(s, unsafe_allow_html=1)
            if to_show:
                st.caption("其他類型")
            tem_stastic[3] += 1
    total = str(sum(tem_stastic))
    s0 = "<span class='highlight zero'>" + "中性句‧‧‧‧‧‧‧‧‧‧‧‧‧‧‧‧‧‧‧‧‧‧‧‧‧‧‧‧" + "<span class='highlight bold'>" + str(tem_stastic[0]) + "</span>" + "句(" + str(round(100*tem_stastic[0]/sum(tem_stastic))) + "%)</span></div>"
    s1 = "<span class='highlight six'>" + "自殺行為(行為)‧‧‧‧‧‧‧‧‧‧‧‧‧‧‧‧‧" + "<span class='highlight bold'>" + str(tem_stastic[2]) + "</span>" + "句(" + str(round(100*tem_stastic[2]/sum(tem_stastic))) + "%)</span></div>"
    s2 = "<span class='highlight one'>" + "自殺與憂鬱(認知或情緒)‧‧‧‧" + "<span class='highlight bold'>" + str(tem_stastic[1]) + "</span>" + "句(" + str(round(100*tem_stastic[1]/sum(tem_stastic))) + "%)</span></div>"    
    s3 = "<span class='highlight three'>" + "其他類型‧‧‧‧‧‧‧‧‧‧‧‧‧‧‧‧‧‧‧‧‧‧‧‧‧" + "<span class='highlight bold'>" + str(tem_stastic[3]) + "</span>" + "句(" + str(round(100*tem_stastic[3]/sum(tem_stastic))) + "%)</span></div>"
    s_total = "總句數： " + total + "句"
    container_total.text(s_total)
    container0.caption(s0, unsafe_allow_html=1)
    container1.caption(s1, unsafe_allow_html=1)
    container2.caption(s2, unsafe_allow_html=1)
    container3.caption(s3, unsafe_allow_html=1)

