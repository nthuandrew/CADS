# CADS
## Sentence Model Flow
### four_class.ipynb -> train_multiclass.sh. -> test_multiclass.sh<br>
Note that now test_multiclass.sh is only for evaluation, not prediction.
For evaluation, comment out the #new and #end of new part in main_sentence.py(115~127) and class_BertForClassification_Pipeline.prepare_dataloader(230~234).
Now, for sentence model, "正向文字" is combined into "無標註"
## Article Model Flow
### article_process.ipynb, (split_A2.py->A2_process.ipynb) -> article_augmentation.ipynb -> short_sentence.ipynb -> main_article.py -> main_article_test.py<br>

## Data Discription

## Sentence Model Discription

## Article Model Discription

