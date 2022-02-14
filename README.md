# NLP Law model:

model description...

#### Code Authors

Chen-Zhi Su, ...

### Folders & Files:

* Modules: "./modules"  classes and functions.
* Log: "./log"
* **Model: "/srv/model/"**
* Data:
  * Original data: "./data/raw"
  * Cleaned data: "./data/cleaned"
  * Training result: "./data/result"
  * Predict result: "./data/pred"
  * ML model: "/srv/model/"
  * Plots: "./data/plot"
  * Global variable: "./data/GV.py"  dicts, lists, and other global variables.

### Model Versions:

* **version_0.0**: 最原始的版本(使用舊資料)、pooling_strategy=cls, epoch=2, 自動擷取中性句取 3000句;人工標注中性句全取
* **version_1.0**: 使用新資料、pooling_strategy=cls, epoch=2(factor),4(sentence), 自動擷取中性句取 3000句;人工標注中性句全取
* **version_1.1**: 只有量刑因子的分類模型，有利不利中性句的分類模型可使用 version_1.0 的。使用新資料、pooling_strategy=cls, epoch=2(factor), 自動擷取中性句與人工擷取中性句數量總和為該量刑因子資料量之兩倍，且人工:自動擷取數量為 6:4 的分佈
* version_1.2:
* **version_2.0**: 使用新資料、pooling_strategy=reduce_mean, epoch=2(factor),4(sentence), 自動擷取中性句取 3000句;人工標注中性句全取
* **version_2.1**: 只有量刑因子的分類模型，有利不利中性句的分類模型可使用 version_2.0 的。使用新資料、pooling_strategy=reduce_mean, epoch=2(factor), 自動擷取中性句與人工擷取中性句數量總和為該量刑因子資料量之兩倍，且人工:自動擷取數量為 6:4 的分佈
* version_2.2:

### Environment Setting:

* python=3.8.10
* GPU version:??
* Install NVDIA GPU driver:??

```
pip install --user nvidia-pyindex
pip install --user nvidia-tensorflow[horovod]
```

* Install python packages:

```
pip install -r requirement.txt
```

* Use Makefile and requirements.txt to install packages:

```

```

### Modules:

Flow: Data_Cleaner --> Create_Feature --> Prepared_Data --> Classify

* main.py: the pipeline of the project. Create random sampling training result...
* Data_Cleaner::load_data(): turn swc skeletons into level trees...
* Create_Feature::load_data(): create Soma-features and Local-features...

### Usage:

```
--mode: train_sentence or train_factor。選擇要訓練不利/有利/中性句，或是量刑因子的分類。
```

```
--train_data: sex or gun or drug。選擇是三種 cases 的哪一種。
```

```
--neutral_data: 如果需要加入自動擷取的中性句，則加入這項，無則不需加入此項。
```

```
--version: 1.0 or 1.1 etc. 會在 /srv/model 建立對應版本號的 folder，train 好的 model 會被存在裡面。
```

```
--epoch: 要 train 幾個 epoch
```

```
--pooling_strategy: cls or reduce_mean。Bert 的最後一層的 pooling 策略可選擇使用 [cls] 的 token embedding 代表整個句子；或是將句子中每個 token 的 embedding 平均。
```

```
--batch_size: defalut 64
```

```
--seed: default 1234
```

```
--max_len: default 128。句子最長字數。
```

```
--lr: default 2e-5。learning rate。
```

* ex: Train 不利/有利/中性句：

```
python main_bert_criminal.py --mode=train_sentence --train_data=sex --version=2.0 --epoch=2 --pooling_strategy=reduce_mean
```

* Train 量刑因子:

```
python main_bert_criminal.py --mode=train_factor --train_data=sex --neutral_data --version=2.0 --epoch=2 --pooling_strategy=reduce_mean
```
