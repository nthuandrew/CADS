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

ex: /srv/model/version_1.0

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
--mode: train_multi_class or train_yes_or_no_class。選擇要訓練多分類，或是量刑因子的量刑因子分類的“是/不是"該量刑因子問題。
```

```
--train_data: 輸入 training data 的路徑。
```

```
--factors: 輸入要分類的 label 名稱。例如:['有利', '不利', '中性']
```

```
--neutral_data: 輸入中性句 neutral data 的路徑。
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
python main_bert_all.py --mode=train_multi_class --train_data="./data/raw/train.csv" --factors=['有利', '不利', '中性'] --version=2.0 --epoch=2 --pooling_strategy=reduce_mean
```

```

```
