# NLP Law model: 
model description...

#### Code Authors
Chen-Zhi Su, ...


### Folders & Files: 
* Modules: "./modules"  classes and functions.
* Log: "./log"
* Data:
    * Original data: "./data/raw"
    * Cleaned data: "./data/cleaned"
    * Training result: "./data/result"
    * Predict result: "./data/pred"
    * ML model: "./data/model"
    * Plots: "./data/plot"
    * Global variable: "./data/GV.py"  dicts, lists, and other global variables.



### Environment Setting:
* python=??
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
