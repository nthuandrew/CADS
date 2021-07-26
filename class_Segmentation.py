from module.util import *
from data.GV import *

class Segmentation():
    def __init__(self, df):
        self.df=df
        return

    def run(self, language):
        df = self.df
        info_dict = {}
        for k, v in output_dir.items():
            if(os.path.exists(self.fname)):
                pass
        
        return df
    
    def run_bert(self, language):
        
        return



if __name__=='__main__':
    bar = progressbar.ProgressBar()
    time.sleep(0.01)
    for df0 in bar(dfs):
        seg = Segmentation(df0)
        df = seg.run(language='chinese')
    time.sleep(0.01)
