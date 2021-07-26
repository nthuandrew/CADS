from module.util import *
from data.GV import *

class Data_Cleaner():
    def __init__(self):
        return

    def nlp_custody(self, df, output_dir, keep):
        info_dict = {}
        for k, v in output_dir.items():
            if(os.path.exists(self.fname)):
                pass
        
        return 
    
    def test(self, df):
        df = sorted(df)
        
        return df



if __name__=='__main__':
    bar = progressbar.ProgressBar()
    time.sleep(0.01)
    for df0 in bar(dfs):
        clean = Data_Cleaner()
        df = clean.load_data(df0)
    time.sleep(0.01)
