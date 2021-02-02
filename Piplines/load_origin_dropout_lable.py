TEST_OR_NOT = False
from numpy import ndarray
def load(
    log_path: str,
    encoding_='utf-8',
    columns=None,
    test=TEST_OR_NOT,
    chunk_size = None
    )-> ndarray or DataFrame:
    '''读取csv文件 返回numpy数组'''

    import pandas as pd
    if test ==True: # only read 10000rows 
        reader = pd.read_csv(
            log_path
            ,encoding=encoding_
            ,names=columns
            ,chunksize=chunk_size)
            
        for chunk in reader:
            # use chunk_size to choose the size of test rows instead of loop
            log = chunk
            return log.values

    else: # read full file
        log = pd.read_csv(
            log_path
            ,encoding=encoding_
            ,names=columns)
        
    print('load running!')
    
    return log.values

path = 'prediction_log\\test_truth.csv'
labels = load(path)














