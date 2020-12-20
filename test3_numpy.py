from typing import List, Dict
from pandas import DataFrame
from numpy import ndarray
#import cudf
import pandas as cudf
import numpy as np # linear algebra
#a = np.array([ [ [111,112,113,114] , [121,122,123,124] ], [ [211,212,213,214], [221,222,223,224] ] ])
#print(a,a.shape)
"""a = np.array(
    [ #1
''' e1'''
        [ #2
        [11] , # 3
        [12]
        ],
''' e2'''
        [
        [21],
        [22]
        ]
    ])"""

nRowsRead = None # specify 'None' if want to read whole file
# test_log.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
ori_test = cudf.read_csv('D:\\zyh\\_workspace\\dropout_prediction\\prediction_log\\test_log.csv', delimiter=',', nrows = nRowsRead,index_col = None)
ori_test.dataframeName = 'test_log.csv'
nRow, nCol = ori_test.shape
print(f'There are {nRow} rows and {nCol} columns')
np_log = ori_test.values
np_log = np_log[:,[0,4,6]]
enroll_id_set = set(np_log[:,0])
enroll_log_dict = {}
c = 0
for i in enroll_id_set:
    mask = np_log[:,0]==i
    log_for_one = np_log[mask][:,[1,2]]
    enroll_log_dict[i] = log_for_one
    c+=1
    if c%1000 == 0 :
        print(c)

# 可以跑 但是只有16000个注册号 有点奇怪
# 用json导出 array 要先 .tolist() 读取的时候直接np.array()
def to_list(dict_):
    # 早上写 
    pass

enroll_log_dict = to_list(enroll_log_dict)
json.dump(enroll_log_dict,open(path,'w'))
d = json.load(open(path))