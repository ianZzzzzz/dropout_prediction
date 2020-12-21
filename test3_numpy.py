from typing import List, Dict
from pandas import DataFrame
from numpy import ndarray
#import cudf
import json
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
def pd_to_np(pd_log:DataFrame)->Dict[int,ndarray]:
    np_log = pd_log.values
    np_log = np_log[:,[0,4,6]]
    enroll_id_set = set(np_log[:,0])
    enroll_log_dict = {}
    c = 0
    import time
    total_silce = 0
    total_dict = 0
    loop_start = time.clock()
    for i in enroll_id_set:

        mask = np_log[:,0]==i
        slice_s = time.clock() #
        log_for_one = np_log[mask][:,[1,2]]
        slice_e = time.clock() #
        slice_g = slice_e - slice_s##
        enroll_log_dict[i] = log_for_one.tolist()
        dict_e = time.clock() #
        dict_g = dict_e -slice_e##
        total_silce = total_silce + slice_g
        totao_dict = total_dict +dict_g
        c+=1
        if c%1000 == 0 :
            print('loop count : ',c,
                'slice time : ',total_silce,
                'dict time : ',total_dict)
    loop_end = time.clock()
    loop_g = loop_end - loop_start
    print('loop time : ',loop_g)
# 可以跑 但是只有16000个注册号 有点奇怪 # 没问题 这只是整个文件的一部分 没有完全解压
# 用json导出 array 要先 .tolist() 读取的时候直接np.array()
def to_json(path,data: Dict[int,ndarray]):
    import json
    path = 'D:\\zyh\\data\\enroll_dict.txt'
    json.dump(enroll_log_dict,open(path,'w'))

def load_dict_list(path)-> Dict[int,ndarray]:
    import time
    import json
    l_s = time.clock()
    dict_log = json.load(open(path))
    s = time.clock()
    for i in dict_log:
        dict_log[i] = np.array(dict_log[i])
    e = time.clock()
    print('dict len : ',len(dict_log),'load time : ',s-l_s,'to array time : ',e-s)
    return dict_log

np_dict_log = load_dict_list(path)
def unique_count(df):
    print(df.dataframeName,'unique_count running!')
    for i in df.columns:
        object_ = df[i]
        print(i,' count:',object_.nunique())
unique_count(ori_test)
