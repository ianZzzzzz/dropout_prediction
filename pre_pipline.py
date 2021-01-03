'''
    进展 ：
      读取csv数据对其归类存字典 转换成训练数据 
      特征包含：
        1 不同action 的组合模式与先后模式
        2 action之间的间隔时间
        3 用户主要的操作分布在开课时间的哪一部分

     考虑后决定暂不使用特征3 因为使用完整的序列会使得训练样本太稀疏 导致训练缓慢

     编码：
        怎么选择合适的编码将22种符号特征编码成向量 
        要求近似的行为编码的余弦相似度等指标也接近

        onehot显然不合适 选择词嵌入方式训练 
        AAAI19论文的代码中将行为分成了四类，

'''
'''程序功能：

        读取通信流量数据日志文件
        对时间与日期列进行拼接 
        以第一行数据为起始时间 
        以秒为单位计算距离起始时间的距离 作为时间序列索引
        对以GB为单位的数据*1024转为MB单位
        以字典方式存储各个区域下的日志'''
# preprocess of "log file to time series"
import pandas as pd
# import cudf as pd # nvidia GPU only # !pip install cudf 
from typing import List, Dict
from numpy import ndarray
from numpy import datetime64
from pandas import DataFrame
import numpy as np
print_batch = int(10000)
def _t(function):
    from functools import wraps
    import time
    '''
    用装饰器实现函数计时
    :param function: 需要计时的函数
    :return: None
    '''
    @wraps(function)
    def function_timer(*args, **kwargs):
        print ('[Function: {name} start...]'.format(name = function.__name__))
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ('[Function: {name} finished, spent time: {time:.2f}s]'.format(name = function.__name__,time = t1 - t0))
        return result
    return function_timer
@_t
def load(
    log_path: str,
    read_mode: str,
    return_mode: str,
    encoding_='utf-8',
    columns=None
    )-> ndarray or DataFrame:
    '''读取csv文件 返回numpy数组'''
    #if read_mode == 'cudf':import cudf as pd
    if read_mode == 'pandas' :
        import pandas as pd
        log = pd.read_csv(log_path,encoding=encoding_,names=columns,)
        print('load running!')
    if return_mode == 'df':return log
    if return_mode == 'values':return log.values

@_t
def convert(
    log: dict )->dict: 
    print('dict total len :',len(log))
    print(' convert running!')
    
    import numpy as np
    
    def find_start_end(c_id:str)->Dict[int,datetime64]:

        ''' 根据course_id 查询课程的总耗时秒数 以及开始时间并返回
            函数调用了全局变量C_INFO_NP必须在课程信息被加载后才能运行'''
        mask = C_INFO_NP[:,1] == c_id

        start = C_INFO_NP[mask][:,2]
        end   = C_INFO_NP[mask][:,3]
        #type: object ['2016-11-16 08:00:00']
        start = str(start)
        end = str(end)
        #type: str ['2016-11-16 08:00:00']
        start = start[2:-2]
        end = end[2:-2]
        #type: str '2016-11-16 08:00:00'
        try:
            end = np.datetime64(end) 
            start = np.datetime64(start)
            seconds_of_gap = int((end - start).item().total_seconds())
        except:print('ERROR start,end :',start,end)
        time_info = {
            'length': seconds_of_gap
            ,'head' : start}
        return time_info
    @_t
    def time_map(log_np:ndarray)->ndarray:
        def to_int(x):
            return int(x)
        md = map(to_int,log_np[:,1]) 
        __time = list(md) # time list

        __head = np.min(__time)
        __tail = np.max(__time)
        __length = __tail - __head +1
        action_series = np.zeros((__length,1),dtype='a32')

        for row in log_np:
            __t = int(row[1]) # time now
            __location = __t - __head
            action_series[__location] = row[0]
        return action_series

    new_dict = {}
    for e_id ,v in log.items():
        # 遍历字典 
        # type(v)==list
        v = np.array(v)
        c_id = v[0,2] 
        # 上一函数为了方便 保留了每一行的course_id 此处只需要一个值
        _log       = v[:,[0,1]]
        time_col   = _log[:,1]
        action_col = _log[:,0] 
        
        time_info   = find_start_end(c_id)
        time_head   = time_info['head']
        time_length = time_info['length']

        list_time   = np.zeros((len(_log),1),dtype = np.uint32)
        list_action = np.zeros((len(_log),1),dtype = 'a32') # a32 存32个英文字符
        

        for row_num in range(len(_log)):
            ''' 为了对action做进一步编码 保留了action接口 '''

            _row = _log[row_num,:]
            _time = _row[1]
            _action = _row[0]
            try:
                _time = np.datetime64(_time)
                _time =  int(
                    ( _time - time_head ).item().total_seconds() )
                list_time[row_num] = _time

                list_action[row_num] = _action
            except:print('ERROR log time [_time] :',_time)
        rebulid = np.concatenate( (list_action, list_time), axis = 1)
        rebulid[:,1] = rebulid[:,1].astype('u4')

        '''出于保留 ‘用户主要的操作分布在开课时间的哪一部分’ 这一特征 
            的目的，将时间转换部分分为两部分写，后期如需重建此特征以上的代码可以不动'''

        action_series = time_map(rebulid)
        new_dict[e_id] =  action_series 
        
    return new_dict



'''此分类引用自AAAI19那篇论文的代码
    video_action = ['seek_video','play_video','pause_video','stop_video','load_video']
    problem_action = ['problem_get','problem_check','problem_save','reset_problem','problem_check_correct', 'problem_check_incorrect']
    forum_action = ['create_thread','create_comment','delete_thread','delete_comment']
    click_action = ['click_info','click_courseware','click_about','click_forum','click_progress']
    close_action = ['close_courseware']
'''
@_t
def cut_nan(
    log: ndarray,
    key_col:list
    )->ndarray:
    '''删除给定列中存在空值的日志行'''
    import numpy as np

    pandas_ = type(pd.DataFrame([]))
    numpy_  = type(pd.DataFrame([]).values)
        # type(type(pd.DataFrame([]))) == <class 'type'> not a str :)

    if type(log)== pandas_ :
        log = log.values
    if type(log)== numpy_:
        pass

    # nan row filter
    for i in key_col:
        mask = np.isnan(log[:,i]) # bug
        log = log[~mask]
    
    return log

@_t
def to_dict_2(log: ndarray)-> Dict[str,ndarray]: 
    '''优化版 时间复杂度低'''
    print('find ',len(log),' row logs')
    i = 0
    log_dict = {}
    for row in log:
        area = row[0]
        data = row[[1,2,3]]
        #print('area: ',area,' data: ',data)
        
        # if log_dict[]里没数据：初始化=[]
        try:
            log_dict[area].append(data)
        except:
            log_dict[area] = []
            log_dict[area].append(data)
            #print(log_dict[area])
        i+=1
        if (i%print_batch)==0:print('already dict : ',i,'row logs')
    
    return log_dict

@_t
def to_json(path,dict_log: Dict[int,ndarray]):
    '''dict[    enroll_id : array.tolist()]-->json_txt
        # json不支持ndarray
        # 用json导出 array 要先 .tolist() 读取的时候直接np.array()
    '''
    '''for k,v in dict_log.items():
        dict_log[k] = v.tolist()
'''
    import json
    json.dump(dict_log,open(path,'w'))

log_path = 'test\\prediction_log\\test_log.csv'
log_col = ['enroll_id','username','course_id','session_id','action','object','time']

log_np = load(
    log_path =log_path,
    read_mode ='pandas',
    return_mode = 'values',
    encoding_ = 'utf-8',
    columns =log_col)
log_dict = to_dict_2(log_np[1:,[0,4,6,2]]) # e_id action time c_id

c_info_path = 'test\\course_info.csv'
c_info_col = ['id','course_id','start','end','course_type','category']
C_INFO_NP = load(
    log_path =c_info_path,
    read_mode ='pandas',
    return_mode = 'values',
    encoding_ = 'utf-8',
    columns =c_info_col
    )

log_np_convert = convert(log_dict)

to_json(path = 'processed_1_log.json',dict_log=log_np_convert)

#dataset format
'''ds_log = np.zeros(
    (len(ts_dict),2),dtype = np.int32)'''
# BUG 存在问题 每个区域下的日志长度是不确定的  
def to_int(x):
    return int(x)
md = map(to_int,log_np[:,1]) 
__time = list(md) # time list

__head = np.min(__time)
__tail = np.max(__time)
__length = __tail - __head +1
action_series = np.zeros((__length,1),dtype='a32')

for row in log_np:
    __t = int(row[1]) # time now
    __location = __t - __head
    action_series[__location] = row[0]



x = np.array([[b'a',int(10)],  [b'v',3],  [b'c',0]])  
print (x)
print ('对 x 调用 argsort() 函数：')
y = np.argsort(x[:,1])  
print (y)
print ('\n')
print ('以排序后的顺序重构原数组：')
print (x[y])
print ('\n')
print ('使用循环重构原数组：')
for i in y:  
    print (x[i], end=" ")