
def find_avg_length_of_series(log_list: list)->list:
    len_array = np.zeros( len(log_list)+1,dtype = np.uint32)
    for i in range(len(log_list)):
        length =  len(log_list[i])
        len_array[i] = length

    series = len_array
    mean_ = np.mean(series)
    return mean_

def cut_toolong_tooshort(
    log_list: list
    ,up:int
    ,down:int
    )-> list:
    '''
       本函数根据设定的上下限 返回长度在上下限之间的序列构成的list
       
    '''

    uesful_series = []
    useless_series = []
    for series in log_list:
        length = len(series)
        
        if (length>down)and(length<up):
            uesful_series.append(series)
        else: 
            useless_series.append(series)
    
    return uesful_series

avg_series_len = find_avg_length_of_series(log_list)




def call_fun_by_str(fun_str):
    def function2():
        print("function2")

    def function3():
        print("function3")
    eval(fun_str)()


call_fun_by_str("function2")

def read_or_write_json(
    path:str
    ,mode:str
    ,_log = None):
    ''' 读写json文件
        mode控制 read or write
    '''
    import json

    def w(__log,__path):
        if type(__log[1])!=type([]):
            # json不支持ndarray
            # 用json导出 array 要先 .tolist() 读取的时候直接np.array()
            for i in range(len(_log)):
                __log[i] = __log[i].tolist()

        json.dump(__log,open(__path,'w'))
        return None

    def r(__log,__path)->list:
        _list = json.load(open(__path,'r'))
        return _list

    return eval(mode)(_log,path)


read_or_write_json(path = 'a.json',mode = 'w',_log = t)
test = read_or_write_json(path = 'a.json',mode = 'r')