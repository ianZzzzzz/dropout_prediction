
import numpy as np
import tensorflow as tf
import os

SEQUENCE_LENGTH = 100
TRAIN_EPOCH=10
BATCH_SIZE = 1


def load__cut_merge_seqences(json_export_path):
    from typing import List, Dict
    from numpy import ndarray 
    import numpy as np
        # load
    def read_or_write_json(
        path:str
        ,mode:str
        ,log=None):
        ''' 读写json文件
            
            mode== 'r' -> read  return dict (enroll_id : log_list)

            mode=='w' -> write return None
        '''
        import json

        def w(__log,__path):
            if type(__log)!=type({}):
                print('ERROR : input data not a dict!')
            else:
                json.dump(__log,open(__path,'w'))
                print('SUCCESS WRITE , path ：', __path)
            return None

        def r(__log,__path)->Dict[int,list]:
            _dict = json.load(open(__path,'r'))
            return _dict

        return eval(mode)(log,path)  

    def dict_to_array(dict_log:dict,drop_key = False)->list:

        ''' 
            函数功能:
                将dict格式数据集转换为list格式数据集
            
            return：list ( [ 
                        [log_list_1,enroll_id_1],
                        [log_list_2,enroll_id_2],
                        [log_list_n,enroll_id_n] ])

                    -1位置为dict数据集的key —>enroll_id 

            note:   用list append执行很快 np.concatenate慢十倍以上'''
        i = 0
        print_key = 100000
        len_ = len(dict_log)

        dataset = []
        if drop_key == False:
            print('Inclouding enroll id in [-1] position.')

            for k,v in dict_log.items():
                
                dataset.append(v)
                
                i+=1
        else:
            print(' Series data only.')
            
            for k,v in dict_log.items():
                v = v[:-1]
                dataset.append(v)

                i+=1

       # print('Append finsih , dataset include ',len(dataset),' samples')

        return dataset

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
        print('SUCCESS cut useless samples ,',len(uesful_series),' remain .')
        return uesful_series

            
    # load
    enroll_dict_list_inside = read_or_write_json(
        path    = json_export_path
        ,mode   = 'r')
    non_id_array = dict_to_array(enroll_dict_list_inside,drop_key= True)

    # wash
    useful_list = cut_toolong_tooshort(non_id_array,up = 5000,down = 100)

    array = useful_list
    f = []

    for list_ in array :


        len_ = len(list_)
        chunk_len = SEQUENCE_LENGTH+1

        sample_per_list = len_//chunk_len
        last_Sample_location_in_list = chunk_len * sample_per_list

        for element in list_[:last_Sample_location_in_list] :
            f.append(element)

    text = np.array(f,dtype = 'str').tolist()

    return text

def prepare_dataset(text): 
    
    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        
        return input_text, target_text

    
    vocab = sorted(set(text)) # 词表

    char_to_index_dict  = {u:i for i, u in enumerate(vocab)}
    index_to_char_array = np.array(vocab)

    # text to vector
    text_as_int = np.array([char_to_index_dict[c] for c in text])
    # convert to tf type dataset 创建训练样本 / 目标
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)


    # 将连续的样本 切分为长度是SEQUENCE_LENGTH +1的片段
    sequences = char_dataset.batch(
        SEQUENCE_LENGTH+1
        , drop_remainder=True) # 样本长度如果不能被SEQUENCE_LENGTH整除 则舍掉最后一批
    # 从 SEQUENCE_LENGTH +1的片段中 
    # 选择 list[:-1]为input ， list[1:]为target
    dataset = sequences.map(split_input_target)


    # shuffle  tf.shuffle 详情见: https://zhuanlan.zhihu.com/p/42417456
    
    BUFFER_SIZE = 10
    dataset = dataset.shuffle(BUFFER_SIZE).batch(
        BATCH_SIZE
        , drop_remainder=True)


    return dataset

def build_model():
        # 词嵌入的维度
        embedding_dim = int(4)
        # RNN 的单元数量
        rnn_units = int(32)
        vocab_size = VOCAB_SIZE # 定义模型输入与输出层需要的参数
        batch_size = BATCH_SIZE
        
        
        
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(
                    vocab_size, 
                    embedding_dim,
                    batch_input_shape=[batch_size, None]),
                tf.keras.layers.GRU(
                    rnn_units,
                    return_sequences=True,
                    stateful=True,
                    recurrent_initializer='glorot_uniform'),
                tf.keras.layers.Dense(vocab_size)
            ])
        return model

def set_model():

    # 实例化
    model = build_model()

    def loss(labels, logits):
        """set loss function

        Args:
            labels ([labels]): [description]
            logits ([logits]): [description]

        Returns:
            [loss]: [sparse_categorical_crossentropy]
        """    
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels
            , logits
            , from_logits=True)
        
        return loss
    model.compile(optimizer='adam', loss=loss)

    return model

def generating_text(model, start_string,vocab):
    """[评估步骤（用学习过的模型生成文本）]

    Args:
        model 
        start_string

    Returns:
        [type]: [description]
    """  

    
    # instantiation
    model = build_model()
    # load weights from check-point
    model.load_weights(
        tf.train.latest_checkpoint(checkpoint_dir))


    model.build(tf.TensorShape([1, None]))



    num_generate = GEN_LENGTH # 要生成的字符个数
    
    
    

    char_to_index_dict  = {u:i for i, u in enumerate(vocab)}
    index_to_char_array = np.array(vocab)



    input_eval = [char_to_index_dict[s] for s in start_string] # 将起始字符串转换为数字（向量化）
    input_eval = tf.expand_dims(input_eval, 0)

    
    text_generated = [] # 空字符串用于存储结果
    
    temperature = 1.0
        # 低温度会生成更可预测的文本
        # 较高温度会生成更令人惊讶的文本
        # 可以通过试验以找到最好的设定

    model.reset_states()

    # 循环生成
    for i in range(num_generate):
        
        predictions = model(input_eval)
        # 删除批次的维度
        predictions = tf.squeeze(predictions, 0)

        # 用分类分布预测模型返回的字符
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(
            predictions
            , num_samples=1
            )[-1,0].numpy()

        # 把预测字符和前面的隐藏状态一起传递给模型作为下一个输入
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(index_to_char_array[predicted_id])
        #print('text_generated : ',text_generated)
    return text_generated

def action_counter(log_list:list):
    '''
        计算单词频次
        return : re = {
            'type':action_type_counter ,
            'action':action_counter}

        action_type_counter = { '1':0,  '2':0, '3':0, '4':0 }

        action_counter = {
            11:0,12:0,13:0,14:0,15:0       # video
            ,21:0,22:0,23:0,24:0,25:0,26:0  # problem
            ,31:0,32:0,33:0,34:0            # common
            ,41:0,42:0,43:0,44:0,45:0,46:0  # click
            } 
    '''


    action_counter = {
        '11':0,'12':0,'13':0,'14':0,'15':0       # video
        ,'21':0,'22':0,'23':0,'24':0,'25':0,'26':0  # problem
        ,'31':0,'32':0,'33':0,'34':0            # common
        ,'41':0,'42':0,'43':0,'44':0,'45':0,'46':0  # click
        }
    for i in range(len(log_list)):

        action_ = log_list[i]
       # action_type = action_[0]

       # action_type_counter[action_type] +=1
        action_counter[action_] +=1

    #re = {'type':action_type_counter ,'action':action_counter}
    re = action_counter
    return re

def action_type_counter(log_list:list):
    '''
        计算单词频次
        return : re = {
            'type':action_type_counter ,
            'action':action_counter}

        action_type_counter = { '1':0,  '2':0, '3':0, '4':0 }

        action_counter = {
            11:0,12:0,13:0,14:0,15:0       # video
            ,21:0,22:0,23:0,24:0,25:0,26:0  # problem
            ,31:0,32:0,33:0,34:0            # common
            ,41:0,42:0,43:0,44:0,45:0,46:0  # click
            } 
    '''

    action_type_counter = { '1':0,  '2':0, '3':0, '4':0 }
   
    for i in range(len(log_list)):

        action_ = log_list[i]
        action_type = action_[0]

        action_type_counter[action_type] +=1
       # action_counter[action_] +=1

    #re = {'type':action_type_counter ,'action':action_counter}
    re = action_type_counter
    return re


json_export_path = 'Piplines\\mid_export_enroll_dict.json'
text_full = load__cut_merge_seqences(json_export_path)
text = text_full[:10000]
# test 


vocab = sorted(set(text)) # 词表
VOCAB_SIZE = len(vocab)

dataset = prepare_dataset(text)
model = set_model() 


# set check-point para
checkpoint_dir = './training_checkpoints'
import os
checkpoint_prefix = os.path.join( checkpoint_dir, "ckpt_{epoch}" )
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    
            filepath    =   checkpoint_prefix
    ,save_weights_only  =   True    )

# fit data


history = model.fit(
    dataset
    , epochs= 20 # TRAIN_EPOCH
    , callbacks =   [checkpoint_callback]   )

# save check-point
tf.train.latest_checkpoint(checkpoint_dir)

# generate 
GEN_LENGTH = 700
generate_text = generating_text(start_string=text_full[10000:10100]
    ,model= model 
    ,vocab= vocab)

action_counter(text_full[10100:10800])
action_counter(generate_text)



{'1': 531, '2': 0, '3': 0, '4': 169}   
>>> word_counter(generate_text)        
{'1': 533, '2': 0, '3': 1, '4': 166}   

{'1': 441, '2': 0, '3': 0, '4': 259}   
>>> word_counter(generate_text)        
{'1': 489, '2': 0, '3': 2, '4': 209}

