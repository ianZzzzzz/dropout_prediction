
import numpy as np
import tensorflow as tf
import os

SEQUENCE_LENGTH = 10
TRAIN_EPOCH=10
BATCH_SIZE = 1

def get_text():
    """[将数据转换为字符串list]

    Returns:
       text [list]
    """    
    array = [
        43, 41, 42, 15, 12, 13, 13, 12, 13, 12, 13, 11, 13, 12, 13, 12, 13, 12, 13, 12, 11, 15, 11, 13, 12, 11, 11, 11, 11, 
        12, 12, 13, 46, 42, 15, 13, 12, 12, 13, 12, 12, 13, 46, 42, 15, 15, 43, 41, 44, 42, 46, 42, 15, 13, 13, 13, 11, 11, 15, 11, 12, 13, 13, 12, 13, 12, 11, 13, 
        46, 42, 15, 12, 13, 13, 13, 12, 13, 12, 13, 12, 15, 11, 13, 12, 12, 13, 13, 46, 42, 15, 12, 13, 12, 13, 12, 13, 12, 11, 12, 11]
    import numpy as np
    text = np.array(array,dtype = 'str').tolist()
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
    # 选择 list[:-1]为input ， lists1:]为target
    dataset = sequences.map(split_input_target)


    # shuffle  tf.shuffle 详情见: https://zhuanlan.zhihu.com/p/42417456
    
    BUFFER_SIZE = 10
    dataset = dataset.shuffle(BUFFER_SIZE).batch(
        BATCH_SIZE
        , drop_remainder=True)


    return dataset

def set_model():

    # 词嵌入的维度
    embedding_dim = int(4)
    # RNN 的单元数量
    rnn_units = int(32)
    vocab_size = len(set(text)) # 定义模型输入与输出层需要的参数

    def build_model(
        vocab_size, 
        embedding_dim, 
        rnn_units, 
        batch_size):
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
    # 实例化
    model = build_model(
        vocab_size = vocab_size,
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=BATCH_SIZE)

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

def generate_text(model, start_string):
    """[评估步骤（用学习过的模型生成文本）]

    Args:
        model 
        start_string

    Returns:
        [type]: [description]
    """  

    
    # instantiation
    model = build_model(
        vocab_size
        , embedding_dim
        , rnn_units
        , batch_size = BATCH_SIZE)
    # load weights from check-point
    model.load_weights(
        tf.train.latest_checkpoint(checkpoint_dir))


    model.build(tf.TensorShape([1, None]))



    num_generate = GEN_LENGTH # 要生成的字符个数

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
        
    return text_generated

def word_counter(log_list:list):
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
    action_counter = {
        '11':0,'12':0,'13':0,'14':0,'15':0       # video
        ,'21':0,'22':0,'23':0,'24':0,'25':0,'26':0  # problem
        ,'31':0,'32':0,'33':0,'34':0            # common
        ,'41':0,'42':0,'43':0,'44':0,'45':0,'46':0  # click
        }
    for i in range(len(log_list)):

        action_ = log_list[i]
        action_type = action_[0]

        action_type_counter[action_type] +=1
        action_counter[action_] +=1

    #re = {'type':action_type_counter ,'action':action_counter}
    re = action_counter
    return re


text = get_text()
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
    , epochs=   TRAIN_EPOCH
    , callbacks =   [checkpoint_callback]   )
# save check-point
tf.train.latest_checkpoint(checkpoint_dir)



GEN_LENGTH = 50
gen_text = generate_text(model, start_string=text[:50])


word_counter(text)
word_counter(text[:100])
word_counter(text[50:100])
word_counter(gen_text)
