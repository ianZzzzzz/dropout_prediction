
import tensorflow as tf

#
import os
import time

array = [
    43, 41, 42, 15, 12, 13, 13, 12, 13, 12, 13, 11, 13, 12, 13, 12, 13, 12, 13, 12, 11, 15, 11, 13, 12, 11, 11, 11, 11, 
    12, 12, 13, 46, 42, 15, 13, 12, 12, 13, 12, 12, 13, 46, 42, 15, 15, 43, 41, 44, 42, 46, 42, 15, 13, 13, 13, 11, 11, 15, 11, 12, 13, 13, 12, 13, 12, 11, 13, 
    46, 42, 15, 12, 13, 13, 13, 12, 13, 12, 13, 12, 15, 11, 13, 12, 12, 13, 13, 46, 42, 15, 12, 13, 12, 13, 12, 13, 12, 11, 12, 11]
import numpy as np
text = np.array(array,dtype = 'str').tolist()



vocab = sorted(set(text))
print ('{} unique characters'.format(len(vocab)))

char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

seq_length = 10
examples_per_epoch = len(text)//seq_length

# 创建训练样本 / 目标
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(1):
    print(idx2char[i.numpy()])

for i in char_dataset.take(5):
    print(index_to_char_array[i.numpy()])


# 设置 batch大小
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
# show 5 batch
for item in sequences.take(4):
  print(item.numpy(),len(item.numpy()))

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    
    return input_text, target_text

dataset = sequences.map(split_input_target)

#show a sample
for input_example, target_example in  dataset.take(1):
    print ('Input data: ', input_example.numpy())
    print ('Target data:', target_example.numpy())

# shuffle tf的shuffle 详情见: https://zhuanlan.zhihu.com/p/42417456
BATCH_SIZE = 1
BUFFER_SIZE = 10
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

vocab_size = len(vocab)
# 词嵌入的维度
embedding_dim = int(4)

# RNN 的单元数量
rnn_units = int(32)

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
    vocab_size = len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

# run test 
for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(
        example_batch_predictions.shape,
         "# (batch_size, sequence_length, vocab_size)")

model.summary()

sampled_indices = tf.random.categorical(
    example_batch_predictions[0], 
    num_samples=1)
sampled_indices = tf.squeeze(
    sampled_indices,
    axis=-1
    ).numpy()


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

# evaluate loss
example_batch_loss  = loss(target_example_batch, example_batch_predictions)

print(
    "Prediction shape: "
    , example_batch_predictions.shape
    , " # (batch_size, sequence_length, vocab_size)"    )
print(
    "scalar_loss:      "
    , example_batch_loss.numpy().mean() )


model.compile(optimizer='adam', loss=loss)

# save check point
checkpoint_dir = './training_checkpoints'
import os
checkpoint_prefix = os.path.join( checkpoint_dir, "ckpt_{epoch}" )

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    
            filepath    =   checkpoint_prefix
    ,save_weights_only  =   True    )


EPOCHS=10
# fit data
history = model.fit(
    dataset
    , epochs    =   EPOCHS
    , callbacks =   [checkpoint_callback]   )
# load check-point
tf.train.latest_checkpoint(checkpoint_dir)

# instantiation
model = build_model(
    vocab_size
    , embedding_dim
    , rnn_units
    , batch_size = 1)
# load weights from check-point
model.load_weights(
    tf.train.latest_checkpoint(checkpoint_dir))


model.build(tf.TensorShape([1, None]))

def generate_text(model, start_string):
    """[评估步骤（用学习过的模型生成文本）]

    Args:
        model 
        start_string

    Returns:
        [type]: [description]
    """  

    
    num_generate = GEN_LENGTH # 要生成的字符个数

    input_eval = [char2idx[s] for s in start_string] # 将起始字符串转换为数字（向量化）
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

        text_generated.append(idx2char[predicted_id])
        
    return text_generated
    #return (start_string + ''.join(text_generated))

GEN_LENGTH = 500
gen_text = generate_text(model, start_string=text[:100])


def word_counter(log_list:list)-> Dict[str,dict]:
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

word_counter(text)
word_counter(text[:100])
word_counter(text[100:600])
word_counter(gen_text)
