#%%
import tensorflow as tf
#%%
# 导入数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# RNN各种参数定义
lr = 0.001 #学习速率
training_iters = 100000 #循环次数
batch_size = 128
n_inputs = 17 #手写字的大小是28*28，这里是手写字中的每行28列的数值
n_steps = 2 #这里是手写字中28行的数据，因为以一行一行像素值处理的话，正好是28行
n_hidden_units = 128 #假设隐藏单元有128个
n_classes = 2# cluster number #因为我们的手写字是0-9，因此最后要分成10个类

# 定义输入和输出的placeholder
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# 对weights和biases初始值定义
weights = {
    # shape(28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # shape(128 , 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {
    # shape(128, )
    'in':tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # shape(10, )
    'out':tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

def RNN(X, weights, biases):
    # X在输入时是一批128个，每批中有28行，28列，因此其shape为(128, 28, 28)。为了能够进行 weights 的矩阵乘法，我们需要把输入数据转换成二维的数据(128*28, 28)
    X = tf.reshape(X, [-1, n_inputs])

    # 对输入数据根据权重和偏置进行计算, 其shape为(128batch * 28steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']

    # 矩阵计算完成之后，又要转换成3维的数据结构了，(128batch, 28steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell，使用LSTM，其中state_is_tuple用来指示相关的state是否是一个元组结构的，如果是元组结构的话，会在state中包含主线状态和分线状态
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # 初始化全0state
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    # 下面进行运算，我们使用dynamic rnn来进行运算。每一步的运算输出都会存储在outputs中，states中存储了主线状态和分线状态，因为我们前面指定了state_is_tuple=True
    # time_major用来指示关于时间序列的数据是否在输入数据中第一个维度中。在本例中，我们的时间序列数据位于第2维中，第一维的数据只是batch数据，因此要设置为False。
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

    # 计算结果，其中states[1]为分线state，也就是最后一个输出值
    results = tf.matmul(states[1], weights['out']) + biases['out']
    return results

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step*batch_size < training_iters:
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={x:batch_xs, y:batch_ys})
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={x:batch_xs, y:batch_ys}))
        step += 1
    
#%%

import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])
  plt.show()
BUFFER_SIZE = 10000
BATCH_SIZE = 64

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

#%%
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
def np_to_df():
    col = [
        'gender', 'birth_year', 'edu_degree', 
        'course_category', 'course_type', 'course_duration', 
        'student_amount', 'course_amount', 'dropout_rate_of_course', 'dropout_rate_of_user',
        'L_mean', 'L_var', 'L_skew', 'L_kurtosis', 
        'S_mean', 'S_var', 'S_skew', 'S_kurtosis',
        '11','12','13','14',
        '21','22','23','24',
        '31','32','33','34',
        '41','42','43','44','label'
        ]
    
    train_df = pd.read_csv(
        'list_data_train_simple_with_label.csv').iloc[:,1:]
    train_df.columns = col
    test_df = pd.read_csv(
        'list_data_test_simple_with_label.csv').iloc[:,1:] 
    test_df.columns = col
    return train_df,test_df
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('label')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

# %%
train,test = np_to_df()
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')
#%%
# 一种从 Pandas Dataframe 创建 tf.data 数据集的实用程序方法（utility method）

batch_size = 5 # 小批量大小用于演示
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

for feature_batch, label_batch in train_ds.take(1):
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of ages:', feature_batch['birth_year'])
  print('A batch of targets:', label_batch )

# %%
example_batch = next(iter(train_ds))[0]
# 用于创建一个特征列
# 并转换一批次数据的一个实用程序方法
def demo(feature_column):
  feature_layer = layers.DenseFeatures(feature_column)
  print(feature_layer(example_batch).numpy())
#%%
age = feature_column.numeric_column("birth_year")
age_buckets = feature_column.bucketized_column(
    age, 
    boundaries=[3,6,9])

# %%
course_category = feature_column.categorical_column_with_vocabulary_list(
      'course_category',
       [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18  
      ])

course_category_one_hot = feature_column.indicator_column(course_category)
demo(course_category_one_hot)
# %%
course_category_embedding = feature_column.embedding_column(
    course_category, dimension=2^5)
demo(course_category_embedding)
# %%
feature_columns = []

# 数值列
value_feat = [
    'birth_year', 'course_duration',
    'student_amount', 'course_amount', 'dropout_rate_of_course', 'dropout_rate_of_user',
    'L_mean', 'L_var', 'L_skew', 'L_kurtosis', 
    'S_mean', 'S_var', 'S_skew', 'S_kurtosis',
    '11','12','13','14',
    '21','22','23','24',
    '31','32','33','34',
    '41','42','43','44']
for header in value_feat:
    feature_columns.append(feature_column.numeric_column(header))

#%% 嵌入列
embedding_feat = [   
    'gender',  'edu_degree', 
    'course_category', 'course_type', ]

for header in embedding_feat:
    embedding_ = feature_column.embedding_column(header, dimension=8)
    feature_columns.append(embedding_)

# 分桶列
#age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
#feature_columns.append(age_buckets)

# 分类列
#thal = feature_column.categorical_column_with_vocabulary_list(
#      'thal', ['fixed', 'normal', 'reversible'])
#thal_one_hot = feature_column.indicator_column(thal)
#feature_columns.append(thal_one_hot)


# %%
feature_layer =  layers.DenseFeatures(feature_columns)
batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# %%

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(16, activation='relu'),
  layers.Dense(16, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])


#%%
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'],
              run_eagerly=True)

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)







# %%

import pandas as pd
import numpy as np

from random import random
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, LSTM, Embedding

def loadDataRandom(len):
    X = []
    Y = []
    
    prev = 0
    cur = 0
    prevState = 0
    
    for i in range(len):
        prev = cur
        cur = random()
        
        curState = 0
        if cur > prev:
            curState = 1
        else:
            curState = 0
        
        y = 0
        if curState == prevState:
            y = 1
            
        X.append([cur, prev, prevState])
        Y.append(y)
        
        # print(cur, prev, curState, prevState, y)
        prevState = curState
        
    return np.array(X), np.array(Y)

# %%
def create_model(input_length):
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(input_length, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
# %%
#X_train, y_train = loadDataRandom(1000)

X_train = train.values[:,:-1]
y_train = train.values[:,-1]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
print(X_train.shape, y_train.shape)

model = create_model(len(X_train[0]))
hist = model.fit(X_train, y_train, batch_size=64, validation_split = 0.2, epochs=20, shuffle=False, verbose=1)

pyplot.plot(hist.history['loss'], label='loss')
pyplot.plot(hist.history['accuracy'], label='acc')
pyplot.plot(hist.history['val_accuracy'], label='val_acc')
pyplot.legend()
pyplot.show()
# %%
X_val, y_val = test.values[:,:-1],test.values[:,:-1]
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
result = model.predict(X_val, verbose=0)

pyplot.plot(result.history['loss'], label='loss')
pyplot.plot(result.history['accuracy'], label='acc')
pyplot.plot(result.history['val_accuracy'], label='val_acc')
pyplot.legend()
pyplot.show()

#%%