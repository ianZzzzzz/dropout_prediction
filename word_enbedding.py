'''
    pipline预处理的时候把字母表换成数字 以便pad_sequences()
    从json读取不含零的dict
    完成mask的制作以及传递
'''
import json
json_dict_path = ' '
log = json.load(open(json_dict_path))

import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)
train_data = np.array([b'E', b'C', b'B', b'C', b'B', b'C', b'U', b'E', b'U', b'E', b'B',    
       b'U', b'E', b'B', b'U', b'E', b'B', b'U', b'E', b'B', b'U', b'E',    
       b'B', b'D', b'U', b'E', b'B', b'U', b'E', b'B', b'C', b'U', b'E',    
       b'B', b'C', b'B'])
train_data = np.array([[1,2,3,4],[1,2,3,4]],dtype=np.int32).tolist()
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,
    value= 0.0,
    padding='post',
    dtype='int32',
    maxlen=16)

# model

vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()
#
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)



