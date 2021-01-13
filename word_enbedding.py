'''
    脚本功能： 
            按照预设的序列长度，截断或填充

             
'''
import json
json_dict_path = ' '
log = json.load(open(json_dict_path))

import tensorflow as tf
from tensorflow import keras
import numpy as np
print(tf.__version__)

train_data = keras.preprocessing.sequence.pad_sequences(
    train_data
    ,value= 0.0
    ,padding='post' # 未知
    ,dtype=np.int8
    ,maxlen=1000    # 单条序列最大长度
    ) 

# model

vocab_size = 22 # 词汇表大小

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
                    epochs=10,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)



