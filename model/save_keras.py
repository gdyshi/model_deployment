import keras
import keras.layers as layers
import numpy as np

import os
os.makedirs('./saved_keras', exist_ok=True)

# 读取数据
f = np.load('./data/mnist.npz')
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
x_train = np.reshape(x_train, [-1, 784])
x_test = np.reshape(x_test, [-1, 784])

# 构建模型
model = keras.Sequential(
    [layers.Dense(64, activation='relu',name='fc1'),
     layers.Dense(64, activation='relu',name='fc2'),
     layers.Dense(10, activation='softmax',name='output')])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=10, batch_size=10000, verbose=2, validation_data=(x_test, y_test))

# 保存模型
model.save('./saved_keras/save.h5')

# 保存模型结构为 json 格式
json_string = model.to_json()
with open('./saved_keras/to_json.json', 'w') as f:
    f.write(json_string)

# 保存模型结构为 yaml 格式
yaml_string = model.to_yaml()
with open('./saved_keras/to_yaml.yaml', 'w') as f:
    f.write(json_string)

# 保存模型权重
model.save_weights('./saved_keras/save_weights.h5')
