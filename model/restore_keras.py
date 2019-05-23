import keras
import numpy as np

# 读取数据
f = np.load('./data/mnist.npz')
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
x_train = np.reshape(x_train, [-1, 784])
x_test = np.reshape(x_test, [-1, 784])

# 恢复模型和权重
model = keras.models.load_model('./saved_keras/save.h5')

# 训练模型
logits=model.predict(x_test)
print(logits.astype(np.int32))
