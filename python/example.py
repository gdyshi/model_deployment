from python_model import python_model
import numpy as np

model = python_model(model_path='../model/saved_pb/tensorflow.pb')


# 读取数据
f = np.load('../model/data/mnist.npz')
x_test, y_test = f['x_test'], f['y_test']
x_test = np.reshape(x_test, [-1, 784])

output=model.inference(x_test)
print(output.astype(np.int32))
