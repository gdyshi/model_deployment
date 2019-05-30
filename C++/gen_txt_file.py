import numpy as np

f = np.load('../model/data/mnist.npz')
x_test, y_test = f['x_test'], f['y_test']
x_test = np.reshape(x_test, [-1, 784])
np.savetxt('test_file.txt',x_test[1:100],delimiter=',')
