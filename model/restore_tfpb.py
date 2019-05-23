import tensorflow as tf
import numpy as np

# 读取数据
f = np.load('./data/mnist.npz')
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
x_train = np.reshape(x_train, [-1, 784])
x_test = np.reshape(x_test, [-1, 784])

# 读取模型
output_graph_def = tf.GraphDef()
# 打开.pb模型
with open('./saved_pb/tensorflow.pb', "rb") as f:
    output_graph_def.ParseFromString(f.read())
    tensors = tf.import_graph_def(output_graph_def, name="")

graph = tf.get_default_graph()
input_x = graph.get_tensor_by_name("input:0")
out_softmax = graph.get_tensor_by_name("output/Softmax:0")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 推理模型
    logits = sess.run(out_softmax, feed_dict={input_x: x_test})
    print(logits.astype(np.int32))

