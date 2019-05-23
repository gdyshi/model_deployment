import tensorflow as tf
import numpy as np

# 读取数据
f = np.load('./data/mnist.npz')
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
x_train = np.reshape(x_train, [-1, 784])
x_test = np.reshape(x_test, [-1, 784])

# 读取模型
saver = tf.train.import_meta_graph('./saved_tf/tf_model.meta')
graph = tf.get_default_graph()

images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
output = tf.get_default_graph().get_tensor_by_name("output/Softmax:0")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint('./saved_tf'))
    feed_dict = {images_placeholder: x_test}
    # 推理模型
    logits=sess.run(output, feed_dict=feed_dict)
    print(logits.astype(np.int32))

