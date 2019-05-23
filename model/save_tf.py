import tensorflow as tf
import numpy as np

import os
os.makedirs('./saved_tf',exist_ok=True)

# 读取数据
f = np.load('./data/mnist.npz')
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
x_train = np.reshape(x_train, [-1, 784])
x_test = np.reshape(x_test, [-1, 784])

# 构建模型
image = tf.placeholder(tf.float32, shape=[None, 784], name="input")
y = tf.placeholder(tf.int32, shape=[None, ], name="labels")
embedding = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                         0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                         0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                         0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                         0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                         0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                         0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                         0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                         0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                         1, 0, 0, 0, 0, 0, 0, 0, 0, 0], shape=[10, 10])
label = tf.nn.embedding_lookup(embedding, y)
layers = tf.keras.layers
x = layers.Dense(64, activation='relu',name='fc1')(image)
x = layers.Dense(64, activation='relu',name='fc2')(x)
logits = layers.Dense(10, activation='softmax',name='output')(x)

losses = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=label))
train_step = tf.train.AdamOptimizer(0.001).minimize(losses)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 训练模型
    for i in range(10):
        _, loss_value = sess.run([train_step, losses], feed_dict={image: x_train, y: y_train})

    # 保存模型
    saver.save(sess, './saved_tf/tf_model')
