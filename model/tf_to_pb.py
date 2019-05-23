import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.python.framework.graph_util import convert_variables_to_constants

import os
os.makedirs('./saved_pb', exist_ok=True)


"""----------------------------------导入tensorflow模型------------------------------"""
saver = tf.train.import_meta_graph('./saved_tf/tf_model.meta')
graph = tf.get_default_graph()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess, tf.train.latest_checkpoint('./saved_tf'))


"""----------------------------------保存为.pb格式------------------------------"""
graph = sess.graph
with graph.as_default():
    output_names = ["output/Softmax"]
    input_graph_def = graph.as_graph_def()
    frozen_graph = convert_variables_to_constants(sess, input_graph_def, output_names)
graph_io.write_graph(frozen_graph, './saved_pb', 'tensorflow.pb', as_text=False)
