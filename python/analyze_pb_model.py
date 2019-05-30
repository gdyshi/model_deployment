import tensorflow as tf
import os
os.makedirs('./pb',exist_ok=True)

# 加载模型
output_graph_def = tf.GraphDef()
with open('../model/saved_pb/tensorflow.pb', "rb") as f:
    output_graph_def.ParseFromString(f.read())
    tensors = tf.import_graph_def(output_graph_def, name="")
sess = tf.Session()
sess.run(tf.global_variables_initializer())
graph = tf.get_default_graph()

# 生成tensorboard文件
file_writer = tf.summary.FileWriter('./pb')
file_writer.add_graph(graph)
file_writer.flush()

# 打印模型中所有的操作
op = graph.get_operations()
for i, m in enumerate(op):
   print('op{}:'.format(i), m.values())

