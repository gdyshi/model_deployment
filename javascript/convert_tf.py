import shutil
import tensorflow as tf
import tensorflowjs as tfjs

# 读取模型
output_graph_def = tf.GraphDef()
# 打开.pb模型
with open('../model/saved_pb/tensorflow.pb', "rb") as f:
    output_graph_def.ParseFromString(f.read())
    tensors = tf.import_graph_def(output_graph_def, name="")

graph = tf.get_default_graph()
input_x = graph.get_tensor_by_name("input:0")
out_softmax = graph.get_tensor_by_name("output/Softmax:0")

shutil.rmtree('./tf_model',ignore_errors=True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.saved_model.simple_save(sess,'./tf_model',inputs={'input':input_x},outputs={"output/Softmax":out_softmax})

shutil.rmtree('./tf_model',ignore_errors=True)
tfjs.converters.convert_tf_saved_model('./tf_model','./webmod_tf')

print('ok')
