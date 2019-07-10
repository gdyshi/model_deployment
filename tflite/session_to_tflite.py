import tensorflow as tf

# 读取模型
output_graph_def = tf.GraphDef()
# 打开.pb模型
with open("../model/saved_pb/tensorflow.pb", "rb") as f:
    output_graph_def.ParseFromString(f.read())
    tensors = tf.import_graph_def(output_graph_def, name="")

graph = tf.get_default_graph()
input_x = graph.get_tensor_by_name("input:0")
out_softmax = graph.get_tensor_by_name("output/Softmax:0")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    converter = tf.lite.TFLiteConverter.from_session(sess, [input_x], [out_softmax])
    tflite_model = converter.convert()
    open("model.tflite", "wb").write(tflite_model)

