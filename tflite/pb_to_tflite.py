import tensorflow as tf

graph_def_file = "../model/saved_pb/tensorflow.pb"
input_arrays = ["input"]
output_arrays = ["output/Softmax"]

converter = tf.lite.TFLiteConverter.from_frozen_graph(
  graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
open("model.tflite", "wb").write(tflite_model)