import tensorflow as tf
from keras import backend as K
from tensorflow.python.framework import graph_io
import keras
from tensorflow.python.framework.graph_util import convert_variables_to_constants

import os
os.makedirs('./saved_pb', exist_ok=True)

"""----------------------------------导入keras模型------------------------------"""
model = keras.models.load_model('./saved_keras/save.h5')

print('input is :', model.input.name)
print('output is:', model.output.name)

"""----------------------------------保存为.pb格式------------------------------"""
sess = K.get_session()

graph = sess.graph
with graph.as_default():
    output_names = [model.output.op.name]
    input_graph_def = graph.as_graph_def()
    frozen_graph = convert_variables_to_constants(sess, input_graph_def, output_names)
graph_io.write_graph(frozen_graph, './saved_pb', 'keras.pb', as_text=False)
