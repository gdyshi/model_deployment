import keras
import tensorflow as tf
import os
os.makedirs('./keras',exist_ok=True)
# 加载模型
model = keras.models.load_model('../model/saved_keras/save.h5')
sess = tf.Session()
sess.run(tf.global_variables_initializer())
graph = tf.get_default_graph()

# 打印模型结构
model.summary()

# 绘制简单的模型结构图
keras.utils.plot_model(model,'./keras/model.png')

# 生成tensorboard文件
tensorboard_callback = keras.callbacks.TensorBoard('./keras')
tensorboard_callback.set_model(model)
tensorboard_callback.writer.flush()

# 打印模型中所有的操作
op = graph.get_operations()
for i, m in enumerate(op):
   print('op{}:'.format(i), m.values())
