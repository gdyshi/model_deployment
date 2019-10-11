import os
import tensorflow as tf
from tensorflow import keras

version = 1
export_path = './export/1'
os.makedirs(export_path,exist_ok=True)

model = keras.models.load_model('../model/saved_keras/save.h5')

tf.saved_model.simple_save(
    keras.backend.get_session(),
    export_path,
    inputs={'input_image': model.input},
    outputs={t.name:t for t in model.outputs})

