from tensorflow import keras
import tensorflowjs as tfjs

model = keras.models.load_model('../model/saved_keras/save.h5')
tfjs.converters.save_keras_model(model,'webmod_keras')

print('ok')
