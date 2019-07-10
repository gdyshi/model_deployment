import tensorflow as tf

class python_model():
    def __init__(self,model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        # Get input and output tensors.
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.input_index = input_details[0]['index']
        self.output_index = output_details[0]['index']
        self.input_shape = input_details[0]['shape']
        self.output_shape = output_details[0]['shape']

    def inference(self,data):
        assert data.shape[0] == self.input_shape[0] and data.shape[1] == self.input_shape[1], 'data shape error!!'

        self.interpreter.set_tensor(self.input_index, data)
        self.interpreter.invoke()
        return  self.interpreter.get_tensor(self.output_index)


