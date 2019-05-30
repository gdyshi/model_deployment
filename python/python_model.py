import tensorflow as tf

class python_model():
    def __init__(self,model_path):
        # 读取模型
        output_graph_def = tf.GraphDef()
        # 打开.pb模型
        with open(model_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tensors = tf.import_graph_def(output_graph_def, name="")

        self.__sess = tf.Session()
        self.__sess.run(tf.global_variables_initializer())
        graph = tf.get_default_graph()
        self.__input = graph.get_tensor_by_name("input:0")
        self.__output = graph.get_tensor_by_name("output/Softmax:0")

    def inference(self,input):
        output = self.__sess.run(self.__output, feed_dict={self.__input: input})
        return output
