import com.google.common.io.ByteStreams;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;
import org.tensorflow.TensorFlow;

public class JavaModel {
    final long BATCH_SIZE = 1;
    long[] shape = new long[]{BATCH_SIZE, 784};
    float[][] result = null;
    Session session = null;

    public JavaModel(String model_file) {
        Graph graph = new Graph();
        this.session = new Session(graph);
        try {
            graph.importGraphDef(loadGraphDef(model_file));
        } catch (Exception e) {
            System.err.println("error");
            System.out.println(e.getMessage());
        }
    }

    public float[][] inference(float[][] matrix) {
        Tensor<Float> input = Tensor.create(matrix, Float.class);
        Tensor<Float> output =
                this.session.runner()
                        .feed("sequential_1_input", input)
                        .fetch("output/Softmax")
                        .run()
                        .get(0).expect(Float.class);
        float[][] probabilities = new float[(int) output.shape()[0]][(int) output.shape()[1]];
        output.copyTo(probabilities);
        return probabilities;
    }

    private static byte[] loadGraphDef(String model_file) throws IOException {
        try (InputStream is = JavaModel.class.getClassLoader().getResourceAsStream(model_file)) {
            return ByteStreams.toByteArray(is);
        }
    }
}
    