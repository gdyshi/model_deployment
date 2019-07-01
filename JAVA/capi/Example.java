public class Example {
    public static void main(String[] args) throws Exception {
        final int BATCH_SIZE=2;
        String filename="mnist.npz";
        JavaModel model = new JavaModel("../../model/saved_pb/tensorflow.pb");
//        准备数据
        float[][] matrix = new float[BATCH_SIZE][784];
//        推理
        float[][] probabilities = model.inference(BATCH_SIZE,matrix);
        model.close();
        int label = argmax(probabilities[0]);
        System.out.printf(
                "%-30s -->%d(%.2f%% likely)\n",
                filename, label, probabilities[0][label] * 100.0);
    }

    private static int argmax(float[] probabilities) {
        int best = 0;
        for (int i = 1; i < probabilities.length; ++i) {
            if (probabilities[i] > probabilities[best]) {
                best = i;
            }
        }
        return best;
    }
}