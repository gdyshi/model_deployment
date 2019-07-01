/**
 *
 * @author gdyshi
 * @version 0.1
 */
public class JavaModel {
    public JavaModel(String model_file) {
        model_init(model_file);
    }

    public void close() {
        model_deinit();
    }
    public float[] convert(float[][] input){
        float[] output = new float[input.length*input[0].length];
        System.out.println(output.length);
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[i].length; j++) {
                output[i+j] = input[i][j];
            }
        }
        return output;
    }
    public float[][] convert(int batch_size, float[] input){
        float[][] output = new float[batch_size][input.length/batch_size];
        System.out.println(output.length);
        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < output[i].length; j++) {
                output[i][j] = input[i+j];
            }
        }
        return output;
    }
    public float[][] inference(int batch_size, float[][] input_vals){
        float[] outpu=new float[batch_size*10];
        float[] inpu=convert(input_vals);

        int ret = model_inference(batch_size, inpu, outpu);

        float[][] output_vals=convert(batch_size,outpu);
        return output_vals;
    }

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("jni_model");
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    private native int model_init(String model_file);

    private native int model_deinit();

    private native int model_inference(int batch_size, float[] input_vals, float[] output_vals);

}