package com.example.gdyshi.hello_tf;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class Model {
    private final int in_dim = 784;
    private final int out_dim = 10;
    private Interpreter interpreter;
    private ByteBuffer input = null;
    private float[][] output = null;
    public Model(Activity activity) {
        MappedByteBuffer model_buffer= null;
        try {
            model_buffer = loadModelFile(activity,"model.tflite");
        } catch (IOException e) {
            e.printStackTrace();
        }

        interpreter = new Interpreter(model_buffer);
        output = new float[1][out_dim];
    }

    public float[] inference(float[][] data){
        interpreter.run(data,output);
        return output[0];
    }
    public void close(){
        interpreter.close();
    }

    private MappedByteBuffer loadModelFile(Activity activity, String model_file) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(model_file);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

}
