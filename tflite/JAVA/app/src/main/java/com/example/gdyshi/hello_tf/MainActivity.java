package com.example.gdyshi.hello_tf;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.text.SpannableString;
import android.text.SpannableStringBuilder;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {
    private TextView textView;
    private Button button;
    private Model model=null;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        button = findViewById(R.id.button);
        textView = findViewById(R.id.textView);
        button.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        float[][] matrix = new float[1][784];
                        float[] probabilities = model.inference(matrix);
                        int label = argmax(probabilities);
                        SpannableStringBuilder builder = new SpannableStringBuilder();
                        SpannableString str1 = new SpannableString(String.valueOf(label));
                        builder.append(str1);
                        textView.setText(builder);
                    }
                }
        );
        init_model();
    }

    private void init_model(){
        if(null == model){
            model = new Model(this);
        }
    }

    @Override
    protected void onDestroy() {
        model.close();
        model=null;
        super.onDestroy();
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
