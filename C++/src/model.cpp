// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
// SPDX-License-Identifier: MIT
// Copyright (c) 2018 - 2019 gdyshi <gdyshi@126.com>.
//
// Permission is hereby  granted, free of charge, to any  person obtaining a copy
// of this software and associated  documentation files (the "Software"), to deal
// in the Software  without restriction, including without  limitation the rights
// to  use, copy,  modify, merge,  publish, distribute,  sublicense, and/or  sell
// copies  of  the Software,  and  to  permit persons  to  whom  the Software  is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE  IS PROVIDED "AS  IS", WITHOUT WARRANTY  OF ANY KIND,  EXPRESS OR
// IMPLIED,  INCLUDING BUT  NOT  LIMITED TO  THE  WARRANTIES OF  MERCHANTABILITY,
// FITNESS FOR  A PARTICULAR PURPOSE AND  NONINFRINGEMENT. IN NO EVENT  SHALL THE
// AUTHORS  OR COPYRIGHT  HOLDERS  BE  LIABLE FOR  ANY  CLAIM,  DAMAGES OR  OTHER
// LIABILITY, WHETHER IN AN ACTION OF  CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE  OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "tf_utils.hpp"
#include "model.h"
#include <iostream>
#include <string.h>
#include <vector>

#define OUTPUT_DIM_LEN  (2)

typedef struct {
    float *data;
    std::size_t data_size;
    std::int64_t *dims;
    std::size_t dim_len;
} tensor_t;

typedef struct {
    TF_Graph *graph;
    TF_Status *status;
    TF_Session *sess;
    TF_Output input_op;
    TF_Output out_op;
    tensor_t input;
    std::int64_t *out_dim;
    int out_data_size;
//    tensor_t output;
} tf_app_t;

static tf_app_t tf_app = {0};

int model_init(char * pb_path) {
    tf_app.graph = tf_utils::LoadGraph(pb_path);
    if (tf_app.graph == nullptr) {
        printf("Can't load graph\n");
        return 1;
    }

    tf_app.input_op = {TF_GraphOperationByName(tf_app.graph, "input"), 0};
    if (tf_app.input_op.oper == nullptr) {
        printf("Can't init input_op\n");
        return 2;
    }
    tf_app.out_op = {TF_GraphOperationByName(tf_app.graph, "output/Softmax"), 0};
    if (tf_app.out_op.oper == nullptr) {
        printf("Can't init out_op\n");
        return 3;
    }

    tf_app.status = TF_NewStatus();
    TF_SessionOptions *options = TF_NewSessionOptions();
    tf_app.sess = TF_NewSession(tf_app.graph, options, tf_app.status);
    TF_DeleteSessionOptions(options);
    if (TF_GetCode(tf_app.status) != TF_OK) {
        printf("Can't init sess\n");
        return 4;
    }

    return 0;
}

int model_deinit() {
    TF_CloseSession(tf_app.sess, tf_app.status);
    if (TF_GetCode(tf_app.status) != TF_OK) {
        printf("Error close session\n");
        return 6;
    }

    TF_DeleteSession(tf_app.sess, tf_app.status);
    if (TF_GetCode(tf_app.status) != TF_OK) {
        printf("Error delete session\n");
        return 7;
    }
    tf_utils::DeleteGraph(tf_app.graph);
    TF_DeleteStatus(tf_app.status);
    return 0;
}

int model_inference(int batch_size, float *input_vals, float *output_vals) {
    std::int64_t input_dims[] = {batch_size, INPUT_SIZE};
    tf_app.input.dims = input_dims;
    tf_app.input.dim_len = 2;
    tf_app.input.data_size = batch_size * INPUT_SIZE * sizeof(float);
    tf_app.input.data = static_cast<float *>(malloc(tf_app.input.data_size));

    std::int64_t output_dims[] = {batch_size, OUTPUT_SIZE};
    tf_app.out_dim = output_dims;
    tf_app.out_data_size = batch_size * OUTPUT_SIZE * sizeof(float);

    memcpy(tf_app.input.data, input_vals, tf_app.input.data_size);
    TF_Tensor *input_tensor = tf_utils::CreateTensor(TF_FLOAT,
                                                     tf_app.input.dims, tf_app.input.dim_len,
                                                     tf_app.input.data, tf_app.input.data_size);
    TF_Tensor *output_tensor = nullptr;
    TF_SessionRun(tf_app.sess,
                  nullptr, // Run options.
                  &tf_app.input_op, &input_tensor, 1, // Input tensors, input tensor values, number of inputs.
                  &tf_app.out_op, &output_tensor, 1, // Output tensors, output tensor values, number of outputs.
                  nullptr, 0, // Target operations, number of targets.
                  nullptr, // Run metadata.
                  tf_app.status // Output status.
    );
    if (TF_GetCode(tf_app.status) != TF_OK) {
        printf("Error run session\n");
        return 5;
    }
    int dims = TF_NumDims(output_tensor);
    if (dims != OUTPUT_DIM_LEN) {
        printf("Error output dim len\n");
        return 5;
    }
    for (int i = 0; i < dims; ++i) {
        if (TF_Dim(output_tensor, i) != tf_app.out_dim[i]) {
            printf("Error output dim len\n");
        }
    }

    const auto data = static_cast<float *>(TF_TensorData(output_tensor));
    memcpy(output_vals, static_cast<float *>(TF_TensorData(output_tensor)), tf_app.out_data_size);
    tf_utils::DeleteTensor(input_tensor);
    tf_utils::DeleteTensor(output_tensor);
    return 0;
}


