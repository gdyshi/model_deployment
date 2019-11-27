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
#include <cstdio>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include "model.h"

using namespace tflite;

std::unique_ptr<Interpreter> interpreter;

static int in_index=-1;
static int out_index=-1;

void run_invoke(){
	if(interpreter->Invoke() != kTfLiteOk)
	{
		fprintf(stderr, "Error Invoke\n");
	}
}

int model_init(char * model_file) {
	// Load model
	static std::unique_ptr<tflite::FlatBufferModel> model =
		tflite::FlatBufferModel::BuildFromFile(model_file);
	if(model == nullptr)
	{
		fprintf(stderr, "Error open model\n");
		return -1;
	}
	static tflite::ops::builtin::BuiltinOpResolver resolver;
	// Build the interpreter
	static InterpreterBuilder builder(*model, resolver);
	builder(&interpreter);
	if(interpreter == nullptr)
	{
		fprintf(stderr, "Error get interpreter\n");
		return -1;
	}

	// Allocate tensor buffers.
	if(interpreter->AllocateTensors() != kTfLiteOk)
	{
		fprintf(stderr, "Error AllocateTensors\n");
		return -1;
	}

	in_index = interpreter->inputs()[0];
	out_index = interpreter->outputs()[0];
	printf("in index:%d,out index:%d\n",in_index,out_index);

	return 0;
}

int model_deinit() {
}

int model_inference(float *input_vals, float *output_vals) {
	memcpy(interpreter->typed_tensor<float>(in_index), &input_vals[0], INPUT_SIZE*sizeof(input_vals[0]));

	// Run inference
	if(interpreter->Invoke() != kTfLiteOk)
	{
		fprintf(stderr, "Error Invoke\n");
		return -1;
	}

	// Read output buffers
	float* output = interpreter->typed_tensor<float>(out_index);
	memcpy(&output_vals[0], output, OUTPUT_SIZE*sizeof(output_vals[0]));
	return 0;
}


