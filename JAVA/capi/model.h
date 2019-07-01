#ifndef HELLO_TF_LIB_H
#define HELLO_TF_LIB_H
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

#include <iostream>
#include <string.h>
#include <vector>

#define INPUT_SIZE  (784)//输入维度
#define OUTPUT_SIZE  (10)//输出维度

int model_init(char * pb_path);

int model_deinit();

int model_inference(int batch_size, float *input_vals, float *output_vals);

#endif //HELLO_TF_LIB_H
