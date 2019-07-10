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

#include "model.h"
#include <iostream>

int read_txt(FILE *fp, float *data) {
    for (int j = 0; j < INPUT_SIZE; ++j) {
        if (1 != fscanf(fp, "%f,", &data[j])) {
            return -1;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
	if (argc < 3) {
		fprintf(stderr, "%s <tflite model> <in file>\n",argv[0]);
		return 1;
	}
	char *model_file = argv[1];
	char *in_file = argv[2];
	FILE *fp= NULL;
    fp = fopen(in_file, "r");
    if (NULL == fp) { exit(1); }
	float in_data[1][INPUT_SIZE];
	float out_data[1][OUTPUT_SIZE];

	model_init(model_file);
	//read_txt(fp,&in_data[0][0]);
	model_inference(&in_data[0][0],&out_data[0][0]);
	model_deinit();

	printf("output:\n");
	for(int i=0; i<OUTPUT_SIZE; i++){
	    printf("%f\t",out_data[1][i]);
	}
	printf("\n");

	return 0;
}

