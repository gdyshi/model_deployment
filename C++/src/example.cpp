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

int read_txt(FILE *fp, int batch_size, float *data) {
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < INPUT_SIZE; ++j) {
            if (1 != fscanf(fp, "%f,", &data[INPUT_SIZE * i + j])) {
                return -1;
            }
        }
    }
    return batch_size;
}

int main(int argc, char **argv) {
#define batch_size  5
    char *filename = NULL;
    char *out_data_name = (char *) "out.csv";
    char *pb_file = (char *) "../../../model/saved_pb/tensorflow.pb";
    FILE *fp_in = NULL;
    FILE *fp_out = NULL;

    float input_vals[batch_size][INPUT_SIZE] = {0};
    float output_vals[batch_size][OUTPUT_SIZE] = {0};
    if (2 > argc) {
        printf("usage: \n%s [TXT FILE] [MODEL FILE default ../../../model/saved_pb/tensorflow.pb] [OUT DATA FILE default out.csv]\n", argv[0]);
        return -1;
    }
    filename = argv[1];
    if (3 <= argc) {
        out_data_name = argv[2];
    }
    if (4 <= argc) {
        out_data_name = argv[3];
    }
    fp_in = fopen(filename, "r");
    if (NULL == fp_in) { exit(1); }
    fp_out = fopen(out_data_name, "w");
    if (NULL == fp_out) { exit(1); }
    fprintf(fp_out, "seq,0,1,2,3,4,5,6,7,8,9\n");
    int seq = 0;
    for (int i = 0; i < 1; ++i) {
        printf("%d\n", i);
        fseek(fp_in, 0, SEEK_SET);
        model_init(pb_file);
        while (batch_size == read_txt(fp_in, batch_size, &input_vals[0][0])) {
            model_inference(batch_size, &input_vals[0][0], &output_vals[0][0]);
            for (int j = 0; j < batch_size; ++j) {
                fprintf(fp_out, "%d,%e,%e,%e,%e,%e,%e,%e,%e,%e,%e\n", seq++, output_vals[j][0], output_vals[j][1],
                        output_vals[j][2], output_vals[j][3], output_vals[j][4], output_vals[j][5], output_vals[j][6],
                        output_vals[j][7], output_vals[j][8], output_vals[j][9]);
            }
        }
        model_deinit();
    }
    fclose(fp_out);
    fclose(fp_in);
    return 0;
}

