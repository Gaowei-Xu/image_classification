#ifndef  __UTILS_H__
#define __UTILS_H__

#include "Blob.h"
#include "config.h"
#include <stdio.h>
#include <stdarg.h>

#define INIT_SUCCESS		0
#define INIT_FAILED			-1

#define GDERROR    "ERROR"
#define GDDETAIL   "DETAIL"
#define GDTRACE    "TRACE"

#define LOG(level, format, ...) \
    do { \
        fprintf(stderr, "[%s] [Function %s@%s, line %d] " format, \
                level, __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__); \
    }while(0)


#define MAX(x,y) (((x) > (y))?(x):(y))
#define MIN(x,y) (((x) < (y))?(x):(y))


// ReLU function
DATA_TYPE ReLU(DATA_TYPE x);

// load model parameters from loal files (hex-format) and store them in the corresponding blob
int initialize_model_para(Blob<DATA_TYPE> &kernel, const char *kernel_filter_path, const char *kernel_bias_path);

// load model parameters from loal files (dec-format) and store them in the corresponding blob
int load_net_para(Blob<DATA_TYPE> &kernel, const char *kernel_filter_path, const char *kernel_bias_path);

// load frames difference threshold file
int load_threshold(const char *frame_diff_path, long int *threshold);

// load image from local ppm-format image into Blob
int loadImagePPM(const char *image_name, Blob<DATA_TYPE> &img);

// load image from local jpg-format image into Blob
int loadImageJPG(const char *image_name, Blob<DATA_TYPE> &img);

// preprocess the original image and obtain the Alexnet standard input size
int preprocess(Blob<DATA_TYPE> &img, Blob<DATA_TYPE> &data);

// convlotional
int convolutional(Blob<DATA_TYPE> &bottom, Blob<DATA_TYPE> &kernel, Blob<DATA_TYPE> &top, int stride, int pad, const char * dump_file_path);
int convolutional_gp(Blob<DATA_TYPE> &bottom, Blob<DATA_TYPE> &kernel, Blob<DATA_TYPE> &top, int stride, int pad, const char * dump_file_path, int group);


// Alexnet LRN module
int lrn(Blob<DATA_TYPE> &bottom, Blob<DATA_TYPE> &top, double k, int n, double alpha, double beta, const char * dump_file_path);

// pooling (kernel size = 3, stride = 2, padding = 0)
int pooling(Blob<DATA_TYPE> &bottom, Blob<DATA_TYPE> &top, int ks, int stride, int padding, const char * dump_file_path);

// fully connection layer
int fc(Blob<DATA_TYPE> &bottom, Blob<DATA_TYPE> &kernel, Blob<DATA_TYPE> &top, const char * dump_file_path, bool runRelu);

// softmax function
int softmax(Blob<DATA_TYPE> &bottom, Blob<DATA_TYPE> &top, const char * dump_file_path);

// calculate the difference of the two different frames
long int calc_diff(Blob<DATA_TYPE> &prev_img, Blob<DATA_TYPE> &img);

#endif // ! __UTILS_H__
