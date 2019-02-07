#ifndef  __CONFIG_H__
#define __CONFIG_H__

#define DATA_TYPE float

#define FIXED_INPUT_WIDTH 640
#define FIXED_INPUT_HEIGHT 480

#define E 2.718281828

//#define DEBUG_MODE

static const char *RAWIMG_RESULTS = "./dump/raw_img.txt";
static const char *DATA_RESULTS   = "./dump/data.txt";

static const char *CONV1_RESULTS  = "./dump/conv1.txt";
static const char *NORM1_RESULTS  = "./dump/norm1.txt";
static const char *POOL1_RESULTS  = "./dump/pool1.txt";

static const char *CONV2_RESULTS  = "./dump/conv2.txt";
static const char *NORM2_RESULTS  = "./dump/norm2.txt";
static const char *POOL2_RESULTS  = "./dump/pool2.txt";

static const char *CONV3_RESULTS  = "./dump/conv3.txt";
static const char *CONV4_RESULTS  = "./dump/conv4.txt";
static const char *CONV5_RESULTS  = "./dump/conv5.txt";

static const char *POOL5_RESULTS  = "./dump/pool5.txt";

static const char *FC6_RESULTS    = "./dump/fc6.txt";
static const char *FC7_RESULTS    = "./dump/fc7.txt";
static const char *FC8_RESULTS    = "./dump/fc8.txt";

static const char *OUTPUT_RESULTS = "./dump/output.txt";


// configurable model parameters
#define LOCAL_FILE_DATA_WIDTH	4

static const char *CONV1_FILTER_PATH	= "./model_para/conv1_filter.txt";
static const char *CONV1_BIAS_PATH		= "./model_para/conv1_bias.txt";
static const char *CONV2_FILTER_PATH	= "./model_para/conv2_filter.txt";
static const char *CONV2_BIAS_PATH		= "./model_para/conv2_bias.txt";
static const char *CONV3_FILTER_PATH	= "./model_para/conv3_filter.txt";
static const char *CONV3_BIAS_PATH		= "./model_para/conv3_bias.txt";
static const char *CONV4_FILTER_PATH	= "./model_para/conv4_filter.txt";
static const char *CONV4_BIAS_PATH		= "./model_para/conv4_bias.txt";
static const char *CONV5_FILTER_PATH	= "./model_para/conv5_filter.txt";
static const char *CONV5_BIAS_PATH		= "./model_para/conv5_bias.txt";
static const char *FC6_FILTER_PATH		= "./model_para/fc6_filter.txt";
static const char *FC6_BIAS_PATH		= "./model_para/fc6_bias.txt";
static const char *FC7_FILTER_PATH		= "./model_para/fc7_filter.txt";
static const char *FC7_BIAS_PATH		= "./model_para/fc7_bias.txt";
static const char *FC8_FILTER_PATH		= "./model_para/fc8_filter.txt";
static const char *FC8_BIAS_PATH		= "./model_para/fc8_bias.txt";

static const char *FRAME_DIFF_THRES_PATH = "./model_para/frame_diff_thres.txt";

#endif // ! __CONFIG_H__
