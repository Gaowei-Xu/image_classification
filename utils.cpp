#include <iostream>
#include <math.h>
#include <assert.h>
#include "utils.h"
#include "Blob.h"

#define ABS(x)  (((x) >= 0.000001) ? (x) : (-x)) 


/*!
 * ReLu function
 */
DATA_TYPE ReLU(DATA_TYPE x) {
	return (DATA_TYPE)((x >= 0.00000001) ? x : 0.0);
}



/*!
 * load model parameters from loal files (hex-format) and store them in the corresponding blob
 */
int initialize_model_para(Blob<DATA_TYPE> &kernel, const char *kernel_filter_path, const char *kernel_bias_path) {
	int num = kernel.num();
	int channel = kernel.channels();
	int height = kernel.height();
	int width = kernel.width();

	int filter_paras_len = num * channel * height * width * LOCAL_FILE_DATA_WIDTH;
	int bias_paras_len = num * LOCAL_FILE_DATA_WIDTH;

	FILE *fp_filter = fopen(kernel_filter_path, "rb");
	if (fp_filter == NULL) {
		LOG(GDERROR, "Cannot open file %s\n", kernel_filter_path);
		return INIT_FAILED;
	}

	if (sizeof(DATA_TYPE) == LOCAL_FILE_DATA_WIDTH) {
		//LOG(GDDETAIL, "Filter data in local file has a width of %d bytes, global filter data width is %d bytes.\n", LOCAL_FILE_DATA_WIDTH, (int)(sizeof(DATA_TYPE)));
		fread(kernel.getDataPtr(), LOCAL_FILE_DATA_WIDTH, filter_paras_len, fp_filter);
	}
	else {
		//LOG(GDDETAIL, "Filter data in local file has a width of %d bytes, global filter data width is %d bytes.\n", LOCAL_FILE_DATA_WIDTH, (int)(sizeof(DATA_TYPE)));
		float *temp_filter = new float[filter_paras_len];
		fread(temp_filter, LOCAL_FILE_DATA_WIDTH, filter_paras_len, fp_filter);
		for (int i = 0; i < filter_paras_len; ++i)
			*(kernel.getDataPtr() + i) = (DATA_TYPE)temp_filter[i];
		delete[]temp_filter;
	}
	fclose(fp_filter);
	
	bool contain_bias = kernel.contain_bias();
	if (contain_bias) {
		FILE *fp_bias = fopen(kernel_bias_path, "rb");
		if (fp_bias == NULL) {
			LOG(GDERROR, "Cannot open file %s\n", kernel_bias_path);
			return INIT_FAILED;
		}

		if (sizeof(DATA_TYPE) == LOCAL_FILE_DATA_WIDTH) {
			//LOG(GDDETAIL, "Bias in local file has a width of %d bytes, global bias data width is %d bytes.\n", LOCAL_FILE_DATA_WIDTH, (int)(sizeof(DATA_TYPE)));
			fread(kernel.getBiasPtr(), LOCAL_FILE_DATA_WIDTH, bias_paras_len, fp_bias);
		}
		else {
			//LOG(GDDETAIL, "Bias in local file has a width of %d bytes, global bias data width is %d bytes.\n", LOCAL_FILE_DATA_WIDTH, (int)(sizeof(DATA_TYPE)));
			float *temp_bias = new float[bias_paras_len];
			fread(temp_bias, LOCAL_FILE_DATA_WIDTH, bias_paras_len, fp_bias);
			for (int i = 0; i < bias_paras_len; ++i)
				*(kernel.getBiasPtr() + i) = (DATA_TYPE)temp_bias[i];
			delete[]temp_bias;
		}
		fclose(fp_bias);
	}
	
	return INIT_SUCCESS;
}


/*!
* load model parameters from loal files (dec-format) and store them in the corresponding blob 
*/
int load_net_para(Blob<DATA_TYPE> &kernel, const char *kernel_filter_path, const char *kernel_bias_path) {
	int num = kernel.num();
	int channel = kernel.channels();
	int height = kernel.height();
	int width = kernel.width();

	int filter_paras_len = num * channel * height * width;
	int bias_paras_len = num;

	FILE *fp_filter = fopen(kernel_filter_path, "rb");
	if (fp_filter == NULL) {
		LOG(GDERROR, "Cannot open file %s\n", kernel_filter_path);
		return INIT_FAILED;
	}

	float temp;
	for (int i = 0; i < filter_paras_len; ++i) {
		fscanf(fp_filter, "%f", &temp);
		*(kernel.getDataPtr() + i) = (DATA_TYPE)temp;
		//printf("filter[%d]: %f\n", i, temp);
	}
	fclose(fp_filter);

	bool contain_bias = kernel.contain_bias();
	if (!contain_bias)
		return INIT_SUCCESS;

	FILE *fp_bias = fopen(kernel_bias_path, "rb");
	if (fp_bias == NULL) {
		LOG(GDERROR, "Cannot open file %s\n", kernel_bias_path);
		return INIT_FAILED;
	}

	for (int i = 0; i < bias_paras_len; ++i) {
		fscanf(fp_bias, "%f", &temp);
		*(kernel.getBiasPtr() + i) = (DATA_TYPE)temp;
		//printf("bias[%d]: %f\n", i, temp);
	}
	fclose(fp_bias);	

	return INIT_SUCCESS;
}



/*!
 * load threshold for nobody judgement
 */
int load_threshold(const char *frame_diff_path, long int * threshold){
	FILE *fp_diff_thres = fopen(frame_diff_path, "rb");
	if (fp_diff_thres == NULL) {
		LOG(GDERROR, "Cannot open file %s\n", frame_diff_path);
		return INIT_FAILED;
	}
    
    long int temp = 0;
    fscanf(fp_diff_thres, "%ld", &temp);
    fclose(fp_diff_thres);

    *threshold = temp;

    return INIT_SUCCESS;
}


/*!
 * load image from local ppm-format image
 * Note: this function is just for debug, in actual cases, the image 
 * is given from the real-time camera
 */
int loadImagePPM(const char *image_name, Blob<DATA_TYPE> &img){
	char line[128];

	FILE* fp = fopen(image_name, "rb");					// open the image fime
	
	if (fp == NULL){
		LOG(GDERROR, "Cannot load ppm-format image %s\n", image_name);
	}

	fscanf(fp, "%s", line);								// read "P6"
	assert(line[0] == 'P' && line[1] == '6');

	int width, height;
	fscanf(fp, "%d %d", &width, &height);				// read "width" and "height"
    assert(width == FIXED_INPUT_WIDTH && height == FIXED_INPUT_HEIGHT);

	int levels;
	fscanf(fp, "%d\n", &levels);						// read "255"
	assert(levels == 255);

	unsigned char *img_temp = new unsigned char[1 * 3 * width * height];
	fread(img_temp, 1 * 3 * width * height, 1, fp);		// read image data

	/*!
	 * the image data is stored in img_temp Bytes by Bytes, according to the 
	 * order of R-G-B, i.e., RGB-RGB-RGB-RGB-RGB....(from left to right, from
	 * up to bottom)
	 */
#ifdef DEBUG_MODE
	FILE *wf_img = fopen(RAWIMG_RESULTS, "wb");
	for (int c = 0; c < 3; c++) {
		for (int h = 0; h < height; h++) {
			for (int w = 0; w < width; w++) {
				img.setData(img.offset(0, c, h, w), (DATA_TYPE)img_temp[3*(h*width + w)+c]); 
				fprintf(wf_img, "%d\n", (int)(img.data_at(0, c, h, w)));
			}
		}
	}
	fclose(wf_img);
	delete[] img_temp;

#else
	for (int c = 0; c < 3; c++) {
		for (int h = 0; h < height; h++) {
			for (int w = 0; w < width; w++) {
				img.setData(img.offset(0, c, h, w), (DATA_TYPE)img_temp[3 * (h*width + w) + c]);
			}
		}
	}
	delete[] img_temp;
#endif  // DEBUG_MODE

	/*
	DATA_TYPE *ptr = img.getDataPtr();					// (RGB ---> BGR)
	DATA_TYPE c;
	int i;
	for (i = 0; i < width * height; i++){
		c = *ptr;
		*ptr = *(ptr + 2);
		*(ptr + 2) = c;
		ptr += 3;
	}
	*/

	fclose(fp);
	return 0;
}


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

/*!
 * load image from local jpg-format image
 * Note: this function is just for debug, in actual cases, the image 
 * is given from the real-time camera
 */
int loadImageJPG(const char *image_name, Blob<DATA_TYPE> &img){
    Mat src = imread(image_name, 1);  // 1 means reading operation as RGB
    
    // Note that src pixel order is BGR, Not RGB
    int nrows = src.rows;
    int ncols = src.cols;
    int nchannels = src.channels();
    assert(ncols == FIXED_INPUT_WIDTH && nrows == FIXED_INPUT_HEIGHT && nchannels == 3);
    
	unsigned char *img_temp = new unsigned char[1 * 3 * ncols * nrows];
    int width = ncols;
    int height = nrows;
    
    if(src.isContinuous()){
        ncols *= nrows;
        nrows = 1;
    }
    
    unsigned char *data;
    for(int j = 0; j < nrows; ++j){
        data = src.ptr<unsigned char>(j);
        
        for(int i = 0; i < ncols; i++){
            // BGR is data[i], data[i+1], data[i+2]
            img_temp[(j*ncols+i)*3] = data[3*i+2];
            img_temp[(j*ncols+i)*3 + 1] = data[3*i+1];
            img_temp[(j*ncols+i)*3 + 2] = data[3*i+0];
        }
    }
    
    src.release();

    
#ifdef DEBUG_MODE
	FILE *wf_img = fopen(RAWIMG_RESULTS, "wb");
	for (int c = 0; c < 3; c++) {
		for (int h = 0; h < height; h++) {
			for (int w = 0; w < width; w++) {
				img.setData(img.offset(0, c, h, w), (DATA_TYPE)img_temp[3*(h*width + w)+c]); 
				fprintf(wf_img, "%d\n", (int)(img.data_at(0, c, h, w)));
			}
		}
	}
	fclose(wf_img);
	delete[] img_temp;

#else
	for (int c = 0; c < 3; c++) {
		for (int h = 0; h < height; h++) {
			for (int w = 0; w < width; w++) {
				img.setData(img.offset(0, c, h, w), (DATA_TYPE)img_temp[3 * (h*width + w) + c]);
			}
		}
	}
	delete[] img_temp;
#endif  // DEBUG_MODE
	
    
    return 0;
}




/*!
 * preprocess the original image and obtain the alexnet standard input blob
 * step 1: resize the img to input dimensions (preserving number of channels)
 * step 2: RGB ----> BRG
 * step 3: subtract mean value of different channels  BGR_MEAN = [103,116,123]
 */
int preprocess(Blob<DATA_TYPE> &img, Blob<DATA_TYPE> &data) {
	int old_num = img.num();
	int old_channel = img.channels();
	int old_height = img.height();
	int old_width = img.width();

	int new_num = data.num();
	int new_channel = data.channels();
	int new_height = data.height();
	int new_width = data.width();

	assert(old_num == 1 && new_num == 1);
	assert(old_channel == new_channel);

	// resize operation
	DATA_TYPE scale_H = (DATA_TYPE)old_height / new_height;
	DATA_TYPE scale_W = (DATA_TYPE)old_width / new_width;

	DATA_TYPE w_map_ori_pos, h_map_ori_pos;
	DATA_TYPE xmin, ymin, xmax, ymax;

	for (int c = 0; c < new_channel; ++c) {
		for (int h = 0; h < new_height; ++h) {
			for (int w = 0; w < new_width; ++w) {
				// bilinear interpolation
				w_map_ori_pos = (DATA_TYPE)((w + 0.5)*scale_W - 0.5);
				h_map_ori_pos = (DATA_TYPE)((h + 0.5)*scale_H - 0.5);

				xmin = floor(w_map_ori_pos);
				ymin = floor(h_map_ori_pos);
				xmax = (DATA_TYPE)(xmin + 1.0);
				ymax = (DATA_TYPE)(ymin + 1.0);

				if (xmin < 0.00001) {
					xmin = (DATA_TYPE)(0.0);
					xmax = (DATA_TYPE)(1.0);
				}

				if (xmin >= old_width-1.0) {
					xmin = (DATA_TYPE)(old_width - 1);
					xmax = (DATA_TYPE)(old_width - 1);
				}

				if (ymin < 0.00001) {
					ymin = (DATA_TYPE)(0.0);
					ymax = (DATA_TYPE)(1.0);
				}

				if (ymin >= old_height - 1.0) {
					ymin = (DATA_TYPE)(old_height - 1);
					ymax = (DATA_TYPE)(old_height - 1);
				}
				
				DATA_TYPE value = 
					img.data_at(0, c, (int)ymin, (int)xmin) * (xmax - w_map_ori_pos) * (ymax - h_map_ori_pos) +
					img.data_at(0, c, (int)ymin, (int)xmax) * (w_map_ori_pos - xmin) * (ymax - h_map_ori_pos) +
					img.data_at(0, c, (int)ymax, (int)xmin) * (xmax - w_map_ori_pos) * (h_map_ori_pos - ymin) +
					img.data_at(0, c, (int)ymax, (int)xmax) * (w_map_ori_pos - xmin) * (h_map_ori_pos - ymin);

				//printf("position (%d, %d) (%d,%d,%d,%d): value = %f\n", h, w, (int)xmin, (int)ymin, (int)xmax, (int)ymax, value);
				data.setData(data.offset(0, c, h, w), value);
			}
		}
	}

	// RGB ----> BGR
	DATA_TYPE temp;
	for (int h = 0; h < new_height; ++h) {
		for (int w = 0; w < new_width; ++w) {
			temp = data.data_at(0, 0, h, w);
			data.setData(data.offset(0, 0, h, w), data.data_at(0, 2, h, w));
			data.setData(data.offset(0, 2, h, w), temp);
		}
	}


	// subtract BGR_MEAN = [103,116,123]
#ifdef DEBUG_MODE

	DATA_TYPE bgr_mean[3] = { 103.0, 116.0, 123.0 };
	FILE *wf_data = fopen(DATA_RESULTS, "wb");
	for (int c = 0; c < new_channel; ++c) {
		for (int h = 0; h < new_height; ++h) {
			for (int w = 0; w < new_width; ++w) {
				data.setData(data.offset(0, c, h, w), data.data_at(0, c, h, w) - bgr_mean[c]);
				fprintf(wf_data, "%.4f\n", (data.data_at(0, c, h, w)));
			}
		}
	}
	fclose(wf_data);

#else

	DATA_TYPE bgr_mean[3] = { 103.0, 116.0, 123.0 };
	for (int c = 0; c < new_channel; ++c) {
		for (int h = 0; h < new_height; ++h) {
			for (int w = 0; w < new_width; ++w) {
				data.setData(data.offset(0, c, h, w), data.data_at(0, c, h, w) - bgr_mean[c]);
			}
		}
	}

#endif // DEBUG_MODE

	return 0;
}


/*!
 * convolutional layer implementation:
 * --- input blob: bottom
 * --- kernel blob: kernel
 * --- output blob: top
 * 
 * ATTENTION: each blob may contain bias
 */
int convolutional(Blob<DATA_TYPE> &bottom, Blob<DATA_TYPE> &kernel, Blob<DATA_TYPE> &top, int stride, int pad, const char * dump_file_path) {
	assert(top.channels() == kernel.num());

	for (int kn = 0; kn < kernel.num(); ++kn) {		// loop the kernel numbers
		int top_c = kn;
		int top_h = 0;
		int top_w = 0;

		for (int h = 0 - pad; h + kernel.height() - 1 < bottom.height() + pad; h += stride) {
			top_w = 0;
			for (int w = 0 - pad; w + kernel.width() - 1 < bottom.width() + pad; w += stride) {
				// the up-left point in bottom blob is (h,w)
				DATA_TYPE value = 0.0;

				// calculate the convolutional result
				for (int kc = 0; kc < kernel.channels(); ++kc) {
					for (int kh = 0; kh < kernel.height(); ++kh) {
						for (int kw = 0; kw < kernel.width(); ++kw) {
							DATA_TYPE temp = 0.0;

							if (h + kh >= 0 && h + kh < bottom.height() && w + kw >= 0 && w + kw < bottom.width()) {
								temp = bottom.data_at(0, kc, h + kh, w + kw);
							}

							value += kernel.data_at(kn, kc, kh, kw) * temp;
						}
					}
				}
				value += kernel.bias_at(kn);
				//printf("response value = %f.\n", value);
				top.setData(top.offset(0, top_c, top_h, top_w), ReLU(value));
				
				top_w += 1;
			}
			top_h += 1;
		}

		//printf("complete %d kernel convolution...\n", kn+1);

	}



#ifdef DEBUG_MODE
	FILE *wf_data = fopen(dump_file_path, "wb");
	for (int c = 0; c < top.channels(); ++c) {
		for (int h = 0; h < top.height(); ++h) {
			for (int w = 0; w < top.width(); ++w) {
				fprintf(wf_data, "%.4f\n", (top.data_at(0, c, h, w)));
			}
		}
	}
	fclose(wf_data);
#endif // DEBUG_MODE


	return 0;
}






/*!
* convolutional layer implementation (including group implementation):
* --- input blob: bottom
* --- kernel blob: kernel
* --- output blob: top
*
* ATTENTION: each blob may contain bias
*/
int convolutional_gp(Blob<DATA_TYPE> &bottom, Blob<DATA_TYPE> &kernel, Blob<DATA_TYPE> &top, int stride, int pad, const char * dump_file_path, int group) {
	assert(top.channels() == kernel.num());

	for (int g = 0; g < group; ++g) {
		int bottom_channel_offset = g * (bottom.channels() / group);

		for (int kn = 0; kn < kernel.num()/group; ++kn) {		// loop the kernel numbers
			int top_c = kn + g * (kernel.num() / group);
			int top_h = 0;
			int top_w = 0;
			int kernel_offset = g * (kernel.num() / group);

			for (int h = 0 - pad; h + kernel.height() - 1 < bottom.height() + pad; h += stride) {
				top_w = 0;
				for (int w = 0 - pad; w + kernel.width() - 1 < bottom.width() + pad; w += stride) {
					// the up-left point in bottom blob is (h,w)
					DATA_TYPE value = 0.0;

					// calculate the convolutional result
					for (int kc = 0; kc < kernel.channels(); ++kc) {
						for (int kh = 0; kh < kernel.height(); ++kh) {
							for (int kw = 0; kw < kernel.width(); ++kw) {
								DATA_TYPE temp = 0.0;

								if (h + kh >= 0 && h + kh < bottom.height() && w + kw >= 0 && w + kw < bottom.width()) {
									temp = bottom.data_at(0, bottom_channel_offset + kc, h + kh, w + kw);
								}

								value += kernel.data_at(kn + kernel_offset, kc, kh, kw) * temp;
							}
						}
					}
					value += kernel.bias_at(kn + kernel_offset);
					top.setData(top.offset(0, top_c, top_h, top_w), ReLU(value));

					top_w += 1;
				}
				top_h += 1;
			}
		}
	}






#ifdef DEBUG_MODE
	FILE *wf_data = fopen(dump_file_path, "wb");
	for (int c = 0; c < top.channels(); ++c) {
		for (int h = 0; h < top.height(); ++h) {
			for (int w = 0; w < top.width(); ++w) {
				fprintf(wf_data, "%.4f\n", (top.data_at(0, c, h, w)));
			}
		}
	}
	fclose(wf_data);
#endif // DEBUG_MODE


	return 0;
}

















/*!
 * Alexnet LRN module
 */
int lrn(Blob<DATA_TYPE> &bottom, Blob<DATA_TYPE> &top, double k, int n, double alpha, double beta, const char * dump_file_path) {
	assert(bottom.channels() == top.channels());
	assert(bottom.width() == top.width());
	assert(bottom.height() == top.height());

	for (int c = 0; c < top.channels(); ++c) {
		for (int h = 0; h < top.height(); ++h) {
			for (int w = 0; w < top.width(); ++w) {
				int ch_from = MAX(0, c - n / 2);
				int ch_to = MIN(top.channels() - 1, c + n / 2);

				DATA_TYPE SQUARE_SUM = 0.0;
				for (int scan_c = ch_from; scan_c <= ch_to; ++scan_c) {
					SQUARE_SUM += (DATA_TYPE)pow(bottom.data_at(0, scan_c, h, w), 2.0);
				}
				
				DATA_TYPE denominator = (DATA_TYPE)(pow((SQUARE_SUM * alpha/n + k), beta));			// this is different from the paper
				top.setData(top.offset(0, c, h, w), bottom.data_at(0, c, h, w) / denominator);
			}
		}
	}

#ifdef DEBUG_MODE
	FILE *wf_data = fopen(dump_file_path, "wb");
	for (int c = 0; c < top.channels(); ++c) {
		for (int h = 0; h < top.height(); ++h) {
			for (int w = 0; w < top.width(); ++w) {
				fprintf(wf_data, "%.4f\n", (top.data_at(0, c, h, w)));
			}
		}
	}
	fclose(wf_data);
#endif // DEBUG_MODE


	return 0;
}






/*!
 * pooling
 */
int pooling(Blob<DATA_TYPE> &bottom, Blob<DATA_TYPE> &top, int ks, int stride, int padding, const char * dump_file_path) {
	assert(bottom.channels() == top.channels());

	for (int bottom_c = 0; bottom_c < bottom.channels(); ++bottom_c) {		// loop the bottom channels
		int top_h = 0;
		int top_w = 0;

		for (int bottom_h = 0 - padding; bottom_h + stride - 1 < bottom.height() + padding; bottom_h += stride) {
			top_w = 0;
			for (int bottom_w = 0 - padding; bottom_w + stride - 1 < bottom.width() + padding; bottom_w += stride) {
				// the up-left point in bottom blob is (bottom_h,bottom_w)
				DATA_TYPE value = 0.0;

				// find the maximum value in the ks x ks region with a offset (bottom_h,bottom_w)
				for (int i = bottom_h; i < bottom_h + ks; ++i) {
					for (int j = bottom_w; j < bottom_w + ks; ++j) {
						if (value < bottom.data_at(0, bottom_c, i, j))
							value = bottom.data_at(0, bottom_c, i, j);
					}
				}

				top.setData(top.offset(0, bottom_c, top_h, top_w), value);

				top_w += 1;
			}
			top_h += 1;
		}
	}



#ifdef DEBUG_MODE
	FILE *wf_data = fopen(dump_file_path, "wb");
	for (int c = 0; c < top.channels(); ++c) {
		for (int h = 0; h < top.height(); ++h) {
			for (int w = 0; w < top.width(); ++w) {
				fprintf(wf_data, "%.4f\n", (top.data_at(0, c, h, w)));
			}
		}
	}
	fclose(wf_data);
#endif // DEBUG_MODE


	return 0;
}





/*!
 * fully connection layer
 */
int fc(Blob<DATA_TYPE> &bottom, Blob<DATA_TYPE> &kernel, Blob<DATA_TYPE> &top, const char * dump_file_path, bool runRelu) {
	assert(top.channels() == kernel.num());

	for (int kn = 0; kn < kernel.num(); ++kn) {		// loop the kernel numbers
		int top_c = kn;
		int top_h = 0;
		int top_w = 0;

		DATA_TYPE value = 0.0;

		for (int c = 0; c < bottom.channels(); ++c) {
			for (int h = 0; h < bottom.height(); ++h) {
				for (int w = 0; w < bottom.width(); ++w) {
					int offset = (c*bottom.height() + h)*bottom.width() + w;
					value += bottom.data_at(0, c, h, w) * kernel.data_at(kn, offset);
				}
			}
		}

		value += kernel.bias_at(kn);
		if (runRelu)
			top.setData(top.offset(0, top_c, 0, 0), ReLU(value));
		else
			top.setData(top.offset(0, top_c, 0, 0), value);
	}



#ifdef DEBUG_MODE
	FILE *wf_data = fopen(dump_file_path, "wb");
	for (int c = 0; c < top.channels(); ++c) {
		for (int h = 0; h < top.height(); ++h) {
			for (int w = 0; w < top.width(); ++w) {
				fprintf(wf_data, "%.4f\n", (top.data_at(0, c, h, w)));
			}
		}
	}
	fclose(wf_data);
#endif // DEBUG_MODE


	return 0;
}




/*!
* Softmax function implementation
*/
int softmax(Blob<DATA_TYPE> &bottom, Blob<DATA_TYPE> &top, const char * dump_file_path) {
	assert(bottom.channels() == top.channels());
	int len = top.channels();

	DATA_TYPE sum = 0.0;
	DATA_TYPE value = 0.0;

	for (int i = 0; i < len; ++i) {
		sum += (DATA_TYPE)pow(E, bottom.data_at(0, i, 0, 0));
	}

	for (int i = 0; i < len; ++i) {
		value = (DATA_TYPE)pow(E, bottom.data_at(0,i,0,0)) / sum;
		top.setData(top.offset(0, i, 0, 0), value);
	}

#ifdef DEBUG_MODE
	FILE *wf_data = fopen(dump_file_path, "wb");
	for (int c = 0; c < top.channels(); ++c) {
		for (int h = 0; h < top.height(); ++h) {
			for (int w = 0; w < top.width(); ++w) {
				fprintf(wf_data, "%.4f\n", (top.data_at(0, c, h, w)));
			}
		}
	}
	fclose(wf_data);
#endif // DEBUG_MODE


	return 0;
}


/*!
 * calculate the difference between the differnt frame
 */
long int calc_diff(Blob<DATA_TYPE> &prev_img, Blob<DATA_TYPE> &img){
    int channel = img.channels();
    int height = img.height();
    int width = img.width();

    long int diff = 0;
    
    DATA_TYPE pixel_diff;

	for (int c = 0; c < channel; c++) {
		for (int h = 0; h < height; h++) {
			for (int w = 0; w < width; w++) {
                pixel_diff = prev_img.data_at(0, c, h, w) - img.data_at(0, c, h, w);
                diff += ABS(pixel_diff);
			}
		}
	}
    
    return diff;
}






