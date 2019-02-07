#include "Blob.h"
#include "config.h"
#include "utils.h"
#include <iostream>
#include <assert.h>

//#include <Windows.h>
#include <time.h>
#include <sys/time.h>

using namespace std;

/*!
 * print the probalility of different classes
 */
void printResult(Blob<DATA_TYPE> &top){
    assert(top.height() == 1 && top.width() == 1);

    const char *desp[] = {  "c0: correct posture",
                            "c1: body lean (left, right)",
                            "c2: body forward lean",
                            "c3: far from camera",
                            "c4: head tilt (left, right)",
                            "c5: head tilt (up, down)",
                            "c6: head tilt (forth, back)",
                            "c7: hand holding head",
                            "c8: lie prone on the desk",
                            "c9: nobody" };


    printf("Probability: \n");
	for (int c = 0; c < top.channels(); ++c) {
        printf("%.4f --- %s \n", (top.data_at(0, c, 0, 0)), desp[c]);
    }
    printf("\n");
}


Blob<DATA_TYPE> conv1_kernel(24, 3, 7, 7, true);		// num, channel, height, width, contain_bias
Blob<DATA_TYPE> conv2_kernel(32, 24, 5, 5, true);		// num, channel, height, width, contain_bias
Blob<DATA_TYPE> conv3_kernel(48, 32, 3, 3, true);		// num, channel, height, width, contain_bias
Blob<DATA_TYPE> conv4_kernel(48, 48, 3, 3, true);		// num, channel, height, width, contain_bias
Blob<DATA_TYPE> conv5_kernel(32, 48, 3, 3, true);		// num, channel, height, width, contain_bias
Blob<DATA_TYPE> fc6_kernel(1024, 480, 1, 1, true);		// num, channel, height, width, contain_bias
Blob<DATA_TYPE> fc7_kernel(256, 1024, 1, 1, true);		// num, channel, height, width, contain_bias
Blob<DATA_TYPE> fc8_kernel(10, 256, 1, 1, true);		// num, channel, height, width, contain_bias


Blob<DATA_TYPE> prev_img(1, 3, 480, 640, false);		// num, channel, height, width, contain_bias

Blob<DATA_TYPE> img(1, 3, 480, 640, false);				// num, channel, height, width, contain_bias
Blob<DATA_TYPE> data(1, 3, 96, 128, false);				// num, channel, height, width, contain_bias
Blob<DATA_TYPE> conv1(1, 24, 30, 41, false);			// num, channel, height, width, contain_bias
Blob<DATA_TYPE> pool1(1, 24, 15, 20, false);			// num, channel, height, width, contain_bias
Blob<DATA_TYPE> conv2(1, 32, 15, 20, false);			// num, channel, height, width, contain_bias
Blob<DATA_TYPE> pool2(1, 32, 7, 10, false);			    // num, channel, height, width, contain_bias
Blob<DATA_TYPE> conv3(1, 48, 7, 10, false);			    // num, channel, height, width, contain_bias
Blob<DATA_TYPE> conv4(1, 48, 7, 10, false);			    // num, channel, height, width, contain_bias
Blob<DATA_TYPE> conv5(1, 32, 7, 10, false);			    // num, channel, height, width, contain_bias
Blob<DATA_TYPE> pool5(1, 32, 3, 5, false);				// num, channel, height, width, contain_bias
Blob<DATA_TYPE> fc6(1, 1024, 1, 1, false);				// num, channel, height, width, contain_bias
Blob<DATA_TYPE> fc7(1, 256, 1, 1, false);				// num, channel, height, width, contain_bias
Blob<DATA_TYPE> fc8(1, 10, 1, 1, false);				// num, channel, height, width, contain_bias
Blob<DATA_TYPE> output(1, 10, 1, 1, false);				// num, channel, height, width, contain_bias

long int frame_diff_thres = 0;

void detector_init()
{
	// Step 2: init model parameters
	int init_status = INIT_FAILED;
	init_status = load_net_para(conv1_kernel, CONV1_FILTER_PATH, CONV1_BIAS_PATH);
	assert(init_status == INIT_SUCCESS);

	init_status = load_net_para(conv2_kernel, CONV2_FILTER_PATH, CONV2_BIAS_PATH);
	assert(init_status == INIT_SUCCESS);

	init_status = load_net_para(conv3_kernel, CONV3_FILTER_PATH, CONV3_BIAS_PATH);
	assert(init_status == INIT_SUCCESS);

	init_status = load_net_para(conv4_kernel, CONV4_FILTER_PATH, CONV4_BIAS_PATH);
	assert(init_status == INIT_SUCCESS);

	init_status = load_net_para(conv5_kernel, CONV5_FILTER_PATH, CONV5_BIAS_PATH);
	assert(init_status == INIT_SUCCESS);
	
	init_status = load_net_para(fc6_kernel, FC6_FILTER_PATH, FC6_BIAS_PATH);
	assert(init_status == INIT_SUCCESS);

	init_status = load_net_para(fc7_kernel, FC7_FILTER_PATH, FC7_BIAS_PATH);
	assert(init_status == INIT_SUCCESS);

	init_status = load_net_para(fc8_kernel, FC8_FILTER_PATH, FC8_BIAS_PATH);
	assert(init_status == INIT_SUCCESS);
	
    init_status = load_threshold(FRAME_DIFF_THRES_PATH, &frame_diff_thres);
	assert(init_status == INIT_SUCCESS);
}

void detector_load_image()
{
	// Step 4: forward AlexNet to calculate the output 1-D results
	int status = 0;

	// Step 4-1: successfully
	//status = loadImagePPM("./test_images/01.ppm", img);
	prev_img = img;
    
    status = loadImageJPG("./test_images/demo.jpg", img);
	if (status != 0) {
		LOG(GDERROR, "failed to load ppm-format image.");
	}
}

void detector_process_image()
{
	long start, end;
    struct timeval t_start, t_conv1, t_conv2, t_conv3, t_conv4, t_conv5, t_end;
    gettimeofday(&t_start, NULL);

    int status = 0;
	// Step 4-2: successfully
	status = preprocess(img, data);
	if (status != 0) {
		LOG(GDERROR, "failed to preprocess image.");
	}

	// Step 4-3: conv1 layer (stride = 2, padding = 0)
	status = convolutional(data, conv1_kernel, conv1, 3, 0, CONV1_RESULTS);
	if (status != 0) {
		LOG(GDERROR, "failed to execute conv1.");
	}

	// Step 4-5: pooling (kernel size = 3, stride = 2, padding = 0)
	status = pooling(conv1, pool1, 3, 2, 0, POOL1_RESULTS);
	if (status != 0) {
		LOG(GDERROR, "failed to execute pool1.");
	}

	// Step 4-6: conv2 layer (stride = 1, padding = 2)
	status = convolutional(pool1, conv2_kernel, conv2, 1, 2, CONV2_RESULTS);
	if (status != 0) {
		LOG(GDERROR, "failed to execute conv2.");
	}

	// Step 4-8: pooling (kernel size = 3, stride = 2, padding = 0)
	status = pooling(conv2, pool2, 3, 2, 0, POOL2_RESULTS);
	if (status != 0) {
		LOG(GDERROR, "failed to execute pool2.");
	}

	// Step 4-9: conv3 layer (stride = 1, padding = 1)
	status = convolutional(pool2, conv3_kernel, conv3, 1, 1, CONV3_RESULTS);
	if (status != 0) {
		LOG(GDERROR, "failed to execute conv3.");
	}

	// Step 4-10: conv4 layer (stride = 1, padding = 1)
	status = convolutional(conv3, conv4_kernel, conv4, 1, 1, CONV4_RESULTS);
	if (status != 0) {
		LOG(GDERROR, "failed to execute conv4.");
	}

	// Step 4-11: conv5 layer (stride = 1, padding = 1)
	status = convolutional(conv4, conv5_kernel, conv5, 1, 1, CONV5_RESULTS);
	if (status != 0) {
		LOG(GDERROR, "failed to execute conv5.");
	}

	// Step 4-12: pool5 layer (kernel size = 3, stride = 2, padding = 0)
	status = pooling(conv5, pool5, 3, 2, 0, POOL5_RESULTS);
	if (status != 0) {
		LOG(GDERROR, "failed to execute pool5.");
	}
    
    
    gettimeofday(&t_conv5, NULL);
    start = ((long)t_start.tv_sec)*1000+(long)t_start.tv_usec/1000;
    end = ((long)t_conv5.tv_sec)*1000+(long)t_conv5.tv_usec/1000;
    printf("Conv cost time: %ld ms;\t", end-start);
    

	// Step 4-13: fc6 layer
	status = fc(pool5, fc6_kernel, fc6, FC6_RESULTS, true);
	if (status != 0) {
		LOG(GDERROR, "failed to execute fc6.");
	}

	// Step 4-14: fc7 layer
	status = fc(fc6, fc7_kernel, fc7, FC7_RESULTS, true);
	if (status != 0) {
		LOG(GDERROR, "failed to execute fc7.");
	}

	// Step 4-15: fc8 layer
	status = fc(fc7, fc8_kernel, fc8, FC8_RESULTS, false);
	if (status != 0) {
		LOG(GDERROR, "failed to execute fc8.");
	}

	// Step 4-16: softmax layer
	status = softmax(fc8, output, OUTPUT_RESULTS);
    
    long int diff = calc_diff(prev_img, img);
    if (diff < frame_diff_thres)
        output.setData(output.offset(0, 9, 0, 0), 1.0);   

    gettimeofday(&t_end, NULL);
    start = ((long)t_conv5.tv_sec)*1000+(long)t_conv5.tv_usec/1000;
    end = ((long)t_end.tv_sec)*1000+(long)t_end.tv_usec/1000;
    printf("fc cost time: %ld ms\n", end-start);
    
    printf("difference between two frames = %ld\n", diff);
    printResult(output);

}



int main()
{
	detector_init();
    while (1){
        detector_load_image();
	    detector_process_image();
    }
}

#if 0
int main() {
	// Step 1: allocate memory for caffemodel parameters of different layers
	Blob<DATA_TYPE> conv1_kernel(24, 3, 7, 7, true);		// num, channel, height, width, contain_bias
	Blob<DATA_TYPE> conv2_kernel(32, 24, 5, 5, true);		// num, channel, height, width, contain_bias
	Blob<DATA_TYPE> conv3_kernel(48, 32, 3, 3, true);		// num, channel, height, width, contain_bias
	Blob<DATA_TYPE> conv4_kernel(48, 48, 3, 3, true);		// num, channel, height, width, contain_bias
	Blob<DATA_TYPE> conv5_kernel(32, 48, 3, 3, true);		// num, channel, height, width, contain_bias
	Blob<DATA_TYPE> fc6_kernel(1024, 480, 1, 1, true);		// num, channel, height, width, contain_bias
	Blob<DATA_TYPE> fc7_kernel(256, 1024, 1, 1, true);		// num, channel, height, width, contain_bias
	Blob<DATA_TYPE> fc8_kernel(10, 256, 1, 1, true);		// num, channel, height, width, contain_bias

	// Step 2: init model parameters
	int init_status = INIT_FAILED;
	init_status = load_net_para(conv1_kernel, CONV1_FILTER_PATH, CONV1_BIAS_PATH);
	assert(init_status == INIT_SUCCESS);

	init_status = load_net_para(conv2_kernel, CONV2_FILTER_PATH, CONV2_BIAS_PATH);
	assert(init_status == INIT_SUCCESS);

	init_status = load_net_para(conv3_kernel, CONV3_FILTER_PATH, CONV3_BIAS_PATH);
	assert(init_status == INIT_SUCCESS);

	init_status = load_net_para(conv4_kernel, CONV4_FILTER_PATH, CONV4_BIAS_PATH);
	assert(init_status == INIT_SUCCESS);

	init_status = load_net_para(conv5_kernel, CONV5_FILTER_PATH, CONV5_BIAS_PATH);
	assert(init_status == INIT_SUCCESS);
	
	init_status = load_net_para(fc6_kernel, FC6_FILTER_PATH, FC6_BIAS_PATH);
	assert(init_status == INIT_SUCCESS);

	init_status = load_net_para(fc7_kernel, FC7_FILTER_PATH, FC7_BIAS_PATH);
	assert(init_status == INIT_SUCCESS);

	init_status = load_net_para(fc8_kernel, FC8_FILTER_PATH, FC8_BIAS_PATH);
	assert(init_status == INIT_SUCCESS);

	// Step 3: allocate memory for Blobs of different layers
	Blob<DATA_TYPE> img(1, 3, 480, 640, false);				// num, channel, height, width, contain_bias
	Blob<DATA_TYPE> data(1, 3, 96, 128, false);			// num, channel, height, width, contain_bias
	Blob<DATA_TYPE> conv1(1, 24, 30, 41, false);			// num, channel, height, width, contain_bias
	Blob<DATA_TYPE> pool1(1, 24, 15, 20, false);			// num, channel, height, width, contain_bias
	Blob<DATA_TYPE> conv2(1, 32, 15, 20, false);			// num, channel, height, width, contain_bias
	Blob<DATA_TYPE> pool2(1, 32, 7, 10, false);			    // num, channel, height, width, contain_bias
	Blob<DATA_TYPE> conv3(1, 48, 7, 10, false);			    // num, channel, height, width, contain_bias
	Blob<DATA_TYPE> conv4(1, 48, 7, 10, false);			    // num, channel, height, width, contain_bias
	Blob<DATA_TYPE> conv5(1, 32, 7, 10, false);			    // num, channel, height, width, contain_bias
	Blob<DATA_TYPE> pool5(1, 32, 3, 5, false);				// num, channel, height, width, contain_bias
	Blob<DATA_TYPE> fc6(1, 1024, 1, 1, false);				// num, channel, height, width, contain_bias
	Blob<DATA_TYPE> fc7(1, 256, 1, 1, false);				// num, channel, height, width, contain_bias
	Blob<DATA_TYPE> fc8(1, 10, 1, 1, false);				// num, channel, height, width, contain_bias
	Blob<DATA_TYPE> output(1, 10, 1, 1, false);				// num, channel, height, width, contain_bias


	//while (true) {
        //DWORD start, stop;
		//start = GetTickCount();
        long start, end;
        struct timeval t_start, t_conv1, t_conv2, t_conv3, t_conv4, t_conv5, t_end;
        gettimeofday(&t_start, NULL);


		// Step 4: forward AlexNet to calculate the output 1-D results
		int status = 0;

		// Step 4-1: successfully
		//status = loadImagePPM("./test_images/01.ppm", img);
		status = loadImageJPG("./test_images/02.jpg", img);
		if (status != 0) {
			LOG(GDERROR, "failed to load ppm-format image.");
		}

		// Step 4-2: successfully
		status = preprocess(img, data);
		if (status != 0) {
			LOG(GDERROR, "failed to preprocess image.");
		}

		// Step 4-3: conv1 layer (stride = 2, padding = 0)
		status = convolutional(data, conv1_kernel, conv1, 3, 0, CONV1_RESULTS);
		if (status != 0) {
			LOG(GDERROR, "failed to execute conv1.");
		}

		// Step 4-5: pooling (kernel size = 3, stride = 2, padding = 0)
		status = pooling(conv1, pool1, 3, 2, 0, POOL1_RESULTS);
		if (status != 0) {
			LOG(GDERROR, "failed to execute pool1.");
		}

		// Step 4-6: conv2 layer (stride = 1, padding = 2)
		status = convolutional(pool1, conv2_kernel, conv2, 1, 2, CONV2_RESULTS);
		if (status != 0) {
			LOG(GDERROR, "failed to execute conv2.");
		}

		// Step 4-8: pooling (kernel size = 3, stride = 2, padding = 0)
		status = pooling(conv2, pool2, 3, 2, 0, POOL2_RESULTS);
		if (status != 0) {
			LOG(GDERROR, "failed to execute pool2.");
		}

		// Step 4-9: conv3 layer (stride = 1, padding = 1)
		status = convolutional(pool2, conv3_kernel, conv3, 1, 1, CONV3_RESULTS);
		if (status != 0) {
			LOG(GDERROR, "failed to execute conv3.");
		}

		// Step 4-10: conv4 layer (stride = 1, padding = 1)
		status = convolutional(conv3, conv4_kernel, conv4, 1, 1, CONV4_RESULTS);
		if (status != 0) {
			LOG(GDERROR, "failed to execute conv4.");
		}

		// Step 4-11: conv5 layer (stride = 1, padding = 1)
		status = convolutional(conv4, conv5_kernel, conv5, 1, 1, CONV5_RESULTS);
		if (status != 0) {
			LOG(GDERROR, "failed to execute conv5.");
		}

		// Step 4-12: pool5 layer (kernel size = 3, stride = 2, padding = 0)
		status = pooling(conv5, pool5, 3, 2, 0, POOL5_RESULTS);
		if (status != 0) {
			LOG(GDERROR, "failed to execute pool5.");
		}
        
        
        gettimeofday(&t_conv5, NULL);
        start = ((long)t_start.tv_sec)*1000+(long)t_start.tv_usec/1000;
        end = ((long)t_conv5.tv_sec)*1000+(long)t_conv5.tv_usec/1000;
        printf("Conv cost time: %ld ms;\t", end-start);
        

		// Step 4-13: fc6 layer
		status = fc(pool5, fc6_kernel, fc6, FC6_RESULTS, true);
		if (status != 0) {
			LOG(GDERROR, "failed to execute fc6.");
		}

		// Step 4-14: fc7 layer
		status = fc(fc6, fc7_kernel, fc7, FC7_RESULTS, true);
		if (status != 0) {
			LOG(GDERROR, "failed to execute fc7.");
		}

		// Step 4-15: fc8 layer
		status = fc(fc7, fc8_kernel, fc8, FC8_RESULTS, false);
		if (status != 0) {
			LOG(GDERROR, "failed to execute fc8.");
		}

		// Step 4-16: softmax layer
		status = softmax(fc8, output, OUTPUT_RESULTS);

        gettimeofday(&t_end, NULL);
        start = ((long)t_conv5.tv_sec)*1000+(long)t_conv5.tv_usec/1000;
        end = ((long)t_end.tv_sec)*1000+(long)t_end.tv_usec/1000;
        printf("fc cost time: %ld ms\n", end-start);
        
        printResult(output);

        


	//}

	return 0;
}


#endif
