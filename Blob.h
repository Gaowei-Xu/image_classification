#ifndef __BLOB_H__
#define __BLOB_H__


#include <iostream>

template <typename Dtype>
class Blob {
public:
	Blob(const int num, const int channels, const int height, const int width, const bool contain_bias) {
		_num = num;
		_channels = channels;
		_height = height;
		_width = width;
		_contain_bias = contain_bias;

		if (_num > 0 && _channels > 0 && _height > 0 && _width > 0)
			_data = new Dtype[_num * _channels * _height * _width]();
		else
			_data = NULL;

		if (_contain_bias)
			_bias = new Dtype[_num]();
		else
			_bias = NULL;
	}


	Blob & operator = (const Blob<Dtype> &other) {
		if (this == &other)
			return *this;

		// free the original space
		if (_data != NULL) {
			delete _data;
			_data = NULL;
		}

		if (_bias != NULL) {
			delete _bias;
			_bias = NULL;
		}

		_height = other._height;
		_width = other._width;
		_contain_bias = other._contain_bias;

		if (_num > 0 && _channels > 0 && _height > 0 && _width > 0){
			_data = new Dtype[_num * _channels * _height * _width]();
            
            // copy data from other._data to this._data
            for( int i = 0; i < _num * _channels * _height * _width; i++){
                _data[i] = other._data[i];
            }
        }
		else
			_data = NULL;

		if (_contain_bias){
			_bias = new Dtype[_num]();
            
            // copy data from other._bias to this._bias
            for (int i = 0; i < _num; ++i)
                _bias[i] = other._bias[i];
        }
		else
			_bias = NULL;
	}

	Blob(const Blob<Dtype> &other) {
		_channels = other._channels;
		_height = other._height;
		_width = other._width;
		_contain_bias = other._contain_bias;

		if (_num > 0 && _channels > 0 && _height > 0 && _width > 0){
			_data = new Dtype[_num * _channels * _height * _width]();
            // copy data from other._data to this._data
            for( int i = 0; i < _num * _channels * _height * _width; i++){
                _data[i] = other._data[i];
            }
        }
		else
			_data = NULL;

		if (_contain_bias){
			_bias = new Dtype[_num]();
            // copy data from other._bias to this._bias
            for (int i = 0; i < _num; ++i)
                _bias[i] = other._bias[i];
        }
		else
			_bias = NULL;
	}


	~Blob() {
		if (_data != NULL) {
			delete _data;
			_data = NULL;
		}

		if (_bias != NULL) {
			delete _bias;
			_bias = NULL;
		}
	}


	inline Dtype data_at(const int n = 0, const int c = 0, const int h = 0, const int w = 0) {
		return _data[offset(n, c, h, w)];
	}

	inline Dtype data_at(const int index) {
		return _data[index];
	}

	inline Dtype bias_at(const int n) {
		return _bias[n];
	}


	inline int offset(const int n = 0, const int c = 0, const int h = 0, const int w = 0) const {
		int offset = 0;
		offset = ((n*_channels + c)*_height + h)*_width + w;
		return offset;
	}

	inline int num() { return _num; }
	inline int channels() { return _channels; }
	inline int height() { return _height; }
	inline int width() { return _width; }
	inline bool contain_bias() { return _contain_bias; }

	inline Dtype * getDataPtr() { return _data; }
	inline Dtype * getBiasPtr() { return _bias; }

	inline void setData(const int n, const int c, const int h, const int w, Dtype value) {
		_data[offset(n, c, h, w)] = value;
	}

	inline void setData(const int index, Dtype value) {
		_data[index] = value;
	}


	inline void setBias(const int n, Dtype value) {
		_bias[n] = value;
	}



private:
	bool _contain_bias;

	int _num;
	int _channels;
	int _height;
	int _width;

	Dtype *_data;
	Dtype *_bias;
};


#endif // !__BLOB_H__










