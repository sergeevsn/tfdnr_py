# TFD Noise Rejection

This repository contains a Python library that implements a Time-Frequency Domain (TFD) noise rejection method for seismic data. The function is designed to remove noise from seismic traces using either a median filter or an adjustment window approach. The filtering is performed after STFT transform, only using amplitude spectrum, preserving phase spectrum. 

## Main Function: `tfd_noise_rejection`

### Description
The `tfd_noise_rejection` function processes 2D seismic data and removes noise based on the specified method. It supports two modes for noise rejection:
1. **Median Filter**: Uses an adaptive threshold based on the median amplitude of traces.
2. **Adjustment Window**: Sets a threshold based on user-defined adjustment points.

### Parameters
- **data (np.array)**: 2D array where each row represents a seismic trace, and each column represents a time sample.
- **stft_window_ms (float)**: Length of the STFT window in milliseconds.
- **dt (float)**: Sampling interval in seconds.
- **trace_aperture (int, optional)**: Aperture for median filter across traces.
- **threshold_multiplier (float)**: Multiplier for the threshold value.
- **method (str)**: Method to use ('median_filter' or 'adjustment_window').
- **threshold_method (str, optional)**: For 'median_filter', specifies how to compute the threshold ('global' or 'per_frequency').
- **median_window_ms (float, optional)**: Median filter window size in milliseconds.
- **min_noise_duration_ms (float, optional)**: Minimum duration of noise for removal.
- **adjustment_points (list, optional)**: List of [trace_index, time_in_seconds] pairs defining the adjustment line.
- **window_type (str, optional)**: Defines whether to use 'above' or 'below' the adjustment line.
- **adj_wnd_mode (str, optional)**: Mode for calculating the threshold ('median', 'mean', or 'RMS').

### Returns
- **np.array**: Filtered seismic data after noise rejection.

## Get the code and run example notebook

```sh
git clone https://github.com/sergeevsn/tfdnr_py.git
cd tfdnr_py
pip install -r requirements.txt
jupyter notebook test.ipynb
```
