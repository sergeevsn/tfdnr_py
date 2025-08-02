# TFD Noise Rejection Function

TFD Noise Rejection is a Python function designed to remove noise from seismic data using the Time-Frequency Domain (TFD) approach. This library provides advanced options for noise reduction, including median filtering across traces and threshold-based filtering.

## Get the code and run example notebook

```sh
git clone https://github.com/sergeevsn/tfdnr_py.git
cd tfdnr_py
pip install -r requirements.txt
jupyter notebook test.ipynb
```

## Function Documentation

### `tfd_noise_rejection`

#### Parameters

- **data** (`np.array`): Input data of shape `(n_traces, n_samples)`.
- **stft_window_ms** (`float`): Duration of STFT window in milliseconds.
- **dt** (`float`): Sampling interval of the signal in seconds.
- **trace_aperture** (`int`): Aperture for median filtering across traces.
- **threshold_multiplier** (`float`): Multiplier for the threshold.
- **threshold_method** (`str`, optional): Method to calculate the threshold ('global' or 'per_frequency'). Default is `'global'`.
- **median_window_ms** (`float`, optional): Size of moving window for median filtering in milliseconds. If `None`, median filtering is not applied. Default is `None`.
- **min_noise_duration_ms** (`float`, optional): Minimum duration (in milliseconds) of noise event to remove. Protects short peaks of useful signal. If `None` (default), it is not used.
- **threshold_mode** (`str`, optional): Mode to calculate the threshold. 'multiplier' (default): Threshold = Median * Multiplier. 'statistical': Threshold = Median + Multiplier * MAD (Mean Absolute Deviation).

#### Returns

- **np.array**: Filtered data of shape `(n_traces, n_samples)`.
