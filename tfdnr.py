import numpy as np
from scipy.signal import stft, istft
from scipy.ndimage import median_filter, binary_opening

def next_power_of_two(n):
    """Returns the smallest power of two greater than or equal to n"""
    return 1 << (n - 1).bit_length()

def tfd_noise_rejection(data, stft_window_ms, dt, trace_aperture, threshold_multiplier,
                        threshold_method='global', median_window_ms=None,
                        min_noise_duration_ms=None, threshold_mode='multiplier'):
    """
    Function to remove noise from data using Time-Frequency Domain (TFD) approach with advanced options.
    
    Args:
        data (np.array): Input data of shape (n_traces, n_samples).
        stft_window_ms (float): Duration of STFT window in milliseconds.
        dt (float): Sampling interval of the signal in seconds.
        trace_aperture (int): Aperture for median filtering across traces.
        threshold_multiplier (float): Multiplier for the threshold.
        threshold_method (str, optional): Method to calculate the threshold ('global' or 'per_frequency').
        median_window_ms (float, optional): Size of moving window for median filtering in milliseconds.
        min_noise_duration_ms (float, optional): Minimum duration (in milliseconds) of noise event to remove. Protects short peaks of useful signal.
            If None (default), it is not used.
        threshold_mode (str, optional): Mode to calculate the threshold. 'multiplier' (default): Threshold = Median * Multiplier.
            'statistical': Threshold = Median + Multiplier * MAD (Mean Absolute Deviation).
    
    Returns:
        np.array: Filtered data.
    """
    n_traces, n_samples_original = data.shape
    fs = 1.0 / dt
    
    # --- Convert parameters and checks ---
    stft_window_sec = stft_window_ms / 1000.0
    nperseg = next_power_of_two(int(stft_window_sec / dt))
    trace_aperture = int(trace_aperture)
    
    if nperseg >= n_samples_original: 
        return data
    
    if trace_aperture >= n_traces: 
        return data
    
    noverlap = nperseg // 2
    f, t, stft_array = stft(data, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap,
                            boundary='even', padded=True, axis=1)
    
    amplitudes = np.abs(stft_array)
    phases = np.angle(stft_array)
    
    # --- Calculate base median and MAD ---
    median_across_traces = np.median(amplitudes, axis=0)
    
    # 1. Median smoothing in time if a window is specified
    if median_window_ms is not None and median_window_ms > 0:
        hop_length = nperseg - noverlap
        dt_stft = hop_length * dt
        median_window_in_stft_samples = int(round((median_window_ms / 1000.0) / dt_stft))
        
        if median_window_in_stft_samples > 1:
            if median_window_in_stft_samples % 2 == 0:
                median_window_in_stft_samples += 1
            median_base = median_filter(median_across_traces, size=(1, median_window_in_stft_samples), mode='reflect')
        else:
            median_base = median_across_traces
    else:
        median_base = median_across_traces
    
    threshold_map = median_base * threshold_multiplier
    
    # Apply 'global' or 'per_frequency' to the threshold map
    if threshold_method == 'global':
        threshold = np.median(threshold_map)
    else:  # per_frequency
        threshold = np.median(threshold_map, axis=1)
        threshold = threshold[np.newaxis, :, np.newaxis]
    
    # --- Create initial noise mask ---
    initial_mask = amplitudes > threshold
    
    # --- New Feature 2: Filter the mask by temporal coherence ---
    final_mask = initial_mask
    if min_noise_duration_ms is not None and min_noise_duration_ms > 0:
        if 'dt_stft' not in locals():  # Calculate if not calculated before
            hop_length = nperseg - noverlap
            dt_stft = hop_length * dt
            duration_in_stft_samples = int(round((min_noise_duration_ms / 1000.0) / dt_stft))
        
        if duration_in_stft_samples > 1:
            # Create structuring element: line along the time axis
            structure = np.ones((1, 1, duration_in_stft_samples))
            # Binary opening operation removes all objects (True) that are smaller than the structure
            final_mask = binary_opening(initial_mask, structure=structure)
    
    # --- Apply final mask and filtering ---
    median_filtered_amp = median_filter(amplitudes, size=(2 * trace_aperture + 1, 1, 1), mode='nearest')
    amplitudes[final_mask] = median_filtered_amp[final_mask]
    
    # --- iSTFT ---
    _, filtered_data = istft(amplitudes * np.exp(1j * phases), fs=fs, window='hann', nperseg=nperseg,
                            noverlap=noverlap, boundary=True, time_axis=-1, input_onesided=True)
    
    if filtered_data.shape[1] > n_samples_original:
        filtered_data = filtered_data[:, :n_samples_original]
    elif filtered_data.shape[1] < n_samples_original:
        pad_width = ((0, 0), (0, n_samples_original - filtered_data.shape[1]))
        filtered_data = np.pad(filtered_data, pad_width, mode='constant')
    
    return filtered_data.astype(data.dtype)