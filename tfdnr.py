import numpy as np
import warnings
from scipy.signal import stft, istft
from scipy.ndimage import median_filter, binary_opening
from scipy.interpolate import interp1d


def next_power_of_two(n):
    """Returns the smallest power of two greater than or equal to n"""
    return 1 << (n - 1).bit_length()


def tfd_noise_rejection(data, stft_window_ms, dt, trace_aperture=None, threshold_multiplier=1.0,
                        method='median_filter',
                        # Parameters for method='median_filter'
                        threshold_method='global', median_window_ms=None, min_noise_duration_ms=None,
                        # Parameters for method='adjustment_window'
                        adjustment_points=None, window_type='above', adj_wnd_mode='median'):
    """
    Remove noise from seismic data using a TFD (Time-Frequency Domain) approach.
    Supports two modes:
      - 'median_filter': adaptive threshold based on median across traces.
      - 'adjustment_window': threshold calculated from a specified region (above/below a line).
    
    Args:
        data (np.array): Input data (n_traces, n_samples).
        stft_window_ms (float): STFT window length in milliseconds.
        dt (float): Sampling interval in seconds.
        trace_aperture (int, optional): Aperture for median filtering across traces.
        threshold_multiplier (float): Multiplier for the threshold.
        method (str): 'median_filter' or 'adjustment_window'.
        threshold_method (str): 'global' or 'per_frequency' (only for 'median_filter').
        median_window_ms (float): Time window for median filtering (ms).
        min_noise_duration_ms (float): Minimum noise duration (ms) to avoid removing useful signal.
        adjustment_points (list): List of [trace_number, time_in_seconds] defining a line.
        window_type (str): 'above' — use data above the line (towards t=0), 'below' — below (towards tmax).
        adj_wnd_mode (str): Method to compute threshold: 'median', 'mean', 'RMS'.

    Returns:
        np.array: Filtered data.
    """
    # --- Input validation ---
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array (n_traces, n_samples)")
    if stft_window_ms <= 0:
        raise ValueError("stft_window_ms must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if threshold_multiplier <= 0:
        raise ValueError("threshold_multiplier must be positive")
    if method not in ['median_filter', 'adjustment_window']:
        raise ValueError("method must be 'median_filter' or 'adjustment_window'")
    if threshold_method not in ['global', 'per_frequency']:
        raise ValueError("threshold_method must be 'global' or 'per_frequency'")
    if window_type not in ['above', 'below']:
        raise ValueError("window_type must be 'above' or 'below'")
    if adj_wnd_mode not in ['median', 'mean', 'RMS']:
        raise ValueError("adj_wnd_mode must be 'median', 'mean', or 'RMS'")
    if median_window_ms is not None and median_window_ms < 0:
        raise ValueError("median_window_ms must be non-negative")
    if min_noise_duration_ms is not None and min_noise_duration_ms < 0:
        raise ValueError("min_noise_duration_ms must be non-negative")

    n_traces, n_samples_original = data.shape
    fs = 1.0 / dt

    # --- STFT Parameters ---
    stft_window_sec = stft_window_ms / 1000.0
    nperseg = next_power_of_two(int(stft_window_sec / dt))
    noverlap = nperseg // 2

    if nperseg >= n_samples_original:
        print("Warning: STFT window is too large compared to signal length. Returning original data.")
        return data

    # Check maximum median filter window duration
    if median_window_ms is not None:
        max_duration_ms = n_samples_original * dt * 1000
        if median_window_ms > max_duration_ms:
            print(f"Warning: median_window_ms ({median_window_ms}) is larger than signal duration ({max_duration_ms:.1f}ms)")

    f, t, stft_array = stft(data, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap,
                            boundary='even', padded=True, axis=1)
    amplitudes = np.abs(stft_array)
    phases = np.angle(stft_array)
    n_freq, n_time_stft = amplitudes.shape[1], amplitudes.shape[2]
    hop_length = nperseg - noverlap
    dt_stft = hop_length * dt

    # --- Mode 1: Median filter ---
    if method == 'median_filter':
        if trace_aperture is None:
            raise ValueError("trace_aperture must be specified for 'median_filter' method.")
        if trace_aperture < 0:
            raise ValueError("trace_aperture must be non-negative")
        if trace_aperture >= n_traces:
            print("Warning: trace_aperture is too large. Returning original data.")
            return data

        # Compute median across traces
        median_across_traces = np.median(amplitudes, axis=0)

        # Median filtering in time
        if median_window_ms is not None and median_window_ms > 0:
            median_window_in_stft_samples = int(round((median_window_ms / 1000.0) / dt_stft))
            if median_window_in_stft_samples > 1:
                if median_window_in_stft_samples % 2 == 0:
                    median_window_in_stft_samples += 1
                # Limit window size to signal length
                median_window_in_stft_samples = min(median_window_in_stft_samples, n_time_stft)
                median_base = median_filter(median_across_traces, size=(1, median_window_in_stft_samples), mode='reflect')
            else:
                median_base = median_across_traces
        else:
            median_base = median_across_traces

        # Threshold map
        threshold_map = median_base * threshold_multiplier

        # Compute threshold
        if threshold_method == 'global':
            threshold = np.median(threshold_map)
        elif threshold_method == 'per_frequency':          
            threshold = np.median(threshold_map, axis=1)[:, np.newaxis]
        else:
            raise ValueError(f"Unknown threshold_method: {threshold_method}")

        # Create initial noise mask
        initial_mask = amplitudes > threshold
        final_mask = initial_mask.copy()

        # Morphological filtering to remove short outliers
        if min_noise_duration_ms is not None and min_noise_duration_ms > 0:
            duration_in_stft_samples = int(round((min_noise_duration_ms / 1000.0) / dt_stft))
            if duration_in_stft_samples > 1:
                # Limit structuring element size
                duration_in_stft_samples = min(duration_in_stft_samples, n_time_stft)
                structure = np.ones((1, 1, duration_in_stft_samples))
                final_mask = binary_opening(initial_mask, structure=structure)

        # Apply median filtering across traces
        if trace_aperture > 0:
            aperture_size = min(2 * trace_aperture + 1, n_traces)
            median_filtered_amp = median_filter(amplitudes, size=(aperture_size, 1, 1), mode='nearest')
            amplitudes = np.where(final_mask, median_filtered_amp, amplitudes)

    # --- Mode 2: Adjustment window ---
    elif method == 'adjustment_window':
        # Create adjustment line
        if adjustment_points is None:
            adjustment_line = np.full(n_traces, 0, dtype=int)  # entire trace considered "below"
        else:
            adjustment_points = np.array(adjustment_points)
            # Validate adjustment_points
            if adjustment_points.ndim != 2 or adjustment_points.shape[1] != 2:
                raise ValueError("adjustment_points must have shape (n_points, 2)")
            if len(adjustment_points) < 2:
                raise ValueError("At least 2 adjustment points are required for interpolation.")
            trace_numbers = adjustment_points[:, 0].astype(int)
            time_seconds = adjustment_points[:, 1].astype(float)
            # Convert time in seconds to sample indices
            time_samples = (time_seconds / dt).astype(int)
            if np.any(trace_numbers < 0) or np.any(trace_numbers >= n_traces):
                raise ValueError("Trace numbers in adjustment_points are out of bounds.")
            # Allow time equal to signal duration
            max_time = n_samples_original * dt
            if np.any(time_seconds < 0) or np.any(time_seconds > max_time):
                raise ValueError(f"Time values in adjustment_points are out of bounds. Must be in range [0, {max_time:.3f}].")
            # Clip sample indices to valid range
            time_samples = np.clip(time_samples, 0, n_samples_original - 1)
            # Sort by trace numbers
            sort_idx = np.argsort(trace_numbers)
            trace_numbers = trace_numbers[sort_idx]
            time_samples = time_samples[sort_idx]
            if len(np.unique(trace_numbers)) != len(trace_numbers):
                raise ValueError("Duplicate trace numbers in adjustment_points are not allowed.")
            # Interpolate line
            interp_func = interp1d(trace_numbers, time_samples, kind='linear',
                                 fill_value='extrapolate', bounds_error=False)
            adjustment_line = interp_func(np.arange(n_traces)).astype(int)
            adjustment_line = np.clip(adjustment_line, 0, n_samples_original - 1)
            # Debug info
            # print(f"Signal length: {n_samples_original} samples, {n_samples_original * dt:.3f} seconds")
            # print(f"Adjustment line range: {np.min(adjustment_line)} - {np.max(adjustment_line)} samples")
            # print(f"STFT parameters: {n_time_stft} time frames, hop_length={hop_length}")

        # Collect amplitudes from adjustment region
        adjustment_amplitudes = []
        traces_with_regions = []  # Track traces with valid regions
        traces_without_regions = []  # Traces with no valid regions

        for i in range(n_traces):
            adj_time_sample = adjustment_line[i]
            # Match time to STFT frame
            adj_time_stft = int(np.round(adj_time_sample / hop_length))
            adj_time_stft = np.clip(adj_time_stft, 0, n_time_stft - 1)
            region = None
            if window_type == 'above':
                # Data above line — closer to t=0 → smaller indices
                if adj_time_stft > 0:
                    region = amplitudes[i, :, :adj_time_stft]
                else:
                    traces_without_regions.append(i)
            elif window_type == 'below':
                # Data below line — closer to tmax → larger indices
                if adj_time_stft < n_time_stft - 1:
                    region = amplitudes[i, :, adj_time_stft + 1:]
                else:
                    traces_without_regions.append(i)
            if region is not None and region.size > 0:
                adjustment_amplitudes.append(region)
                traces_with_regions.append(i)

        # Strict check: at least one sample must be available for adjustment
        if not adjustment_amplitudes:
            raise ValueError(
                f"No adjustment regions found with current settings. "
                f"All {len(traces_without_regions)} traces fall on boundaries for window_type='{window_type}'. "
                f"Adjust the adjustment_points or window_type to ensure at least some traces have valid regions."
            )

        # Combine all amplitudes from adjustment region
        all_adj_amps = np.concatenate([arr.flatten() for arr in adjustment_amplitudes])

        # Safety check (should not occur after previous check)
        if len(all_adj_amps) == 0:
            raise ValueError("Empty adjustment region after concatenation. This should not happen.")

        # Compute threshold value
        if adj_wnd_mode == 'median':
            threshold_value = np.median(all_adj_amps)
        elif adj_wnd_mode == 'mean':
            threshold_value = np.mean(all_adj_amps)
        elif adj_wnd_mode == 'RMS':
            threshold_value = np.sqrt(np.mean(all_adj_amps ** 2))
        threshold = threshold_value * threshold_multiplier

        # Create noise mask and replace noisy values
        noise_mask = amplitudes > threshold

        # Compute median from "clean" traces for better replacement quality
        clean_amplitudes = amplitudes.copy()
        clean_amplitudes[noise_mask] = np.nan  # Mask noisy values
        # Compute median ignoring NaN
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median_amplitudes = np.nanmedian(clean_amplitudes, axis=0, keepdims=True)

        # If all values in a time frame are noisy, use global median
        nan_mask = np.isnan(median_amplitudes)
        if np.any(nan_mask):
            global_median = np.nanmedian(amplitudes[~noise_mask]) if np.any(~noise_mask) else np.median(amplitudes)
            median_amplitudes[nan_mask] = global_median

        # Replace noisy values with median
        amplitudes = np.where(noise_mask, median_amplitudes, amplitudes)

    # --- Inverse STFT ---
    try:
        _, filtered_data = istft(amplitudes * np.exp(1j * phases), fs=fs, window='hann',
                               nperseg=nperseg, noverlap=noverlap, boundary=True,
                               time_axis=-1, input_onesided=True)
    except Exception as e:
        print(f"Error during iSTFT: {e}. Returning original data.")
        return data

    # Adjust output data size
    if filtered_data.shape[1] > n_samples_original:
        filtered_data = filtered_data[:, :n_samples_original]
    elif filtered_data.shape[1] < n_samples_original:
        pad_width = ((0, 0), (0, n_samples_original - filtered_data.shape[1]))
        filtered_data = np.pad(filtered_data, pad_width, mode='constant')

    return filtered_data.astype(data.dtype)