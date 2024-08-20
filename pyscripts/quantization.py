import numpy as np
import tensorflow as tf

def calibrate_percentile(embedding, calibration_percentage=99):
    """
    Calibrate the range of values in the embedding based on a specified percentile.
    """
    lower_bound = np.percentile(embedding, (100 - calibration_percentage) / 2)
    upper_bound = np.percentile(embedding, 100 - (100 - calibration_percentage) / 2)
    return lower_bound, upper_bound

def scalar_quantisation_percentile_og(values, qmin=-128, qmax=127, lower_bound=None, upper_bound=None):
    """
    Quantize the values to int8 using the calibrated range.
    """
    # Ensure the input is a numpy array
    values = np.array(values, dtype=np.float32)
    
    # If no calibration bounds are provided, calculate them
    if lower_bound is None or upper_bound is None:
        lower_bound, upper_bound = calibrate_percentile(values)
    
    # Calculate the scale factor and zero point
    # qmin = -128
    # qmax = 127
    scale = (upper_bound - lower_bound) / (qmax - qmin)
    zero_point = qmin - round(lower_bound / scale)
    
    # Quantize the values
    quantized_values = np.clip(np.round(values / scale + zero_point), qmin, qmax).astype(np.int8)
    
    return quantized_values

def quantize(v, precision=1000):
    return np.round(v * precision).astype(np.uint32)

def scalar_quantisation_percentile(values, qmin=-127, qmax=128, lower_bound=None, upper_bound=None):
    """
    Quantize the values to uint8 using the calibrated range.
    """
    # Ensure the input is a numpy array
    values = np.array(values, dtype=np.float32)
    
    # If no calibration bounds are provided, calculate them
    if lower_bound is None or upper_bound is None:
        lower_bound, upper_bound = calibrate_percentile(values)
    
    # Calculate the scale factor and zero point
    scale = (upper_bound - lower_bound) / (qmax - qmin)
    zero_point = qmin - round(lower_bound / scale)
    
    # Quantize the values
    quantized_values = np.clip(np.round(values / scale + zero_point), qmin, qmax).astype(np.int32)
    
    return quantized_values

def max_calibration(embedding):
    """
    Calibrate the range of values in the embedding using the maximum absolute value method.
    """
    max_abs_val = np.max(np.abs(embedding))
    return -max_abs_val, max_abs_val

def scalar_quantisation_max(values, lower_bound=None, upper_bound=None):
    """
    Quantize the values to int8 using the calibrated range.
    """
    # Ensure the input is a numpy array
    values = np.array(values, dtype=np.float32)
    
    # If no calibration bounds are provided, calculate them using max calibration
    if lower_bound is None or upper_bound is None:
        lower_bound, upper_bound = max_calibration(values)
    
    # Calculate the scale factor and zero point
    qmin = -128
    qmax = 127
    scale = (upper_bound - lower_bound) / (qmax - qmin)
    zero_point = qmin - round(lower_bound / scale)
    
    # Quantize the values
    quantized_values = np.clip(np.round(values / scale + zero_point), qmin, qmax).astype(np.int8)
    
    return quantized_values



def calculate_kl_divergence(P, Q):
    """
    Calculate the KL divergence between distributions P and Q.
    """
    P = np.array(P, dtype=np.float32)
    Q = np.array(Q, dtype=np.float32)
    P = P / np.sum(P)
    Q = Q / np.sum(Q)
    kl_div = np.sum(np.where(P != 0, P * np.log(P / Q), 0))
    return kl_div

def calibrate_using_kl_divergence(hist, num_bins=2048, num_quantized_bins=128):
    """
    Calibrate the quantization range using KL-divergence method.
    """
    histogram = np.array(hist, dtype=np.float32)
    total_bins = len(histogram)
    thresholds = []

    for i in range(num_quantized_bins, total_bins):
        P = histogram[:i].copy()
        P[i-1] += np.sum(histogram[i:])
        Q = np.zeros(i, dtype=np.float32)
        
        step = i / num_quantized_bins
        for j in range(num_quantized_bins):
            start = int(j * step)
            end = min(int((j + 1) * step), i)
            Q[start:end] += np.sum(P[start:end])
        
        Q = np.interp(np.arange(i), np.arange(0, i, step), Q[:int(i/step)])
        kl_div = calculate_kl_divergence(P, Q)
        thresholds.append((kl_div, i))

    thresholds.sort(key=lambda x: x[0])
    optimal_threshold = thresholds[0][1]
    return optimal_threshold

def collect_histogram(values, num_bins=2048):
    """
    Collect the histogram of the given values.
    """
    min_val = np.min(values)
    max_val = np.max(values)
    histogram, bin_edges = np.histogram(values, bins=num_bins, range=(min_val, max_val))
    return histogram, min_val, max_val

def scalar_quantisation_tensorrt(values, histogram=None, num_bins=2048):
    """
    Quantize the values to int8 using KL-divergence calibration method.
    """
    # Ensure the input is a numpy array
    values = np.array(values, dtype=np.float32)
    
    # Collect histogram if not provided
    if histogram is None:
        histogram, min_val, max_val = collect_histogram(values, num_bins)
    else:
        min_val, max_val = np.min(values), np.max(values)
    
    # Perform KL-divergence calibration to find the optimal threshold
    optimal_threshold_bin = calibrate_using_kl_divergence(histogram, num_bins)
    
    # Calculate the scale factor and zero point
    threshold_value = min_val + (max_val - min_val) * (optimal_threshold_bin / num_bins)
    scale = threshold_value / 127.0
    zero_point = 0
    
    # Quantize the values
    quantized_values = np.clip(np.round(values / scale), -128, 127).astype(np.int8)
    
    return quantized_values

def quantize_tensor(input_tensor, T=tf.qint8, mode='MIN_COMBINED', round_mode='HALF_AWAY_FROM_ZERO', name=None, narrow_range=False, axis=None, ensure_minimum_range=0.01):
    quantized_tensor, _, _ = tf.quantization.quantize(
        input=input_tensor,
        min_range= np.min(input_tensor),
        max_range=np.max(input_tensor),
        T=T,
        mode=mode,
        round_mode=round_mode,
        name=name,
        narrow_range=narrow_range,
        axis=axis,
        ensure_minimum_range=ensure_minimum_range
    )
    return quantized_tensor.numpy()