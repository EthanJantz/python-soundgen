"""
Utility functions for soundgen Python implementation.
Includes contour generation, interpolation, and various helper functions.
"""

import numpy as np
from scipy import interpolate
from scipy.signal import hilbert
from typing import List, Tuple, Union, Optional
import warnings

def get_smooth_contour(anchors: List[Tuple[float, float]], 
                      length: int,
                      sampling_rate: float = 16000,
                      value_floor: Optional[float] = None,
                      value_ceiling: Optional[float] = None,
                      this_is_pitch: bool = False,
                      method: str = 'spline') -> np.ndarray:
    """
    Generate smooth contour from anchor points (equivalent to R's getSmoothContour).
    
    Args:
        anchors: List of (time, value) tuples
        length: Length of output contour in samples
        sampling_rate: Sampling rate (Hz)
        value_floor: Minimum allowed value
        value_ceiling: Maximum allowed value  
        this_is_pitch: Whether this is a pitch contour (affects interpolation)
        method: Interpolation method ('spline', 'linear')
        
    Returns:
        Smooth contour as numpy array
    """
    if not anchors or length <= 0:
        return np.array([])
    
    # Convert anchors to arrays
    times = np.array([a[0] for a in anchors])
    values = np.array([a[1] for a in anchors])
    
    # Normalize times to 0-1 range if not already
    if times.max() > 1.0:
        times = times / times.max()
    
    # Create output time points
    output_times = np.linspace(0, 1, length)
    
    # Handle single anchor point
    if len(anchors) == 1:
        contour = np.full(length, values[0])
    else:
        # Interpolate based on method
        if method == 'spline' and len(anchors) >= 3:
            # Use cubic spline for smooth curves
            cs = interpolate.CubicSpline(times, values, bc_type='natural')
            contour = cs(output_times)
        else:
            # Use linear interpolation
            contour = np.interp(output_times, times, values)
    
    # Apply bounds if specified
    if value_floor is not None:
        contour = np.maximum(contour, value_floor)
    if value_ceiling is not None:
        contour = np.minimum(contour, value_ceiling)
    
    return contour

def get_glottal_cycles(pitch: np.ndarray, sampling_rate: float = 3500) -> np.ndarray:
    """
    Convert pitch contour to glottal cycle indices (equivalent to R's getGlottalCycles).
    
    Args:
        pitch: Pitch contour in Hz
        sampling_rate: Sampling rate of pitch contour
        
    Returns:
        Array of glottal cycle start indices
    """
    if len(pitch) == 0:
        return np.array([])
    
    # Match R implementation exactly
    glottal_cycles = []
    i = 0  # Start at first index (0-based in Python vs 1-based in R)
    
    while i < len(pitch):
        glottal_cycles.append(i)
        # Take steps proportionate to current F0 (matches R)
        step_size = max(2, int(np.floor(sampling_rate / pitch[i])))
        i += step_size
    
    return np.array(glottal_cycles)

def upsample(pitch_per_gc: np.ndarray, sampling_rate: float = 16000) -> dict:
    """
    Upsample pitch contour from glottal cycles to full sampling rate (matches R implementation).
    
    Args:
        pitch_per_gc: Pitch values per glottal cycle
        sampling_rate: Target sampling rate
        
    Returns:
        Dictionary with 'pitch' and 'gc' arrays
    """
    if len(pitch_per_gc) == 0:
        return {'pitch': np.array([]), 'gc': np.array([])}
    
    # Calculate duration of each glottal cycle in samples (matches R)
    gc_length_points = np.round(sampling_rate / pitch_per_gc).astype(int)
    gc_upsampled = np.concatenate([[0], np.cumsum(gc_length_points)])
    
    # Fill in missing values through linear interpolation (matches R)
    pitch_upsampled = []
    
    for i in range(len(pitch_per_gc) - 1):
        # Interpolate between consecutive pitch values
        interp_segment = np.linspace(pitch_per_gc[i], pitch_per_gc[i + 1], 
                                   gc_length_points[i], endpoint=False)
        pitch_upsampled.extend(interp_segment)
    
    # Add final segment with constant pitch
    pitch_upsampled.extend([pitch_per_gc[-1]] * gc_length_points[-1])
    
    return {'pitch': np.array(pitch_upsampled), 'gc': gc_upsampled}

def get_random_walk(length: int, 
                   rw_range: float = 1.0,
                   rw_smoothing: float = 0.5,
                   trend: Union[float, Tuple[float, float]] = 0,
                   method: str = 'filter') -> np.ndarray:
    """
    Generate random walk for parameter variation (equivalent to R's getRandomWalk).
    
    Args:
        length: Length of random walk
        rw_range: Range of variation 
        rw_smoothing: Amount of smoothing (0=no smoothing, 1=maximum smoothing)
        trend: Trend strength (single value or (min, max) range)
        method: Method for smoothing ('filter' or 'spline')
        
    Returns:
        Random walk array
    """
    if length <= 1:
        return np.array([1.0])
    
    # Generate initial random values
    walk = np.random.randn(length) * rw_range
    
    # Add trend if specified
    if isinstance(trend, (tuple, list)):
        trend_val = np.random.uniform(trend[0], trend[1])
    else:
        trend_val = trend
    
    if trend_val != 0:
        trend_line = np.linspace(0, trend_val, length)
        walk += trend_line
    
    # Apply smoothing
    if rw_smoothing > 0 and method == 'filter':
        # Simple moving average smoothing
        kernel_size = max(1, int(length * rw_smoothing / 10))
        kernel = np.ones(kernel_size) / kernel_size
        # Pad to avoid edge effects
        padded = np.pad(walk, kernel_size//2, mode='edge')
        smoothed = np.convolve(padded, kernel, mode='same')
        if kernel_size//2 > 0:
            walk = smoothed[kernel_size//2:-kernel_size//2]
        else:
            walk = smoothed
        
        # Ensure we return exactly the requested length
        walk = walk[:length]
        if len(walk) < length:
            walk = np.pad(walk, (0, length - len(walk)), mode='edge')
    
    # Normalize to have mean around 1
    walk = walk - np.mean(walk) + 1
    
    return walk

def get_binary_random_walk(rw_values: np.ndarray, 
                          noise_amount: float = 0,
                          min_length: int = 1) -> np.ndarray:
    """
    Convert random walk to binary regime indicators.
    
    Args:
        rw_values: Random walk values (0-100 scale)
        noise_amount: Percentage determining noise regimes
        min_length: Minimum length of each regime
        
    Returns:
        Binary array indicating noise regimes (0=clean, 1=subharmonics, 2=jitter+subharmonics)
    """
    if noise_amount <= 0:
        return np.zeros(len(rw_values), dtype=int)
    
    # Determine thresholds based on noise amount
    threshold1 = 100 - noise_amount
    threshold2 = 100 - noise_amount * 0.5
    
    # Create regime indicators
    regimes = np.zeros(len(rw_values), dtype=int)
    regimes[rw_values > threshold1] = 1  # Subharmonics only
    regimes[rw_values > threshold2] = 2  # Subharmonics + jitter
    
    # Enforce minimum regime length
    if min_length > 1:
        regimes = _enforce_min_length(regimes, min_length)
    
    return regimes

def _enforce_min_length(regimes: np.ndarray, min_length: int) -> np.ndarray:
    """Helper function to enforce minimum regime length."""
    if len(regimes) <= min_length:
        return regimes
    
    result = regimes.copy()
    current_regime = result[0]
    regime_start = 0
    
    for i in range(1, len(result)):
        if result[i] != current_regime:
            # Check if previous regime was too short
            if i - regime_start < min_length:
                # Extend previous regime
                result[regime_start:i] = current_regime
            else:
                current_regime = result[i]
                regime_start = i
    
    return result

def cross_fade(signal1: np.ndarray, 
              signal2: np.ndarray,
              length_ms: float = 15,
              sampling_rate: float = 16000) -> np.ndarray:
    """
    Cross-fade between two signals (equivalent to R's crossFade).
    
    Args:
        signal1: First signal
        signal2: Second signal to append
        length_ms: Cross-fade duration in milliseconds
        sampling_rate: Sampling rate
        
    Returns:
        Combined signal with cross-fade
    """
    if len(signal1) == 0:
        return signal2
    if len(signal2) == 0:
        return signal1
    
    fade_samples = int(length_ms * sampling_rate / 1000)
    fade_samples = min(fade_samples, len(signal1), len(signal2))
    
    if fade_samples <= 0:
        return np.concatenate([signal1, signal2])
    
    # Create fade curves
    fade_out = np.linspace(1, 0, fade_samples)
    fade_in = np.linspace(0, 1, fade_samples)
    
    # Apply cross-fade
    overlap_start = len(signal1) - fade_samples
    result = signal1.copy()
    
    # Fade out end of signal1 and fade in start of signal2
    result[overlap_start:] *= fade_out
    result[overlap_start:] += signal2[:fade_samples] * fade_in
    
    # Append remainder of signal2
    if len(signal2) > fade_samples:
        result = np.concatenate([result, signal2[fade_samples:]])
    
    return result

def fade_in_out(signal: np.ndarray, 
               do_fade_in: bool = True,
               do_fade_out: bool = True, 
               length_fade: int = 1000) -> np.ndarray:
    """
    Apply fade-in and/or fade-out to signal.
    
    Args:
        signal: Input signal
        do_fade_in: Whether to apply fade-in
        do_fade_out: Whether to apply fade-out
        length_fade: Fade length in samples
        
    Returns:
        Signal with fades applied
    """
    if len(signal) == 0:
        return signal
    
    result = signal.copy()
    fade_len = min(length_fade, len(signal) // 2)
    
    if do_fade_in and fade_len > 0:
        fade_curve = np.linspace(0, 1, fade_len)
        result[:fade_len] *= fade_curve
    
    if do_fade_out and fade_len > 0:
        fade_curve = np.linspace(1, 0, fade_len)
        result[-fade_len:] *= fade_curve
    
    return result

def match_lengths(signal: np.ndarray, target_length: int) -> np.ndarray:
    """
    Pad or trim signal to match target length.
    
    Args:
        signal: Input signal
        target_length: Desired length
        
    Returns:
        Signal adjusted to target length
    """
    current_length = len(signal)
    
    if current_length == target_length:
        return signal
    elif current_length < target_length:
        # Pad with zeros
        padding = target_length - current_length
        return np.pad(signal, (0, padding), mode='constant')
    else:
        # Trim to target length
        return signal[:target_length]

def add_vectors(base_vector: np.ndarray, 
               add_vector: np.ndarray,
               insertion_point: int = 0) -> np.ndarray:
    """
    Add one vector to another at specified insertion point.
    
    Args:
        base_vector: Base signal to add to
        add_vector: Signal to add
        insertion_point: Sample index where to start adding
        
    Returns:
        Combined signal
    """
    if len(add_vector) == 0:
        return base_vector
    
    # Ensure base vector is long enough
    min_length = insertion_point + len(add_vector)
    if len(base_vector) < min_length:
        base_vector = np.pad(base_vector, (0, min_length - len(base_vector)), mode='constant')
    
    # Add the vectors
    result = base_vector.copy()
    end_point = insertion_point + len(add_vector)
    result[insertion_point:end_point] += add_vector
    
    return result

def wiggle_anchors(anchors: List[Tuple[float, float]], 
                  temperature: float = 0.1,
                  low: Tuple[float, float] = (0, 0),
                  high: Tuple[float, float] = (1, 1000),
                  temp_coef: float = 1.0,
                  wiggle_all_rows: bool = False) -> List[Tuple[float, float]]:
    """
    Add random variation to anchor points based on temperature.
    
    Args:
        anchors: List of (time, value) anchor points
        temperature: Amount of randomness
        low: (time_min, value_min) bounds  
        high: (time_max, value_max) bounds
        temp_coef: Temperature coefficient
        wiggle_all_rows: Whether to wiggle all anchors or just some
        
    Returns:
        Wiggled anchor points
    """
    if temperature <= 0 or not anchors:
        return anchors
    
    result = []
    sd_time = (high[0] - low[0]) * temperature * temp_coef / 100
    sd_value = (high[1] - low[1]) * temperature * temp_coef / 100
    
    for time, value in anchors:
        if wiggle_all_rows or np.random.random() < 0.5:
            # Wiggle this anchor
            new_time = np.clip(np.random.normal(time, sd_time), low[0], high[0])
            new_value = np.clip(np.random.normal(value, sd_value), low[1], high[1])
            result.append((new_time, new_value))
        else:
            result.append((time, value))
    
    return result