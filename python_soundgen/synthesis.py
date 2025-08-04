"""
Core synthesis functions for soundgen Python implementation.
Handles harmonic and noise generation, the main synthesis pipeline.
"""

import numpy as np
from scipy.signal import stft, istft
from typing import List, Tuple, Optional, Union, Dict, Any
import warnings

from .parameters import ParameterSet, THROWAWAY_DB
from .utils import (get_smooth_contour, get_glottal_cycles, upsample, 
                   get_random_walk, get_binary_random_walk, cross_fade, 
                   fade_in_out, match_lengths, add_vectors)
from .spectral import get_rolloff, get_spectral_envelope

def generate_harmonics(pitch: np.ndarray,
                      attack_len: float = 50,
                      noise_amount: float = 0,
                      noise_intensity: float = 0,
                      jitter_dep: float = 0,
                      jitter_length_ms: float = 1,
                      vibrato_freq: float = 100,
                      vibrato_dep: float = 0,
                      shimmer_dep: float = 0,
                      creaky_breathy: float = 0,
                      rolloff_exp: float = -18,
                      rolloff_exp_delta: float = -2,
                      adjust_rolloff_per_khz: float = -6,
                      quadratic_delta: float = 0,
                      quadratic_n_harm: int = 3,
                      formant_strength: float = 1,
                      temperature: float = 0,
                      min_epoch_length_ms: float = 300,
                      g0: float = 100,
                      sideband_width_hz: float = 0,
                      rolloff_lip: float = 6,
                      trill_dep: float = 0,
                      trill_freq: float = 30,
                      ampl_anchors: Optional[List[Tuple[float, float]]] = None,
                      overlap: float = 75,
                      window_length_points: int = 2048,
                      sampling_rate: float = 44100,
                      pitch_floor: float = 75,
                      pitch_ceiling: float = 3500,
                      pitch_sampling_rate: float = 3500) -> np.ndarray:
    """
    Generate harmonics component (equivalent to R's generateHarmonics).
    
    Args:
        pitch: Pitch contour in Hz
        attack_len: Attack duration in ms
        noise_amount: Amount of noise (0-100%)
        noise_intensity: Intensity of noise effects (0-100%)
        jitter_dep: Pitch jitter depth in semitones
        jitter_length_ms: Duration of pitch jumps in ms
        vibrato_freq: Vibrato frequency in Hz
        vibrato_dep: Vibrato depth in semitones
        shimmer_dep: Amplitude shimmer depth (0-100%)
        creaky_breathy: Voice quality (-1=creaky, +1=breathy)
        rolloff_exp: Basic rolloff in dB/octave
        rolloff_exp_delta: Rolloff change per octave
        adjust_rolloff_per_khz: Rolloff adjustment per kHz
        quadratic_delta: Quadratic adjustment for lower harmonics
        quadratic_n_harm: Number of harmonics affected by quadratic term
        formant_strength: Formant amplitude scaling
        temperature: Amount of stochastic variation
        min_epoch_length_ms: Minimum epoch length for regime changes
        g0: Subharmonic target frequency
        sideband_width_hz: Subharmonic sideband width
        rolloff_lip: Lip radiation effect
        trill_dep: Amplitude modulation depth
        trill_freq: Amplitude modulation frequency
        ampl_anchors: Amplitude envelope anchors
        overlap: FFT window overlap percentage
        window_length_points: FFT window length
        sampling_rate: Audio sampling rate
        pitch_floor: Minimum allowed pitch
        pitch_ceiling: Maximum allowed pitch
        pitch_sampling_rate: Pitch contour sampling rate
        
    Returns:
        Generated harmonic waveform
    """
    if len(pitch) == 0:
        return np.array([])
    
    pitch = np.array(pitch)
    
    ## PRE-SYNTHESIS EFFECTS
    
    # Apply vibrato
    if vibrato_dep > 0:
        time_points = np.arange(len(pitch)) / pitch_sampling_rate
        vibrato = 2 ** (np.sin(2 * np.pi * time_points * vibrato_freq) * vibrato_dep / 12)
        pitch = pitch * vibrato
    
    # Convert pitch to glottal cycles
    gc_indices = get_glottal_cycles(pitch, pitch_sampling_rate)
    if len(gc_indices) == 0:
        return np.array([])
    
    pitch_per_gc = pitch[gc_indices]
    n_gc = len(pitch_per_gc)
    
    # Generate amplitude contour
    rolloff_exp_ampl = 0  # Initialize as scalar
    if ampl_anchors is not None:
        ampl_contour = get_smooth_contour(ampl_anchors, n_gc, value_floor=0, 
                                        value_ceiling=-THROWAWAY_DB, sampling_rate=sampling_rate)
        if len(ampl_contour) > 0:
            ampl_contour = ampl_contour / abs(THROWAWAY_DB) - 1
            rolloff_exp_ampl = ampl_contour * 15  # rolloff_per_ampl constant
        else:
            rolloff_exp_ampl = 0
    
    # Generate random walk for stochastic effects
    if temperature > 0:
        rw = get_random_walk(n_gc, temperature, trend=(-0.5, 0.5), rw_smoothing=0.3)
        # Ensure exact length match
        if len(rw) != n_gc:
            rw = np.interp(np.linspace(0, 1, n_gc), np.linspace(0, 1, len(rw)), rw)
        
        rw_0_100 = (rw - np.min(rw)) / (np.max(rw) - np.min(rw)) * 100
        rw_bin = get_binary_random_walk(rw_0_100, noise_amount, 
                                       int(min_epoch_length_ms / 1000 * np.mean(pitch_per_gc)))
        rw = rw - np.mean(rw) + 1
        
        vocal_fry_on = rw_bin > 0
        jitter_on = shimmer_on = rw_bin == 2
    else:
        rw = np.ones(n_gc)
        vocal_fry_on = jitter_on = shimmer_on = np.ones(n_gc, dtype=bool)
    
    # Apply jitter (pitch variation)
    if jitter_dep > 0 and noise_amount > 0:
        pitch_per_gc = _apply_jitter(pitch_per_gc, jitter_dep, jitter_length_ms, 
                                   rw, jitter_on)
    
    # Apply random pitch drift
    if temperature > 0:
        drift = _apply_pitch_drift(pitch_per_gc, temperature)
        pitch_per_gc = pitch_per_gc * drift
    else:
        drift = np.ones(n_gc)
    
    # Enforce pitch bounds
    pitch_per_gc = np.clip(pitch_per_gc, pitch_floor, pitch_ceiling)
    
    ## HARMONIC STACK PREPARATION
    
    # Calculate number of harmonics (matches R implementation)
    # Note: R has apparent bug/feature that reduces harmonic count significantly
    n_harmonics = int(np.ceil((sampling_rate / 2 - np.min(pitch_per_gc)) / np.min(pitch_per_gc)))
    # Add reasonable safety limit
    n_harmonics = min(n_harmonics, 100)
    
    # Get rolloff for harmonics
    rolloff_source = get_rolloff(
        pitch_per_gc=pitch_per_gc,
        n_harmonics=n_harmonics,
        rolloff_exp=(rolloff_exp + rolloff_exp_ampl) * rw**3,
        rolloff_exp_delta=rolloff_exp_delta * rw**3,
        adjust_rolloff_per_khz=adjust_rolloff_per_khz * rw,
        quadratic_delta=quadratic_delta,
        quadratic_n_harm=quadratic_n_harm,
        sampling_rate=sampling_rate
    )
    
    # Apply shimmer (amplitude variation)
    if shimmer_dep > 0 and noise_amount > 0:
        shimmer = 2 ** (np.random.normal(0, shimmer_dep/100, rolloff_source.shape[1]) * 
                       rw * shimmer_on)
        rolloff_source = rolloff_source * shimmer[np.newaxis, :]
    
    # Add vocal fry (subharmonics) - simplified version
    if sideband_width_hz > 0 and noise_amount > 0:
        rolloff_source = _add_vocal_fry_simple(rolloff_source, pitch_per_gc, g0, 
                                             sideband_width_hz, vocal_fry_on, rw)
    
    # Discard harmonics that are zero throughout (matches R optimization)
    active_harmonics = np.any(rolloff_source > 0, axis=1)
    rolloff_source = rolloff_source[active_harmonics, :]
    
    ## WAVEFORM SYNTHESIS
    
    # Upsample pitch contour
    upsampled = upsample(pitch_per_gc, sampling_rate)
    pitch_upsampled = upsampled['pitch']
    gc_upsampled = upsampled['gc']
    
    if len(pitch_upsampled) == 0:
        return np.array([])
    
    # Generate waveform
    waveform = _synthesize_waveform(rolloff_source, pitch_upsampled, gc_upsampled, 
                                  gc_indices, sampling_rate)
    
    ## POST-SYNTHESIS EFFECTS
    
    # Apply amplitude envelope
    if ampl_anchors is not None:
        ampl_anchors_linear = [(t, 2**(v/10)) for t, v in ampl_anchors if v > THROWAWAY_DB]
        if ampl_anchors_linear:
            ampl_envelope = get_smooth_contour(ampl_anchors_linear, len(waveform), 
                                             value_floor=0, sampling_rate=sampling_rate)
            waveform = waveform * ampl_envelope
    
    # Normalize
    if np.max(np.abs(waveform)) > 0:
        waveform = waveform / np.max(np.abs(waveform))
    
    # Apply attack/decay
    if attack_len > 0:
        fade_samples = int(attack_len * sampling_rate / 1000)
        waveform = fade_in_out(waveform, length_fade=fade_samples)
    
    # Apply pitch drift to amplitude
    if temperature > 0 and len(drift) > 1:
        drift_upsampled = np.interp(np.arange(len(waveform)), 
                                   np.linspace(0, len(waveform)-1, len(drift)), drift)
        waveform = waveform * drift_upsampled
    
    return waveform

def _apply_jitter(pitch_per_gc: np.ndarray, jitter_dep: float, jitter_length_ms: float,
                 rw: np.ndarray, jitter_on: np.ndarray) -> np.ndarray:
    """Apply jitter (pitch variation) to glottal cycles."""
    n_gc = len(pitch_per_gc)
    ratio = pitch_per_gc * jitter_length_ms / 1000
    
    # Find jitter change points with safety checks
    indices = [0]
    i = 0
    max_iterations = n_gc * 2  # Safety limit to prevent infinite loops
    iteration_count = 0
    
    while i < n_gc - 1 and iteration_count < max_iterations:
        # Ensure minimum step size to prevent infinite loops
        step_size = max(1, int(ratio[i]))  # At least 1 step
        next_i = min(n_gc - 1, indices[-1] + step_size)
        
        # If we're not advancing, force advancement
        if next_i <= i:
            next_i = i + 1
            
        i = next_i
        indices.append(i)
        iteration_count += 1
    
    # Remove duplicates and ensure bounds
    indices = sorted(list(set(indices)))
    indices = [idx for idx in indices if idx < n_gc]
    
    # Ensure we have at least one valid index
    if not indices:
        indices = [0]
    
    # Generate jitter values
    jitter_values = 2 ** (np.random.normal(0, jitter_dep/12, len(indices)) * 
                         rw[np.array(indices)] * jitter_on[np.array(indices)])
    
    # Interpolate to all glottal cycles
    jitter_per_gc = np.interp(np.arange(n_gc), indices, jitter_values)
    
    return pitch_per_gc * jitter_per_gc

def _apply_pitch_drift(pitch_per_gc: np.ndarray, temperature: float) -> np.ndarray:
    """Apply slow random drift to pitch."""
    n_gc = len(pitch_per_gc)
    
    # Calculate smoothing and range based on temperature and duration
    rw_smoothing = 0.9 - temperature * 0.5 - 1.2 / (1 + np.exp(-0.008 * (n_gc - 10))) + 0.6
    rw_range = temperature * 0.3 + n_gc / 1000 / 12
    
    drift = get_random_walk(n_gc, rw_range, rw_smoothing, method='spline')
    drift = 2 ** (drift - np.mean(drift))
    
    return drift

def _add_vocal_fry_simple(rolloff_source: np.ndarray, pitch_per_gc: np.ndarray, 
                         g0: float, sideband_width_hz: float, vocal_fry_on: np.ndarray,
                         rw: np.ndarray) -> np.ndarray:
    """Add vocal fry (subharmonics) - simplified version."""
    # This is a simplified implementation - the full R version is quite complex
    # For now, just add some subharmonic energy
    
    n_harm, n_gc = rolloff_source.shape
    g0_adjusted = g0 * rw**4
    sideband_adjusted = sideband_width_hz * rw**4 * vocal_fry_on
    
    # Add subharmonic energy at g0 frequency
    for gc in range(n_gc):
        if vocal_fry_on[gc] and sideband_adjusted[gc] > 0:
            subharmonic_ratio = g0_adjusted[gc] / pitch_per_gc[gc]
            if 0.1 < subharmonic_ratio < 0.9:  # Reasonable subharmonic range
                # Add energy at subharmonic frequency
                subharmonic_strength = 0.3 * sideband_adjusted[gc] / sideband_width_hz
                # This is very simplified - just boost lower harmonics
                boost_harmonics = min(3, n_harm)
                rolloff_source[:boost_harmonics, gc] *= (1 + subharmonic_strength)
    
    return rolloff_source

def _synthesize_waveform(rolloff_source: np.ndarray, pitch_upsampled: np.ndarray,
                        gc_upsampled: np.ndarray, gc_indices: np.ndarray,
                        sampling_rate: float) -> np.ndarray:
    """Synthesize waveform from harmonic specifications."""
    if len(pitch_upsampled) == 0:
        return np.array([])
    
    # Phase integration
    phase_integrand = np.cumsum(pitch_upsampled) / sampling_rate
    waveform = np.zeros(len(pitch_upsampled))
    
    n_harmonics = rolloff_source.shape[0]
    
    # Synthesize each harmonic
    for h in range(n_harmonics):
        harmonic_num = h + 1
        
        # Get amplitude envelope for this harmonic
        if len(rolloff_source[h, :]) == 1:
            # Static amplitude
            amplitude = np.full(len(pitch_upsampled), rolloff_source[h, 0])
        else:
            # Interpolate amplitude to upsampled length
            gc_positions = np.linspace(0, len(pitch_upsampled)-1, len(rolloff_source[h, :]))
            amplitude = np.interp(np.arange(len(pitch_upsampled)), gc_positions, rolloff_source[h, :])
        
        # Generate harmonic
        harmonic_wave = amplitude * np.sin(2 * np.pi * phase_integrand * harmonic_num)
        waveform += harmonic_wave
    
    return waveform

def generate_noise(length: int,
                  breathing_anchors: Optional[List[Tuple[float, float]]] = None,
                  rolloff_breathing: float = -6,
                  attack_len: float = 10,
                  window_length_points: int = 1024,
                  sampling_rate: float = 44100,
                  overlap: float = 75,
                  filter_breathing: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Generate noise component (equivalent to R's generateNoise).
    
    Args:
        length: Length of output in samples
        breathing_anchors: Amplitude envelope as (time, dB) pairs
        rolloff_breathing: Spectral rolloff in dB/octave
        attack_len: Fade in/out duration in ms
        window_length_points: FFT window length
        sampling_rate: Sampling rate
        overlap: FFT window overlap percentage
        filter_breathing: Optional spectral filter matrix
        
    Returns:
        Generated noise waveform  
    """
    if length <= 0:
        return np.array([])
    
    # Default breathing anchors if not provided
    if breathing_anchors is None:
        breathing_anchors = [(0, THROWAWAY_DB), (length/sampling_rate*1000, THROWAWAY_DB)]
    
    # Convert amplitude anchors from dB to linear
    breathing_anchors_linear = [(t, 2**(v/10)) for t, v in breathing_anchors]
    
    # Generate breathing strength contour
    breathing_strength = get_smooth_contour(
        breathing_anchors_linear, length, sampling_rate=sampling_rate, value_floor=0
    )
    
    if np.sum(breathing_strength) == 0:
        return np.zeros(length)
    
    # Set up FFT parameters
    step = int(window_length_points * (1 - overlap/100))
    n_windows = (length + window_length_points - 1) // step
    nr = window_length_points // 2
    
    # Create spectral filter
    if filter_breathing is None:
        filter_breathing = np.ones((nr, 1))
    
    # Ensure filter has right dimensions
    if filter_breathing.ndim == 1:
        filter_breathing = filter_breathing.reshape(-1, 1)
    
    # Apply basic rolloff to filter
    rolloff_curve = 2 ** (rolloff_breathing / 10 * np.log2(np.arange(1, nr + 1)))
    if filter_breathing.shape[1] == 1:
        filter_breathing = filter_breathing * rolloff_curve.reshape(-1, 1)
    else:
        for col in range(filter_breathing.shape[1]):
            filter_breathing[:, col] *= rolloff_curve
    
    # Generate spectral noise
    noise_spectrum = np.random.randn(nr, n_windows) + 1j * np.random.randn(nr, n_windows)
    
    # Apply filter
    if filter_breathing.shape[1] == 1:
        filtered_spectrum = noise_spectrum * filter_breathing
    else:
        filter_indices = np.linspace(0, filter_breathing.shape[1]-1, n_windows).astype(int)
        filtered_spectrum = noise_spectrum * filter_breathing[:, filter_indices]
    
    # Convert to time domain using overlap-add
    breathing = _overlap_add_synthesis(filtered_spectrum, window_length_points, step)
    
    # Trim or pad to exact length
    breathing = match_lengths(breathing, length)
    
    # Normalize and apply breathing strength
    if np.max(np.abs(breathing)) > 0:
        breathing = breathing / np.max(np.abs(breathing))
    breathing = breathing * breathing_strength
    
    # Apply attack/decay
    if attack_len > 0:
        fade_samples = int(attack_len * sampling_rate / 1000)
        breathing = fade_in_out(breathing, length_fade=fade_samples)
    
    return breathing

def _overlap_add_synthesis(spectrum: np.ndarray, window_length: int, step: int) -> np.ndarray:
    """Synthesize time-domain signal from spectrum using overlap-add."""
    nr, n_windows = spectrum.shape
    
    # Create full spectrum (including negative frequencies)
    full_spectrum = np.zeros((window_length, n_windows), dtype=complex)
    full_spectrum[:nr, :] = spectrum
    full_spectrum[nr:, :] = np.conj(spectrum[::-1, :])  # Mirror for real signal
    
    # IFFT each window
    windowed_signals = np.fft.ifft(full_spectrum, axis=0).real
    
    # Overlap-add
    output_length = (n_windows - 1) * step + window_length
    output = np.zeros(output_length)
    
    for i in range(n_windows):
        start_idx = i * step
        end_idx = start_idx + window_length
        output[start_idx:end_idx] += windowed_signals[:, i]
    
    return output