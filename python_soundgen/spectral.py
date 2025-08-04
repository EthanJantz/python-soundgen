"""
Spectral processing functions for soundgen Python implementation.
Handles rolloff calculation, formant filtering, and spectral envelope generation.
"""

import numpy as np
from scipy import stats
from scipy.signal import stft, istft
from typing import List, Tuple, Optional, Union, Dict, Any
import matplotlib.pyplot as plt

from .parameters import FormantSpec, SPEED_SOUND, THROWAWAY_DB
from .utils import get_smooth_contour, get_random_walk

def get_rolloff(pitch_per_gc: np.ndarray,
               n_harmonics: int = 100,
               rolloff_exp: float = -12,
               rolloff_exp_delta: float = -2,
               quadratic_delta: float = 0,
               quadratic_n_harm: int = 2,
               quadratic_ceiling: Optional[float] = None,
               adjust_rolloff_per_khz: float = -6,
               baseline_hz: float = 200,
               throwaway_db: float = THROWAWAY_DB,
               sampling_rate: float = 44100,
               plot: bool = False) -> np.ndarray:
    """
    Control rolloff of harmonics (equivalent to R's getRolloff).
    
    Args:
        pitch_per_gc: Fundamental frequency per glottal cycle (Hz)
        n_harmonics: Maximum number of harmonics to generate
        rolloff_exp: Basic rolloff rate (dB/octave)
        rolloff_exp_delta: Change in rolloff rate per octave
        quadratic_delta: Quadratic adjustment for lower harmonics (dB)
        quadratic_n_harm: Number of harmonics affected by quadratic term
        quadratic_ceiling: Alternative specification for quadratic harmonics (Hz)
        adjust_rolloff_per_khz: Rolloff adjustment per kHz of f0
        baseline_hz: Reference frequency for rolloff adjustment
        throwaway_db: Discard harmonics weaker than this (dB)
        sampling_rate: Sampling rate (Hz)
        plot: Whether to plot the rolloff
        
    Returns:
        Matrix of amplitude multipliers (harmonics x glottal_cycles)
    """
    pitch_per_gc = np.atleast_1d(pitch_per_gc)
    n_gc = len(pitch_per_gc)
    
    # Initialize rolloff matrix
    rolloff = np.zeros((n_harmonics, n_gc))
    
    # Calculate exponential decay with delta adjustment
    deltas = np.zeros((n_harmonics, n_gc))
    if np.any(rolloff_exp_delta != 0):
        for h in range(1, n_harmonics):  # Start from harmonic 2 (index 1)
            deltas[h, :] = rolloff_exp_delta * (pitch_per_gc * (h + 1) - baseline_hz) / 1000
    
    # Apply basic exponential rolloff
    for h in range(n_harmonics):
        harmonic_num = h + 1  # Harmonic number (1-based)
        rolloff[h, :] = ((rolloff_exp + adjust_rolloff_per_khz * 
                         (pitch_per_gc - baseline_hz) / 1000) * np.log2(harmonic_num)) + deltas[h, :]
        
        # Discard harmonics above Nyquist frequency
        nyquist_mask = harmonic_num * pitch_per_gc >= sampling_rate / 2
        rolloff[h, nyquist_mask] = -np.inf
    
    # Apply quadratic adjustment to lower harmonics
    if quadratic_delta != 0:
        if quadratic_ceiling is not None:
            # Variable number of harmonics based on frequency ceiling
            quadratic_n_harm_vec = np.round(quadratic_ceiling / pitch_per_gc).astype(int)
        else:
            # Fixed number of harmonics
            quadratic_n_harm_vec = np.full(n_gc, quadratic_n_harm, dtype=int)
        
        # Ensure minimum of 3 harmonics for proper parabola
        quadratic_n_harm_vec[quadratic_n_harm_vec == 2] = 3
        
        for i in range(n_gc):
            n_harm = quadratic_n_harm_vec[i]
            if n_harm < 3:
                # Single harmonic adjustment (just F0)
                if n_harm >= 1:
                    rolloff[0, i] += quadratic_delta
            else:
                # Parabolic adjustment
                # Parabola: f(x) = ax² + bx + c
                # Constraints: f(1) = 0, f(n_harm) = 0, f((1+n_harm)/2) = quadratic_delta
                a = -4 * quadratic_delta / (n_harm - 1) ** 2
                b = -a * (1 + n_harm)
                c = a * n_harm
                
                for h in range(min(n_harm, n_harmonics)):
                    x = h + 1  # Harmonic number (1-based)
                    adjustment = a * x**2 + b * x + c
                    rolloff[h, i] += adjustment
    
    # Apply throwaway threshold
    if np.isfinite(throwaway_db):
        rolloff[rolloff < throwaway_db] = -np.inf
    
    # Normalize so F0 amplitude is always 0 dB
    for i in range(n_gc):
        max_val = np.max(rolloff[:, i])
        if np.isfinite(max_val):
            rolloff[:, i] -= max_val
    
    # Plotting
    if plot:
        _plot_rolloff(rolloff, pitch_per_gc, sampling_rate)
    
    # Convert from dB to linear amplitude multipliers
    rolloff_linear = 2 ** (rolloff / 10)
    
    # Remove harmonics that are zero throughout
    active_harmonics = np.any(rolloff_linear > 0, axis=1)
    rolloff_linear = rolloff_linear[active_harmonics, :]
    
    return rolloff_linear

def _plot_rolloff(rolloff: np.ndarray, pitch_per_gc: np.ndarray, sampling_rate: float):
    """Helper function to plot rolloff characteristics."""
    plt.figure(figsize=(10, 6))
    
    if len(pitch_per_gc) == 1 or np.var(pitch_per_gc) == 0:
        # Single pitch - simple plot
        pitch = pitch_per_gc[0]
        valid_idx = rolloff[:, 0] > -np.inf
        harmonics = np.arange(1, len(rolloff) + 1)[valid_idx]
        frequencies = harmonics * pitch / 1000  # kHz
        amplitudes = rolloff[valid_idx, 0]
        
        plt.plot(frequencies, amplitudes, 'bo-')
        plt.title('Glottal Source Rolloff')
    else:
        # Multiple pitches - show min and max
        pitch_min = np.min(pitch_per_gc)
        pitch_max = np.max(pitch_per_gc)
        idx_min = np.argmin(pitch_per_gc)
        idx_max = np.argmax(pitch_per_gc)
        
        # Plot minimum pitch
        valid_min = rolloff[:, idx_min] > -np.inf
        harmonics_min = np.arange(1, len(rolloff) + 1)[valid_min]
        freq_min = harmonics_min * pitch_min / 1000
        amp_min = rolloff[valid_min, idx_min]
        
        # Plot maximum pitch  
        valid_max = rolloff[:, idx_max] > -np.inf
        harmonics_max = np.arange(1, len(rolloff) + 1)[valid_max]
        freq_max = harmonics_max * pitch_max / 1000
        amp_max = rolloff[valid_max, idx_max]
        
        plt.plot(freq_min, amp_min, 'bo-', label=f'Lowest pitch ({pitch_min:.0f} Hz)')
        plt.plot(freq_max, amp_max, 'ro-', label=f'Highest pitch ({pitch_max:.0f} Hz)')
        plt.legend()
        plt.title('Glottal Source Rolloff')
    
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Amplitude (dB)')
    plt.xlim(0, sampling_rate / 2000)  # Up to Nyquist in kHz
    plt.grid(True, alpha=0.3)
    plt.show()

def get_spectral_envelope(nr: int,
                         nc: int,
                         exact_formants: Optional[List[FormantSpec]] = None,
                         formant_strength: float = 1.0,
                         rolloff_lip: float = 6.0,
                         mouth_anchors: Optional[List[Tuple[float, float]]] = None,
                         mouth_opening_threshold: float = 0.0,
                         ampl_boost_open_mouth_db: float = 0.0,
                         vocal_tract_length: Optional[float] = None,
                         temperature: float = 0.0,
                         extra_formants_ampl: float = 30.0,
                         smooth_linear_factor: float = 1.0,
                         sampling_rate: float = 44100) -> np.ndarray:
    """
    Generate spectral envelope for formant filtering (equivalent to R's getSpectralEnvelope).
    
    Args:
        nr: Number of frequency bins (window_length_points / 2)
        nc: Number of time steps
        exact_formants: List of formant specifications
        formant_strength: Scale factor for formant amplitude
        rolloff_lip: Lip radiation effect (dB/octave)
        mouth_anchors: Mouth opening contour as (time, opening) pairs
        mouth_opening_threshold: Threshold for considering mouth "open"
        ampl_boost_open_mouth_db: Amplitude boost when mouth is open
        vocal_tract_length: Vocal tract length in cm
        temperature: Amount of stochastic variation
        extra_formants_ampl: Amplitude of additional formants
        smooth_linear_factor: Smoothing factor for formant interpolation
        sampling_rate: Sampling rate (Hz)
        
    Returns:
        Spectral envelope matrix (nr x nc)
    """
    # Handle default formants
    if exact_formants is None:
        if vocal_tract_length is not None:
            # Create schwa based on vocal tract length
            freq = SPEED_SOUND / (4 * vocal_tract_length)
            width = 50 + (np.log2(freq) - 5) * 20
            exact_formants = [FormantSpec(time=0, freq=freq, amp=30, width=width)]
        else:
            # Default schwa
            exact_formants = [FormantSpec(time=0, freq=500, amp=30, width=100)]
    
    # Estimate vocal tract length if not provided
    if vocal_tract_length is None and len(exact_formants) > 1:
        formant_freqs = [f.freq if isinstance(f.freq, (int, float)) else f.freq[0] 
                        for f in exact_formants[:3]]  # Use first 3 formants
        formant_dispersion = np.mean(np.diff(formant_freqs))
        vocal_tract_length = SPEED_SOUND / (2 * formant_dispersion)
    elif vocal_tract_length is None:
        freq = exact_formants[0].freq
        if isinstance(freq, (list, np.ndarray)):
            freq = freq[0]
        vocal_tract_length = SPEED_SOUND / (4 * freq)
    
    # Convert formants to matrix format and upsample to nc time steps
    formants_upsampled = _upsample_formants(exact_formants, nc, smooth_linear_factor)
    
    # Add stochastic formants if temperature > 0
    if temperature > 0:
        formants_upsampled = _add_stochastic_formants(
            formants_upsampled, nc, vocal_tract_length, temperature, 
            extra_formants_ampl, formant_strength, sampling_rate
        )
    
    # Process mouth opening
    mouth_opening, mouth_open_binary = _process_mouth_opening(
        mouth_anchors, nc, mouth_opening_threshold
    )
    
    # Adjust formants for mouth opening
    if vocal_tract_length is not None:
        formants_upsampled = _adjust_formants_for_mouth_opening(
            formants_upsampled, mouth_opening, vocal_tract_length, sampling_rate, nr
        )
    
    # Add nasalization for closed mouth segments
    formants_upsampled = _add_nasalization(formants_upsampled, mouth_open_binary, sampling_rate, nr)
    
    # Generate spectral envelope
    spectral_envelope = _generate_spectral_envelope(
        formants_upsampled, nr, nc, formant_strength, sampling_rate
    )
    
    # Add lip radiation
    spectral_envelope = _add_lip_radiation(
        spectral_envelope, rolloff_lip, mouth_opening, mouth_open_binary, 
        ampl_boost_open_mouth_db, nr, nc
    )
    
    # Convert from dB to linear scale
    return 2 ** (spectral_envelope / 10)

def _upsample_formants(formants: List[FormantSpec], nc: int, 
                      smooth_linear_factor: float) -> Dict[str, np.ndarray]:
    """Upsample formant specifications to nc time steps."""
    upsampled = {}
    
    for i, formant in enumerate(formants):
        formant_key = f'f{i+1}'
        
        # Convert single values to arrays
        time = np.atleast_1d(formant.time)
        freq = np.atleast_1d(formant.freq)
        amp = np.atleast_1d(formant.amp) 
        width = np.atleast_1d(formant.width)
        
        if len(time) == 1:
            # Static formant
            upsampled[formant_key] = {
                'time': np.zeros(nc),
                'freq': np.full(nc, freq[0]),
                'amp': np.full(nc, amp[0]),
                'width': np.full(nc, width[0])
            }
        else:
            # Dynamic formant - interpolate
            time_norm = time / np.max(time) if np.max(time) > 0 else time
            output_times = np.linspace(0, 1, nc)
            
            upsampled[formant_key] = {
                'time': output_times,
                'freq': np.interp(output_times, time_norm, freq),
                'amp': np.interp(output_times, time_norm, amp),
                'width': np.interp(output_times, time_norm, width)
            }
    
    return upsampled

def _add_stochastic_formants(formants: Dict, nc: int, vocal_tract_length: float,
                           temperature: float, extra_formants_ampl: float,
                           formant_strength: float, sampling_rate: float) -> Dict:
    """Add stochastic formants and wiggle existing ones."""
    # Calculate formant dispersion
    if len(formants) > 1:
        freqs = [formants[key]['freq'][0] for key in sorted(formants.keys())]
        formant_dispersion = np.mean(np.diff(freqs))
    else:
        formant_dispersion = SPEED_SOUND / (2 * vocal_tract_length)
    
    # Add extra formants
    n_formants = len(formants)
    highest_key = f'f{n_formants}'
    max_freq = np.max(formants[highest_key]['freq'])
    
    formant_counter = n_formants
    while max_freq < (sampling_rate / 2 - 1000):  # Stop before Nyquist
        formant_counter += 1
        new_key = f'f{formant_counter}'
        
        # Generate random walk for frequency variation
        rw = get_random_walk(nc, temperature * 0.15, 0.0)
        
        # New formant frequency
        freq_boost = np.random.gamma(
            formant_dispersion**2 / (formant_dispersion * temperature)**2,
            formant_dispersion * temperature
        )
        new_freq = max_freq + freq_boost * rw
        
        # New formant amplitude
        amp_base = extra_formants_ampl * formant_strength
        new_amp = np.random.gamma(
            (formant_strength / temperature)**2,
            amp_base * temperature**2 / formant_strength
        ) * rw
        
        # New formant width
        new_width = 50 + (np.log2(np.mean(new_freq)) - 5) * 20
        
        formants[new_key] = {
            'time': np.linspace(0, 1, nc),
            'freq': new_freq,
            'amp': new_amp,
            'width': np.full(nc, new_width)
        }
        
        max_freq = np.max(new_freq)
    
    # Wiggle existing formants
    for key in list(formants.keys()):
        for param in ['freq', 'amp', 'width']:
            rw = get_random_walk(nc, temperature * 0.15, 0.3, 
                               trend=np.random.normal(0, 1))
            formants[key][param] = formants[key][param] * rw
    
    return formants

def _process_mouth_opening(mouth_anchors: Optional[List[Tuple[float, float]]], 
                          nc: int, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """Process mouth opening contour."""
    if mouth_anchors is None:
        # Default to half-open
        mouth_opening = np.full(nc, 0.5)
        mouth_open_binary = np.ones(nc, dtype=int)
    else:
        mouth_opening = get_smooth_contour(mouth_anchors, nc, value_floor=0, value_ceiling=1)
        mouth_opening[mouth_opening < threshold] = 0
        mouth_open_binary = (mouth_opening > 0).astype(int)
    
    return mouth_opening, mouth_open_binary

def _adjust_formants_for_mouth_opening(formants: Dict, mouth_opening: np.ndarray,
                                     vocal_tract_length: float, sampling_rate: float,
                                     nr: int) -> Dict:
    """Adjust formant frequencies based on mouth opening."""
    bin_width = sampling_rate / (2 * nr)
    
    # Calculate frequency adjustment
    adjustment_hz = (mouth_opening - 0.5) * SPEED_SOUND / (4 * vocal_tract_length)
    adjustment_bins = adjustment_hz / bin_width
    
    # Apply adjustment and convert to bin indices
    for key in formants:
        # Convert frequency to bin indices
        formants[key]['freq'] = (formants[key]['freq'] - bin_width/2) / bin_width + 1
        formants[key]['width'] = formants[key]['width'] / bin_width
        
        # Apply mouth opening adjustment
        formants[key]['freq'] = formants[key]['freq'] + adjustment_bins
        
        # Ensure positive frequencies
        formants[key]['freq'] = np.maximum(formants[key]['freq'], 1)
    
    return formants

def _add_nasalization(formants: Dict, mouth_open_binary: np.ndarray,
                     sampling_rate: float, nr: int) -> Dict:
    """Add nasalization effects for closed mouth segments."""
    nasalized_idx = mouth_open_binary == 0
    
    if not np.any(nasalized_idx):
        return formants
    
    bin_width = sampling_rate / (2 * nr)
    
    # Add nasal pole
    formants['fnp'] = {
        'time': formants['f1']['time'].copy(),
        'freq': formants['f1']['freq'].copy(),
        'amp': np.zeros_like(formants['f1']['amp']),
        'width': formants['f1']['width'].copy()
    }
    
    # Set nasal pole properties for nasalized segments
    formants['fnp']['amp'][nasalized_idx] = formants['f1']['amp'][nasalized_idx] * 2/3
    formants['fnp']['width'][nasalized_idx] = formants['f1']['width'][nasalized_idx] * 2/3
    
    # Adjust frequency: 250 Hz below or above F1 depending on F1 frequency
    f1_freq_hz = formants['f1']['freq'] * bin_width
    freq_adjustment = np.where(f1_freq_hz > 550, -250, 250) / bin_width
    formants['fnp']['freq'][nasalized_idx] = (formants['f1']['freq'][nasalized_idx] + 
                                             freq_adjustment[nasalized_idx])
    
    # Add nasal zero
    formants['fnz'] = {
        'time': formants['fnp']['time'].copy(),
        'freq': (formants['fnp']['freq'] + formants['f1']['freq']) / 2,
        'amp': np.zeros_like(formants['f1']['amp']),
        'width': formants['fnp']['width'].copy()
    }
    
    formants['fnz']['amp'][nasalized_idx] = -formants['f1']['amp'][nasalized_idx] * 2/3
    
    # Modify F1 for nasalized segments
    formants['f1']['amp'][nasalized_idx] *= 4/5
    formants['f1']['width'][nasalized_idx] *= 5/4
    
    return formants

def _generate_spectral_envelope(formants: Dict, nr: int, nc: int, 
                               formant_strength: float, sampling_rate: float) -> np.ndarray:
    """Generate the spectral envelope from formant specifications."""
    spectral_envelope = np.zeros((nr, nc))
    
    for formant_key in formants:
        formant = formants[formant_key]
        
        # Extract parameters
        freq_bins = formant['freq']  # Already in bin units
        width_bins = formant['width']  # Already in bin units
        amplitudes = formant['amp']
        
        # Generate formant for each time step
        for c in range(nc):
            if amplitudes[c] == 0:
                continue
                
            # Gamma distribution parameters
            mean_gamma = freq_bins[c]
            std_gamma = width_bins[c]
            
            if std_gamma <= 0:
                continue
                
            shape = mean_gamma**2 / std_gamma**2
            rate = mean_gamma / std_gamma**2
            
            # Skip if parameters are invalid
            if not (np.isfinite(shape) and np.isfinite(rate) and shape > 0 and rate > 0):
                continue
            
            # Generate gamma distribution
            bins = np.arange(1, nr + 1)
            formant_response = stats.gamma.pdf(bins, a=shape, scale=1/rate)
            
            # Normalize and scale
            if np.max(formant_response) > 0:
                formant_response = formant_response / np.max(formant_response) * amplitudes[c]
                spectral_envelope[:, c] += formant_response
    
    return spectral_envelope * formant_strength

def _add_lip_radiation(spectral_envelope: np.ndarray, rolloff_lip: float, 
                      mouth_opening: np.ndarray, mouth_open_binary: np.ndarray,
                      ampl_boost_db: float, nr: int, nc: int) -> np.ndarray:
    """Add lip radiation effects."""
    # Create lip radiation filter
    lip_db = rolloff_lip * np.log2(np.arange(1, nr + 1))
    
    # Apply for each time step
    for c in range(nc):
        spectral_envelope[:, c] = ((spectral_envelope[:, c] + 
                                   lip_db * mouth_open_binary[c]) *
                                  2**(mouth_opening[c] * ampl_boost_db / 10))
    
    return spectral_envelope