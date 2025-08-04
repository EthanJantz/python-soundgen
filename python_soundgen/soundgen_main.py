"""
Main soundgen synthesis function and high-level API.
This is the Python equivalent of R's generateBout function.
"""

import numpy as np
from scipy.signal import stft, istft
from typing import List, Tuple, Optional, Union, Dict, Any
import warnings

from .parameters import (ParameterSet, convert_string_to_formants, 
                        create_schwa_formants, DEFAULT_FORMANTS, THROWAWAY_DB)
from .utils import (get_smooth_contour, add_vectors, match_lengths, 
                   wiggle_anchors, get_random_walk)
from .synthesis import generate_harmonics, generate_noise
from .spectral import get_spectral_envelope

def generate_bout(repeat_bout: int = 1,
                 n_syl: int = 1,
                 syl_dur_mean: float = 300,
                 pause_dur_mean: float = 200,
                 pitch_anchors: List[Tuple[float, float]] = None,
                 pitch_anchors_global: Optional[List[Tuple[float, float]]] = None,
                 temperature: float = 0.025,
                 male_female: float = 0,
                 creaky_breathy: float = 0,
                 noise_amount: float = 0,
                 noise_intensity: float = 50,
                 jitter_dep: float = 3,
                 jitter_length_ms: float = 1,
                 vibrato_freq: float = 5,
                 vibrato_dep: float = 0,
                 shimmer_dep: float = 0,
                 attack_len: float = 50,
                 rolloff_exp: float = -12,
                 rolloff_exp_delta: float = -12,
                 quadratic_delta: float = 0,
                 quadratic_n_harm: int = 3,
                 adjust_rolloff_per_khz: float = -6,
                 rolloff_lip: float = 6,
                 exact_formants: Union[str, List, None] = None,
                 formant_strength: float = 1,
                 extra_formants_ampl: float = 30,
                 vocal_tract_length: float = 15.5,
                 g0: float = 100,
                 sideband_width_hz: float = 100,
                 min_epoch_length_ms: float = 300,
                 trill_dep: float = 0,
                 trill_freq: float = 30,
                 breathing_anchors: Optional[List[Tuple[float, float]]] = None,
                 exact_formants_unvoiced: Optional[List] = None,
                 rolloff_breathing: float = -6,
                 mouth_anchors: List[Tuple[float, float]] = None,
                 ampl_anchors: Optional[List[Tuple[float, float]]] = None,
                 ampl_anchors_global: Optional[List[Tuple[float, float]]] = None,
                 sampling_rate: float = 16000,
                 window_length_points: int = 2048,
                 overlap: float = 75,
                 add_silence: float = 100,
                 pitch_floor: float = 50,
                 pitch_ceiling: float = 3500,
                 pitch_sampling_rate: float = 3500,
                 plot_spectro: bool = False,
                 play_sound: bool = False,
                 save_path: Optional[str] = None) -> np.ndarray:
    """
    Generate a bout of one or more syllables (equivalent to R's generateBout).
    
    This is the main synthesis function that generates vocalizations by combining
    harmonic and noise components with formant filtering.
    
    Args:
        repeat_bout: Number of times to repeat the bout
        n_syl: Number of syllables in the bout
        syl_dur_mean: Average syllable duration (ms)
        pause_dur_mean: Average pause duration between syllables (ms)
        pitch_anchors: Pitch contour anchors as (time, Hz) pairs
        pitch_anchors_global: Global pitch contour across syllables
        temperature: Amount of stochastic variation (0=deterministic)
        male_female: Gender adjustment (-1=male, +1=female)
        creaky_breathy: Voice quality (-1=creaky, +1=breathy)
        noise_amount: Proportion of sound with noise effects (0-100%)
        noise_intensity: Intensity of noise effects (0-100%)
        jitter_dep: Pitch jitter depth (semitones)
        jitter_length_ms: Duration of pitch jumps (ms)
        vibrato_freq: Vibrato frequency (Hz)
        vibrato_dep: Vibrato depth (semitones)
        shimmer_dep: Amplitude shimmer depth (0-100%)
        attack_len: Fade-in/out duration (ms)
        rolloff_exp: Basic rolloff (dB/octave)
        rolloff_exp_delta: Rolloff change per octave
        quadratic_delta: Quadratic adjustment for lower harmonics
        quadratic_n_harm: Number of harmonics affected by quadratic term
        adjust_rolloff_per_khz: Rolloff adjustment per kHz of f0
        rolloff_lip: Lip radiation effect (dB/octave)
        exact_formants: Formant specification (string or list)
        formant_strength: Formant amplitude scaling
        extra_formants_ampl: Amplitude of additional formants
        vocal_tract_length: Vocal tract length (cm)
        g0: Subharmonic target frequency (Hz)
        sideband_width_hz: Subharmonic sideband width (Hz)
        min_epoch_length_ms: Minimum epoch length for regime changes (ms)
        trill_dep: Amplitude modulation depth (0-1)
        trill_freq: Amplitude modulation frequency (Hz)
        breathing_anchors: Breathing noise anchors as (time, dB) pairs
        exact_formants_unvoiced: Formants for noise component
        rolloff_breathing: Noise rolloff (dB/octave)
        mouth_anchors: Mouth opening anchors as (time, opening) pairs
        ampl_anchors: Amplitude envelope anchors
        ampl_anchors_global: Global amplitude envelope
        sampling_rate: Audio sampling rate (Hz)
        window_length_points: FFT window length
        overlap: FFT window overlap (%)
        add_silence: Silence before/after bout (ms)
        pitch_floor: Minimum pitch (Hz)
        pitch_ceiling: Maximum pitch (Hz)
        pitch_sampling_rate: Pitch contour sampling rate (Hz)
        plot_spectro: Whether to plot spectrogram
        play_sound: Whether to play the sound
        save_path: Path to save audio file
        
    Returns:
        Generated audio waveform as numpy array
    """
    
    # Set default pitch anchors if not provided
    if pitch_anchors is None:
        pitch_anchors = [(0, 100), (0.1, 150), (0.9, 135), (1, 100)]
    
    # Set default mouth anchors if not provided
    if mouth_anchors is None:
        mouth_anchors = [(0, 0.5), (1, 0.5)]
    
    # Handle formant specification
    if exact_formants is None:
        exact_formants = create_schwa_formants(vocal_tract_length)
    elif isinstance(exact_formants, str):
        exact_formants = convert_string_to_formants(exact_formants)
    
    # Apply hyperparameter adjustments
    if creaky_breathy < 0:
        # Creaky voice adjustments
        noise_amount = min(100, noise_amount - creaky_breathy * 50)
        jitter_dep = max(0, jitter_dep - creaky_breathy / 2)
        shimmer_dep = max(0, shimmer_dep - creaky_breathy * 5)
        sideband_width_hz = sideband_width_hz * 2**(-creaky_breathy)
    elif creaky_breathy > 0:
        # Breathy voice adjustments - add breathing
        if breathing_anchors is None:
            breathing_anchors = [(0, THROWAWAY_DB), (syl_dur_mean, THROWAWAY_DB)]
        breathing_anchors = [(t, v + creaky_breathy * 120) for t, v in breathing_anchors]
        # Clamp breathing values
        breathing_anchors = [(t, min(v, 40)) for t, v in breathing_anchors]
    
    # Adjust rolloff for voice quality
    rolloff_exp = rolloff_exp - creaky_breathy * 10
    rolloff_exp_delta = rolloff_exp_delta - creaky_breathy * 5
    
    # Adjust g0 based on noise intensity
    g0 = 2 * (g0 - 50) / (1 + np.exp(-0.1 * (50 - noise_intensity))) + 50
    jitter_dep = 2 * jitter_dep / (1 + np.exp(0.1 * (50 - noise_intensity)))
    
    # Apply male/female adjustments
    if male_female != 0:
        # Adjust pitch (1 octave range)
        pitch_anchors = [(t, f * 2**male_female) for t, f in pitch_anchors]
        
        # Adjust formants (25% range)
        if isinstance(exact_formants, list):
            for formant in exact_formants:
                if hasattr(formant, 'freq'):
                    if isinstance(formant.freq, (int, float)):
                        formant.freq *= 1.25**male_female
                    else:
                        formant.freq = [f * 1.25**male_female for f in formant.freq]
        
        # Adjust vocal tract length (25% range)
        vocal_tract_length = vocal_tract_length * (1 - 0.25 * male_female)
    
    # Generate syllable timing
    syllables = _divide_into_syllables(n_syl, syl_dur_mean, pause_dur_mean, temperature)
    
    # Calculate pitch deltas for global contour
    if pitch_anchors_global is not None and n_syl > 1:
        pitch_deltas = 2**(get_smooth_contour(pitch_anchors_global, n_syl) / 12)
    else:
        pitch_deltas = np.ones(n_syl)
    
    # Parameters that can vary per syllable
    params_to_vary = [
        'noise_intensity', 'attack_len', 'jitter_dep', 'shimmer_dep',
        'rolloff_exp', 'rolloff_exp_delta', 'formant_strength',
        'min_epoch_length_ms', 'g0', 'sideband_width_hz'
    ]
    
    # Generate bout
    bout_parts = []
    
    for b in range(repeat_bout):
        voiced_parts = []
        unvoiced_parts = []
        syllable_start_indices = []
        
        # Generate each syllable
        for s in range(len(syllables)):
            syl_start, syl_end = syllables[s]['start'], syllables[s]['end']
            syl_dur = syl_end - syl_start
            
            # Apply temperature variation to parameters
            syl_params = _vary_parameters_per_syllable(
                locals(), params_to_vary, temperature
            )
            
            # Wiggle anchors if temperature > 0
            pitch_anchors_syl = pitch_anchors.copy()
            ampl_anchors_syl = ampl_anchors.copy() if ampl_anchors else None
            
            if temperature > 0:
                pitch_anchors_syl = wiggle_anchors(
                    pitch_anchors_syl, temperature, 
                    low=(0, pitch_floor), high=(1, pitch_ceiling),
                    temp_coef=15
                )
                if ampl_anchors_syl:
                    ampl_anchors_syl = wiggle_anchors(
                        ampl_anchors_syl, temperature,
                        low=(0, 0), high=(1, -THROWAWAY_DB),
                        temp_coef=10
                    )
            
            # Generate pitch contour for this syllable
            pitch_samples = int(syl_dur * pitch_sampling_rate / 1000)
            pitch_contour = get_smooth_contour(
                pitch_anchors_syl, pitch_samples, sampling_rate=pitch_sampling_rate,
                value_floor=pitch_floor, value_ceiling=pitch_ceiling, this_is_pitch=True
            ) * pitch_deltas[s]
            
            # Generate voiced part
            if syl_dur >= 10:  # Minimum syllable duration
                syllable = generate_harmonics(
                    pitch=pitch_contour,
                    attack_len=syl_params['attack_len'],
                    noise_amount=noise_amount,
                    noise_intensity=syl_params['noise_intensity'],
                    jitter_dep=syl_params['jitter_dep'],
                    jitter_length_ms=jitter_length_ms,
                    vibrato_freq=vibrato_freq,
                    vibrato_dep=vibrato_dep,
                    shimmer_dep=syl_params['shimmer_dep'],
                    creaky_breathy=creaky_breathy,
                    rolloff_exp=syl_params['rolloff_exp'],
                    rolloff_exp_delta=syl_params['rolloff_exp_delta'],
                    adjust_rolloff_per_khz=adjust_rolloff_per_khz,
                    quadratic_delta=quadratic_delta,
                    quadratic_n_harm=quadratic_n_harm,
                    formant_strength=syl_params['formant_strength'],
                    temperature=temperature,
                    min_epoch_length_ms=syl_params['min_epoch_length_ms'],
                    g0=syl_params['g0'],
                    sideband_width_hz=syl_params['sideband_width_hz'],
                    rolloff_lip=rolloff_lip,
                    trill_dep=trill_dep,
                    trill_freq=trill_freq,
                    ampl_anchors=ampl_anchors_syl,
                    overlap=overlap,
                    window_length_points=window_length_points,
                    sampling_rate=sampling_rate,
                    pitch_floor=pitch_floor,
                    pitch_ceiling=pitch_ceiling,
                    pitch_sampling_rate=pitch_sampling_rate
                )
            else:
                syllable = np.zeros(int(syl_dur * sampling_rate / 1000))
            
            voiced_parts.append(syllable)
            
            # Calculate syllable start index for noise insertion
            if s == 0:
                start_idx = 0
            else:
                start_idx = sum(len(voiced_parts[i]) for i in range(s))
                # Add pause length
                if s < len(syllables):
                    pause_samples = int((syllables[s]['start'] - syllables[s-1]['end']) * 
                                      sampling_rate / 1000)
                    start_idx += pause_samples
            
            syllable_start_indices.append(start_idx)
            
            # Generate unvoiced part if breathing specified
            if breathing_anchors is not None:
                breathing_dur = int(syl_dur * sampling_rate / 1000)
                
                # Create spectral envelope for unvoiced component
                if exact_formants_unvoiced is None:
                    spectral_envelope_unvoiced = None
                else:
                    spectral_envelope_unvoiced = get_spectral_envelope(
                        nr=window_length_points // 2,
                        nc=max(1, breathing_dur // (sampling_rate // 100)),  # ~10ms frames
                        exact_formants=exact_formants_unvoiced,
                        formant_strength=formant_strength,
                        rolloff_lip=rolloff_lip,
                        mouth_anchors=mouth_anchors,
                        vocal_tract_length=vocal_tract_length,
                        temperature=temperature,
                        sampling_rate=sampling_rate
                    )
                
                unvoiced_syl = generate_noise(
                    length=breathing_dur,
                    breathing_anchors=breathing_anchors,
                    rolloff_breathing=rolloff_breathing,
                    attack_len=attack_len,
                    window_length_points=window_length_points,
                    sampling_rate=sampling_rate,
                    overlap=overlap,
                    filter_breathing=spectral_envelope_unvoiced
                )
                unvoiced_parts.append((start_idx, unvoiced_syl))
        
        # Combine syllables with pauses
        voiced_sound = _combine_syllables_with_pauses(voiced_parts, syllables, sampling_rate)
        
        # Apply global amplitude envelope if specified
        if ampl_anchors_global is not None:
            ampl_global_linear = [(t, 2**(v/10)) for t, v in ampl_anchors_global 
                                 if v > THROWAWAY_DB]
            if ampl_global_linear:
                ampl_envelope = get_smooth_contour(
                    ampl_global_linear, len(voiced_sound), sampling_rate=sampling_rate,
                    value_floor=0, value_ceiling=2**(-THROWAWAY_DB/10)
                )
                voiced_sound = voiced_sound * ampl_envelope
        
        # Apply formant filtering
        if len(voiced_sound) > 0:
            filtered_sound = _apply_formant_filtering(
                voiced_sound, exact_formants, formant_strength, extra_formants_ampl,
                rolloff_lip, mouth_anchors, temperature, vocal_tract_length,
                window_length_points, overlap, sampling_rate
            )
        else:
            filtered_sound = voiced_sound
        
        # Add unvoiced components
        if exact_formants_unvoiced is None and unvoiced_parts:
            # Mix before filtering (breathing-like noise)
            for start_idx, unvoiced_syl in unvoiced_parts:
                filtered_sound = add_vectors(filtered_sound, unvoiced_syl, start_idx)
        elif unvoiced_parts:
            # Add after filtering (different formants for noise)
            for start_idx, unvoiced_syl in unvoiced_parts:
                filtered_sound = add_vectors(filtered_sound, unvoiced_syl, start_idx)
        
        # Apply trill (amplitude modulation)
        if trill_dep > 0:
            trill_modulation = 1 - np.sin(2 * np.pi * np.arange(len(filtered_sound)) / 
                                         sampling_rate * trill_freq) * trill_dep / 2
            filtered_sound = filtered_sound * trill_modulation
        
        bout_parts.append(filtered_sound)
    
    # Combine multiple bouts with pauses
    if len(bout_parts) == 1:
        bout = bout_parts[0]
    else:
        pause_samples = int(pause_dur_mean * sampling_rate / 1000)
        bout = bout_parts[0]
        for part in bout_parts[1:]:
            bout = np.concatenate([bout, np.zeros(pause_samples), part])
    
    # Add silence before and after
    if add_silence > 0:
        silence_samples = int(add_silence * sampling_rate / 1000)
        bout = np.concatenate([np.zeros(silence_samples), bout, np.zeros(silence_samples)])
    
    # Handle output options
    if play_sound:
        try:
            import sounddevice as sd
            sd.play(bout, sampling_rate)
        except ImportError:
            warnings.warn("sounddevice not available for audio playback")
    
    if save_path:
        try:
            import soundfile as sf
            sf.write(save_path, bout, int(sampling_rate))
        except ImportError:
            warnings.warn("soundfile not available for saving audio")
    
    if plot_spectro:
        try:
            import matplotlib.pyplot as plt
            from scipy.signal import spectrogram
            f, t, Sxx = spectrogram(bout, sampling_rate, nperseg=1024)
            plt.figure(figsize=(10, 6))
            plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.title('Spectrogram')
            plt.colorbar(label='Power [dB]')
            plt.show()
        except ImportError:
            warnings.warn("matplotlib not available for spectrogram plotting")
    
    return bout

def _divide_into_syllables(n_syl: int, syl_dur_mean: float, pause_dur_mean: float, 
                          temperature: float) -> List[Dict]:
    """Generate syllable timing information."""
    syllables = []
    current_time = 0
    
    for i in range(n_syl):
        # Add some variation to durations if temperature > 0
        if temperature > 0:
            syl_dur = np.random.normal(syl_dur_mean, syl_dur_mean * temperature / 10)
            syl_dur = np.clip(syl_dur, 10, 5000)  # Reasonable bounds
        else:
            syl_dur = syl_dur_mean
        
        syllables.append({
            'start': current_time,
            'end': current_time + syl_dur
        })
        
        current_time += syl_dur
        
        # Add pause after syllable (except for last one)
        if i < n_syl - 1:
            if temperature > 0:
                pause_dur = np.random.normal(pause_dur_mean, pause_dur_mean * temperature / 10)
                pause_dur = np.clip(pause_dur, 10, 1000)
            else:
                pause_dur = pause_dur_mean
            current_time += pause_dur
    
    return syllables

def _vary_parameters_per_syllable(params: Dict, params_to_vary: List[str], 
                                 temperature: float) -> Dict:
    """Apply temperature-based variation to syllable parameters."""
    varied_params = {}
    
    if temperature <= 0:
        return {key: params[key] for key in params_to_vary}
    
    for param_name in params_to_vary:
        if param_name in params:
            current_value = params[param_name]
            
            # Simple variation - could be improved with proper bounds
            variation = np.random.normal(0, abs(current_value) * temperature / 10)
            new_value = current_value + variation
            
            # Apply some basic bounds
            if param_name in ['noise_intensity', 'shimmer_dep']:
                new_value = np.clip(new_value, 0, 100)
            elif param_name in ['attack_len', 'g0', 'sideband_width_hz']:
                new_value = max(1, new_value)
                if param_name in ['attack_len', 'g0', 'sideband_width_hz']:
                    new_value = int(round(new_value))
            
            varied_params[param_name] = new_value
        else:
            varied_params[param_name] = 0  # Default value
    
    return varied_params

def _combine_syllables_with_pauses(syllables: List[np.ndarray], 
                                  syllable_info: List[Dict],
                                  sampling_rate: float) -> np.ndarray:
    """Combine syllables with appropriate pauses."""
    if not syllables:
        return np.array([])
    
    if len(syllables) == 1:
        return syllables[0]
    
    combined = []
    for i, syllable in enumerate(syllables):
        combined.append(syllable)
        
        # Add pause after syllable (except last one)
        if i < len(syllables) - 1:
            pause_dur = syllable_info[i+1]['start'] - syllable_info[i]['end']
            pause_samples = int(pause_dur * sampling_rate / 1000)
            if pause_samples > 0:
                combined.append(np.zeros(pause_samples))
    
    return np.concatenate(combined)

def _apply_formant_filtering(sound: np.ndarray, exact_formants: List,
                           formant_strength: float, extra_formants_ampl: float,
                           rolloff_lip: float, mouth_anchors: List,
                           temperature: float, vocal_tract_length: float,
                           window_length_points: int, overlap: float,
                           sampling_rate: float) -> np.ndarray:
    """Apply formant filtering using FFT."""
    if len(sound) == 0 or np.sum(np.abs(sound)) == 0:
        return sound
    
    # Adjust window length for short sounds
    window_length = min(window_length_points, len(sound) // 2)
    if window_length < 64:  # Minimum reasonable window size
        return sound
    
    # Calculate STFT parameters
    hop_length = int(window_length * (1 - overlap/100))
    
    # Compute STFT
    f, t, Zxx = stft(sound, fs=sampling_rate, window='hann', 
                     nperseg=window_length, noverlap=window_length-hop_length)
    
    nr, nc = Zxx.shape
    
    # Generate spectral envelope
    spectral_envelope = get_spectral_envelope(
        nr=nr, nc=nc,
        exact_formants=exact_formants,
        formant_strength=formant_strength,
        extra_formants_ampl=extra_formants_ampl,
        rolloff_lip=rolloff_lip,
        mouth_anchors=mouth_anchors,
        vocal_tract_length=vocal_tract_length,
        temperature=temperature,
        sampling_rate=sampling_rate
    )
    
    # Apply filtering
    filtered_Zxx = Zxx * spectral_envelope
    
    # Inverse STFT
    _, filtered_sound = istft(filtered_Zxx, fs=sampling_rate, window='hann',
                             nperseg=window_length, noverlap=window_length-hop_length)
    
    # Normalize
    if np.max(np.abs(filtered_sound)) > 0:
        filtered_sound = filtered_sound / np.max(np.abs(filtered_sound))
    
    return filtered_sound