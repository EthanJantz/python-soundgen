"""
Python implementation of soundgen - parametric voice synthesis.

This package provides tools for parametric synthesis of sounds with harmonic 
and noise components, such as vocalizations of animals or human voice.

Main functions:
- generate_bout: Main synthesis function for generating vocal sounds
- generate_harmonics: Generate harmonic component
- generate_noise: Generate noise component  
- get_rolloff: Control harmonic rolloff
- get_spectral_envelope: Create formant filters

Example usage:
    import soundgen
    
    # Simple vowel synthesis
    sound = soundgen.generate_bout(
        pitch_anchors=[(0, 200), (1, 150)],
        exact_formants='a',
        syl_dur_mean=1000
    )
    
    # More complex synthesis with effects
    sound = soundgen.generate_bout(
        pitch_anchors=[(0, 100), (0.5, 200), (1, 80)],
        exact_formants='aeiou',
        jitter_dep=1.0,
        shimmer_dep=20,
        temperature=0.1,
        noise_amount=30
    )
"""

from .soundgen_main import generate_bout
from .synthesis import generate_harmonics, generate_noise
from .spectral import get_rolloff, get_spectral_envelope
from .parameters import (ParameterSet, FormantSpec, DEFAULT_FORMANTS, 
                        convert_string_to_formants, create_schwa_formants)
from .utils import (get_smooth_contour, get_glottal_cycles, get_random_walk,
                   cross_fade, fade_in_out, upsample)

__version__ = "0.1.0"
__author__ = "Python soundgen implementation"

# Main API functions
__all__ = [
    'generate_bout',
    'generate_harmonics', 
    'generate_noise',
    'get_rolloff',
    'get_spectral_envelope',
    'ParameterSet',
    'FormantSpec',
    'DEFAULT_FORMANTS',
    'convert_string_to_formants',
    'create_schwa_formants',
    'get_smooth_contour',
    'get_glottal_cycles',
    'get_random_walk',
    'cross_fade',
    'fade_in_out',
    'upsample'
]