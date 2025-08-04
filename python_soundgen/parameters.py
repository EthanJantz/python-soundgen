"""
Parameter management for soundgen Python implementation.
Handles parameter validation, bounds checking, and default presets.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Optional, Any

# Default parameter bounds (from R's permittedValues)
PERMITTED_VALUES = {
    'sylDur_mean': {'low': 10, 'high': 5000},
    'pauseDur_mean': {'low': 10, 'high': 1000}, 
    'pitch': {'low': 50, 'high': 3500},
    'breathing_ampl': {'low': -120, 'high': 40},
    'mouthOpening': {'low': 0, 'high': 1},
    'rolloff_exp': {'low': -30, 'high': 30},
    'rolloff_exp_delta': {'low': -30, 'high': 30},
    'formantStrength': {'low': 0, 'high': 3},
    'min_epoch_length_ms': {'low': 50, 'high': 1000},
    'g0': {'low': 25, 'high': 1000},
    'sideband_width_hz': {'low': 10, 'high': 1000},
    'noiseIntensity': {'low': 0, 'high': 100},
    'attackLen': {'low': 1, 'high': 1000},
    'jitterDep': {'low': 0, 'high': 12},
    'shimmerDep': {'low': 0, 'high': 100}
}

# Physical constants
SPEED_SOUND = 35400  # cm/s, speed of sound in warm air
THROWAWAY_DB = -120  # discard harmonics weaker than this

# Temperature-related scaling factors (from R sysdata)
PITCH_ANCHORS_WIGGLE_PER_TEMP = 15
BREATHING_ANCHORS_WIGGLE_PER_TEMP = 25  
AMPL_ANCHORS_WIGGLE_PER_TEMP = 10
FORMANT_DRIFT_PER_TEMP = 0.15
FORMANT_DISPERSION_PER_TEMP = 0.25
PITCH_DRIFT_PER_TEMP = 0.3
PITCH_DRIFT_WIGGLE_PER_TEMP = 0.5
ROLLOFF_PER_AMPL = 15
RANDOM_WALK_TREND_STRENGTH = 0.5

@dataclass
class FormantSpec:
    """Specification for a single formant."""
    time: Union[float, List[float]]
    freq: Union[float, List[float]] 
    amp: Union[float, List[float]]
    width: Union[float, List[float]]

# Default formant presets for different vowels (M1 speaker)
DEFAULT_FORMANTS = {
    'a': [
        FormantSpec(time=0, freq=860, amp=30, width=120),   # F1
        FormantSpec(time=0, freq=1280, amp=40, width=120),  # F2  
        FormantSpec(time=0, freq=2900, amp=25, width=200),  # F3
        FormantSpec(time=0, freq=4200, amp=20, width=300)   # F4
    ],
    'e': [
        FormantSpec(time=0, freq=530, amp=30, width=80),
        FormantSpec(time=0, freq=1840, amp=35, width=120),
        FormantSpec(time=0, freq=2480, amp=25, width=200),
        FormantSpec(time=0, freq=4200, amp=20, width=300)
    ],
    'i': [
        FormantSpec(time=0, freq=270, amp=30, width=80),
        FormantSpec(time=0, freq=2300, amp=35, width=100),
        FormantSpec(time=0, freq=3010, amp=25, width=200),
        FormantSpec(time=0, freq=4200, amp=20, width=300)
    ],
    'o': [
        FormantSpec(time=0, freq=400, amp=30, width=80),
        FormantSpec(time=0, freq=800, amp=35, width=120),
        FormantSpec(time=0, freq=2830, amp=25, width=200),
        FormantSpec(time=0, freq=4200, amp=20, width=300)
    ],
    'u': [
        FormantSpec(time=0, freq=320, amp=30, width=80),
        FormantSpec(time=0, freq=800, amp=35, width=120),
        FormantSpec(time=0, freq=2560, amp=25, width=200),
        FormantSpec(time=0, freq=4200, amp=20, width=300)
    ],
    'schwa': [
        FormantSpec(time=0, freq=500, amp=30, width=100),
        FormantSpec(time=0, freq=1500, amp=30, width=150),
        FormantSpec(time=0, freq=2500, amp=25, width=200)
    ]
}

def rnorm_bounded(n: int, mean: float, sd: float, low: float, high: float, 
                  round_to_integer: bool = False) -> Union[float, np.ndarray]:
    """
    Generate bounded normal random values (equivalent to R's rnorm_bounded).
    
    Args:
        n: Number of values to generate
        mean: Mean of distribution
        sd: Standard deviation 
        low: Lower bound
        high: Upper bound
        round_to_integer: Whether to round result to integers
    
    Returns:
        Single value if n=1, array if n>1
    """
    if sd <= 0:
        result = np.full(n, mean)
    else:
        # Generate values and clip to bounds
        result = np.random.normal(mean, sd, n)
        result = np.clip(result, low, high)
    
    if round_to_integer:
        result = np.round(result).astype(int)
    
    return result[0] if n == 1 else result

def validate_parameter(name: str, value: Any) -> Any:
    """
    Validate parameter value against permitted bounds.
    
    Args:
        name: Parameter name
        value: Parameter value
        
    Returns:
        Validated (possibly clipped) value
        
    Raises:
        ValueError: If parameter name not recognized
    """
    if name not in PERMITTED_VALUES:
        return value  # No validation for unknown parameters
    
    bounds = PERMITTED_VALUES[name]
    
    if isinstance(value, (int, float)):
        return np.clip(value, bounds['low'], bounds['high'])
    elif isinstance(value, (list, np.ndarray)):
        return np.clip(value, bounds['low'], bounds['high'])
    else:
        return value

def convert_string_to_formants(formant_string: str) -> List[FormantSpec]:
    """
    Convert vowel string to formant specifications.
    
    Args:
        formant_string: String of vowel characters like 'aeiou'
        
    Returns:
        List of formant specifications
        
    Raises:
        ValueError: If unknown vowel character encountered
    """
    if len(formant_string) == 1:
        if formant_string in DEFAULT_FORMANTS:
            return DEFAULT_FORMANTS[formant_string]
        else:
            raise ValueError(f"Unknown vowel: {formant_string}")
    
    # For multiple vowels, interpolate between them
    # For now, just return the first vowel's formants
    # TODO: Implement proper vowel sequence interpolation
    first_vowel = formant_string[0]
    if first_vowel in DEFAULT_FORMANTS:
        return DEFAULT_FORMANTS[first_vowel]
    else:
        raise ValueError(f"Unknown vowel: {first_vowel}")

def create_schwa_formants(vocal_tract_length: float = 15.5) -> List[FormantSpec]:
    """
    Create schwa formants based on vocal tract length.
    
    Args:
        vocal_tract_length: Length of vocal tract in cm
        
    Returns:
        List of formant specifications for schwa
    """
    freq = SPEED_SOUND / (4 * vocal_tract_length)
    width = 50 + (np.log2(freq) - 5) * 20
    
    return [
        FormantSpec(time=0, freq=freq, amp=30, width=width)
    ]

class ParameterSet:
    """Container for synthesis parameters with validation and wiggling."""
    
    def __init__(self, **kwargs):
        """Initialize with default values, overridden by kwargs."""
        
        # Default parameter values (from R generateBout function)
        self.repeat_bout = 1
        self.n_syl = 1
        self.syl_dur_mean = 300
        self.pause_dur_mean = 200
        self.pitch_anchors = [(0, 100), (0.1, 150), (0.9, 135), (1, 100)]
        self.pitch_anchors_global = None
        self.temperature = 0.025
        self.male_female = 0
        self.creaky_breathy = 0
        self.noise_amount = 0
        self.noise_intensity = 50
        self.jitter_dep = 3
        self.jitter_length_ms = 1
        self.vibrato_freq = 5
        self.vibrato_dep = 0
        self.shimmer_dep = 0
        self.attack_len = 50
        self.rolloff_exp = -12
        self.rolloff_exp_delta = -12
        self.quadratic_delta = 0
        self.quadratic_n_harm = 3
        self.adjust_rolloff_per_khz = -6
        self.rolloff_lip = 6
        self.exact_formants = DEFAULT_FORMANTS['schwa']
        self.formant_strength = 1
        self.extra_formants_ampl = 30
        self.vocal_tract_length = 15.5
        self.g0 = 100
        self.sideband_width_hz = 100
        self.min_epoch_length_ms = 300
        self.trill_dep = 0
        self.trill_freq = 30
        self.breathing_anchors = None
        self.exact_formants_unvoiced = None
        self.rolloff_breathing = -6
        self.mouth_anchors = [(0, 0.5), (1, 0.5)]
        self.ampl_anchors = None
        self.ampl_anchors_global = None
        self.sampling_rate = 16000
        self.window_length_points = 2048
        self.overlap = 75
        self.add_silence = 100
        self.pitch_floor = 50
        self.pitch_ceiling = 3500
        self.pitch_sampling_rate = 3500
        
        # Update with provided values
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Convert camelCase to snake_case for compatibility
                snake_key = ''.join(['_' + c.lower() if c.isupper() else c for c in key]).lstrip('_')
                if hasattr(self, snake_key):
                    setattr(self, snake_key, value)
    
    def validate_all(self):
        """Validate all parameters against permitted bounds."""
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                value = getattr(self, attr_name)
                validated_value = validate_parameter(attr_name, value)
                setattr(self, attr_name, validated_value)
    
    def apply_temperature_wiggle(self, parameters_to_vary: List[str]):
        """Apply temperature-based random variation to specified parameters."""
        if self.temperature <= 0:
            return
            
        for param_name in parameters_to_vary:
            if not hasattr(self, param_name):
                continue
                
            current_value = getattr(self, param_name)
            if param_name in PERMITTED_VALUES:
                bounds = PERMITTED_VALUES[param_name]
                low, high = bounds['low'], bounds['high']
                sd = (high - low) * self.temperature / 10
                
                should_round = param_name in ['attack_len', 'g0', 'sideband_width_hz']
                new_value = rnorm_bounded(1, current_value, sd, low, high, should_round)
                setattr(self, param_name, new_value)