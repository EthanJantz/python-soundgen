# Python Soundgen

A Python implementation of the soundgen parametric voice synthesis algorithm, originally developed in R by Andrey Anikin.

## Overview

This package provides tools for parametric synthesis of sounds with harmonic and noise components, such as vocalizations of animals or human voice. It implements a source-filter model with three main components:

1. **Harmonic Component Generation** - Creates the glottal source with controllable pitch, jitter, shimmer
2. **Noise Component Generation** - Adds aspiration noise and breathing sounds  
3. **Spectral Filtering** - Applies formant filtering and lip radiation effects

## Installation

```bash
# Required dependencies
pip install numpy scipy matplotlib

# Optional dependencies for audio I/O
pip install soundfile sounddevice
```

## Quick Start

```python
import soundgen

# Generate a simple vowel sound
sound = soundgen.generate_bout(
    pitch_anchors=[(0, 200), (1, 150)],  # Pitch contour: start at 200Hz, end at 150Hz
    exact_formants='a',                   # Use vowel [a] formants
    syl_dur_mean=1000,                   # 1 second duration
    sampling_rate=16000
)

# Save to file (requires soundfile)
import soundfile as sf
sf.write('vowel_a.wav', sound, 16000)

# Play sound (requires sounddevice) 
import sounddevice as sd
sd.play(sound, 16000)
```

## Key Features

### Basic Synthesis Parameters

- **Pitch Control**: Specify pitch contours with anchor points
- **Formant Control**: Use preset vowels ('a', 'e', 'i', 'o', 'u') or custom formant specifications
- **Duration Control**: Set syllable and pause durations
- **Voice Quality**: Adjust creaky/breathy voice characteristics

### Advanced Effects

- **Jitter**: Random pitch variation for rough/harsh voice quality
- **Shimmer**: Random amplitude variation 
- **Vibrato**: Regular pitch modulation
- **Vocal Fry**: Subharmonic generation for creaky voice
- **Breathing**: Aspiration noise component
- **Temperature**: Overall stochasticity level

### Multi-Syllable Synthesis

```python
# Generate 3-syllable vocalization
sound = soundgen.generate_bout(
    n_syl=3,
    syl_dur_mean=400,
    pause_dur_mean=150,
    pitch_anchors=[(0, 180), (0.5, 220), (1, 160)],
    pitch_anchors_global=[(0, -3), (0.5, 5), (1, -2)],  # Global pitch modulation
    exact_formants='aiu',
    temperature=0.1
)
```

### Voice Characteristics

```python
# Male voice (lower pitch, larger vocal tract)
male_sound = soundgen.generate_bout(
    pitch_anchors=[(0, 120), (1, 100)],
    exact_formants='a',
    male_female=-0.5,  # Negative = more male
    vocal_tract_length=17.5
)

# Female voice (higher pitch, smaller vocal tract)  
female_sound = soundgen.generate_bout(
    pitch_anchors=[(0, 120), (1, 100)],
    exact_formants='a', 
    male_female=0.5,   # Positive = more female
    vocal_tract_length=13.5
)
```

### Noisy/Animal-like Vocalizations

```python
# Rough, animal-like call with subharmonics
sound = soundgen.generate_bout(
    pitch_anchors=[(0, 200), (0.5, 350), (1, 180)],
    noise_amount=80,        # High noise amount
    g0=150,                 # Subharmonic frequency
    sideband_width_hz=200,  # Subharmonic bandwidth
    jitter_dep=1.5,         # Strong pitch jitter
    temperature=0.15,       # High stochasticity
    vocal_tract_length=8    # Short vocal tract
)
```

## API Reference

### Main Functions

- `generate_bout()` - Main synthesis function
- `generate_harmonics()` - Generate harmonic component only
- `generate_noise()` - Generate noise component only

### Utility Functions

- `get_rolloff()` - Calculate harmonic amplitude rolloff
- `get_spectral_envelope()` - Create formant filter
- `get_smooth_contour()` - Interpolate anchor points
- `convert_string_to_formants()` - Convert vowel strings to formant specs

### Parameter Classes

- `ParameterSet` - Container for synthesis parameters
- `FormantSpec` - Formant specification dataclass

## Examples

The package includes comprehensive examples in `examples.py`:

```python
from soundgen.examples import run_all_examples

# Run all examples and save to files
sounds = run_all_examples(play=False, save_dir='output')

# Individual examples
from soundgen.examples import (
    basic_vowel_example,
    noisy_vocalization_example, 
    creaky_voice_example,
    multisyllable_example
)

sound = basic_vowel_example(play=True)
```

## Parameter Reference

### Pitch Parameters
- `pitch_anchors`: List of (time, frequency) points defining pitch contour
- `pitch_floor/ceiling`: Pitch bounds (Hz)
- `vibrato_freq/dep`: Vibrato frequency (Hz) and depth (semitones)

### Voice Quality Parameters  
- `temperature`: Overall stochasticity (0=deterministic, 0.1=moderate variation)
- `creaky_breathy`: Voice quality (-1=creaky, +1=breathy)
- `jitter_dep`: Random pitch variation (semitones)
- `shimmer_dep`: Random amplitude variation (0-100%)

### Formant Parameters
- `exact_formants`: Vowel string ('a', 'aeiou') or formant specifications
- `formant_strength`: Formant amplitude scaling
- `vocal_tract_length`: Length in cm (affects formant frequencies)

### Noise Parameters
- `noise_amount`: Proportion of sound with noise effects (0-100%)
- `breathing_anchors`: Breathing noise contour
- `g0`: Subharmonic target frequency (Hz)
- `sideband_width_hz`: Subharmonic bandwidth (Hz)

## Algorithm Details

The synthesis follows these steps:

1. **Pitch Processing**: Apply vibrato, jitter, and drift to base pitch contour
2. **Harmonic Generation**: Create sine waves for each harmonic with amplitude rolloff
3. **Noise Generation**: Create filtered noise component if specified  
4. **Formant Filtering**: Apply spectral envelope using FFT/IFFT
5. **Post-processing**: Add amplitude modulation, cross-fade epochs

The implementation preserves the mathematical foundations of the original R version while leveraging Python's scientific computing ecosystem (NumPy, SciPy) for efficient computation.

## Differences from R Version

- Simplified vocal fry (subharmonics) implementation
- Uses SciPy's STFT/ISTFT instead of custom FFT routines
- Some advanced features (morphing, detailed acoustic analysis) not yet implemented
- Python-style parameter naming (snake_case vs camelCase)

## Contributing

This is a research implementation. Contributions welcome for:
- Enhanced vocal fry modeling
- Additional formant presets
- Performance optimizations
- Missing R features

## References

- Original R soundgen: https://github.com/tatters/soundgen
- Algorithm documentation: https://cogsci.se/soundgen/algorithm.html
- Anikin, A. (2019). Soundgen: An open-source tool for synthesizing nonlinguistic vocalizations. Behavior Research Methods, 51(2), 778-792.

## License

This implementation follows the GPL license of the original R package.