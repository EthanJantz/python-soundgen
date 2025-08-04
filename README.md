# Python Soundgen

A Python implementation of parametric voice synthesis, recreating the functionality of the R soundgen package. Generate realistic vocal sounds with controllable pitch, formants, and voice effects.

## Overview

This package implements a **source-filter model** to synthesize vocalizations with three main components:

1. **Harmonic Component** - Glottal source with controllable pitch and effects
2. **Noise Component** - Aspiration noise and breathing sounds  
3. **Spectral Filtering** - Formant filtering to simulate vocal tract resonance

Perfect for generating animal vocalizations, human voice synthesis, and acoustic research.

## Installation

```bash
# Clone repository
git clone <repository-url>
cd soundgen

# Install with dependencies using uv
uv sync

# Or install manually
pip install numpy scipy matplotlib soundfile sounddevice
```

## Quick Start

```python
import python_soundgen as soundgen

# Generate a simple vowel sound
sound = soundgen.generate_bout(
    pitch_anchors=[(0, 200), (1, 150)],  # Falling pitch: 200→150 Hz
    exact_formants='a',                   # Vowel [a]
    syl_dur_mean=500,                    # 500ms duration
    sampling_rate=16000
)

# Save to file
import soundfile as sf
sf.write('vowel_a.wav', sound, 16000)

# Play sound
import sounddevice as sd
sd.play(sound, 16000)
```

## Core API

### Main Function: `generate_bout()`

The primary synthesis function with extensive parameter control:

```python
sound = soundgen.generate_bout(
    # === Pitch Control ===
    pitch_anchors=[(0, 100), (0.5, 200), (1, 150)],  # (time, frequency) pairs
    pitch_floor=50,                      # Minimum pitch (Hz)
    pitch_ceiling=3500,                  # Maximum pitch (Hz)
    
    # === Formants (Vocal Tract) ===
    exact_formants='a',                  # Vowel: 'a', 'e', 'i', 'o', 'u'
    # OR custom: [FormantSpec(freq=800, amp=30, width=120), ...]
    vocal_tract_length=15.5,             # Vocal tract length (cm)
    
    # === Timing ===
    syl_dur_mean=500,                    # Syllable duration (ms)
    n_syl=1,                            # Number of syllables
    pause_dur_mean=200,                  # Pause between syllables (ms)
    
    # === Voice Effects ===
    jitter_dep=0,                       # Pitch jitter (semitones)
    shimmer_dep=0,                      # Amplitude shimmer (0-100%)
    vibrato_freq=5,                     # Vibrato frequency (Hz)
    vibrato_dep=0,                      # Vibrato depth (semitones)
    
    # === Voice Quality ===
    creaky_breathy=0,                   # -1=creaky, +1=breathy
    noise_amount=0,                     # Noise/subharmonics (0-100%)
    temperature=0,                      # Stochasticity (0-1)
    
    # === Technical ===
    sampling_rate=16000,                # Audio sample rate
    add_silence=100                     # Silence padding (ms)
)
```

## Usage Examples

### Basic Vowel Sounds

```python
# Generate different vowels
vowels = ['a', 'e', 'i', 'o', 'u']
for vowel in vowels:
    sound = soundgen.generate_bout(
        pitch_anchors=[(0, 200), (1, 180)],
        exact_formants=vowel,
        syl_dur_mean=400
    )
    sf.write(f'vowel_{vowel}.wav', sound, 16000)
```

### Pitch Contours

```python
# Rising pitch
sound = soundgen.generate_bout(
    pitch_anchors=[(0, 100), (1, 300)],
    exact_formants='a',
    syl_dur_mean=800
)

# Complex pitch curve
sound = soundgen.generate_bout(
    pitch_anchors=[(0, 150), (0.3, 250), (0.7, 200), (1, 120)],
    exact_formants='o',
    syl_dur_mean=1000
)
```

### Voice Effects

```python
# Vibrato (regular pitch modulation)
sound = soundgen.generate_bout(
    pitch_anchors=[(0, 150), (1, 150)],
    exact_formants='a',
    vibrato_freq=6,      # 6 Hz vibrato
    vibrato_dep=1.0,     # 1 semitone depth
    syl_dur_mean=800
)

# Jittery/rough voice
sound = soundgen.generate_bout(
    pitch_anchors=[(0, 140), (1, 130)],
    exact_formants='e',
    jitter_dep=1.0,      # Random pitch variation
    shimmer_dep=20,      # Random amplitude variation
    noise_amount=30,     # Add subharmonics
    temperature=0.1      # General stochasticity
)

# Breathy voice
sound = soundgen.generate_bout(
    pitch_anchors=[(0, 160), (1, 140)],
    exact_formants='a',
    creaky_breathy=0.7,  # Positive = breathy
    breathing_anchors=[(0, -20), (500, -15), (1000, -25)]  # Aspiration noise
)
```

### Multi-Syllable Sounds

```python
# Two-syllable vocalization
sound = soundgen.generate_bout(
    n_syl=2,                            # 2 syllables
    syl_dur_mean=300,                   # 300ms each
    pause_dur_mean=100,                 # 100ms pause
    pitch_anchors=[(0, 180), (1, 140)], # Pitch per syllable
    pitch_anchors_global=[(0, 0), (1, -3)], # Overall trend (semitones)
    exact_formants='au'                 # Vowel sequence
)

# Animal-like call
sound = soundgen.generate_bout(
    n_syl=3,
    syl_dur_mean=200,
    pause_dur_mean=80,
    pitch_anchors=[(0, 300), (0.5, 500), (1, 250)],
    noise_amount=60,
    g0=150,                            # Subharmonic frequency
    sideband_width_hz=200,             # Subharmonic bandwidth
    temperature=0.15,
    vocal_tract_length=8               # Shorter vocal tract
)
```

### Custom Formants

```python
# Manual formant specification
from python_soundgen import FormantSpec

custom_formants = [
    FormantSpec(time=0, freq=800, amp=35, width=100),   # F1
    FormantSpec(time=0, freq=1200, amp=30, width=120),  # F2
    FormantSpec(time=0, freq=2800, amp=25, width=200)   # F3
]

sound = soundgen.generate_bout(
    pitch_anchors=[(0, 180), (1, 160)],
    exact_formants=custom_formants,
    syl_dur_mean=600
)
```

### Gender and Age Effects

```python
# Male voice (lower pitch, larger vocal tract)
male_sound = soundgen.generate_bout(
    pitch_anchors=[(0, 120), (1, 100)],
    exact_formants='a',
    male_female=-0.5,        # Negative = male
    vocal_tract_length=18    # Longer vocal tract
)

# Female voice (higher pitch, smaller vocal tract)  
female_sound = soundgen.generate_bout(
    pitch_anchors=[(0, 120), (1, 100)],  # Same base pitch
    exact_formants='a',
    male_female=0.5,         # Positive = female  
    vocal_tract_length=13    # Shorter vocal tract
)
```

## Utility Functions

```python
# Generate smooth parameter contours
contour = soundgen.get_smooth_contour(
    anchors=[(0, 100), (0.5, 200), (1, 150)],
    length=1000  # Number of points
)

# Calculate harmonic rolloff
rolloff = soundgen.get_rolloff(
    pitch_per_gc=[200],      # Pitch values
    n_harmonics=50,          # Number of harmonics  
    rolloff_exp=-12          # Rolloff rate (dB/octave)
)

# Create spectral envelope for formant filtering
envelope = soundgen.get_spectral_envelope(
    nr=512,                  # Frequency bins
    nc=100,                  # Time frames
    exact_formants=soundgen.DEFAULT_FORMANTS['a'],
    sampling_rate=16000
)
```

## Parameter Reference

### Pitch Parameters
- `pitch_anchors`: List of (time, Hz) defining pitch contour
- `pitch_floor/ceiling`: Pitch bounds in Hz
- `vibrato_freq/dep`: Vibrato rate (Hz) and depth (semitones)
- `jitter_dep`: Random pitch variation (semitones)

### Formant Parameters  
- `exact_formants`: Vowel string ('a', 'aeiou') or FormantSpec list
- `formant_strength`: Formant amplitude scaling factor
- `vocal_tract_length`: Length in cm (affects formant frequencies)

### Voice Quality Parameters
- `temperature`: Overall stochasticity (0=deterministic, 0.1+=random)
- `creaky_breathy`: Voice quality (-1=creaky, +1=breathy)
- `noise_amount`: Proportion with noise/subharmonics (0-100%)
- `shimmer_dep`: Random amplitude variation (0-100%)

### Timing Parameters
- `syl_dur_mean`: Average syllable duration (ms)
- `n_syl`: Number of syllables
- `pause_dur_mean`: Average pause between syllables (ms)

### Advanced Parameters
- `g0`: Subharmonic target frequency (Hz)
- `sideband_width_hz`: Subharmonic bandwidth (Hz)
- `rolloff_exp`: Harmonic amplitude rolloff (dB/octave)
- `attack_len`: Fade-in/out duration (ms)

## Generated Sample Files

The repository includes example sounds in the `samples/` directory demonstrating different features:

- `vowel_a.wav` - Basic vowel [a]
- `vowel_i.wav` - Basic vowel [i]  
- `pitch_rising.wav` - Rising pitch contour
- `pitch_curved.wav` - Complex pitch curve
- `pitch_vibrato.wav` - Vibrato effect

Generate your own samples:
```bash
uv run python examples/simple_samples.py
```

## Algorithm Details

The synthesis follows these steps:

1. **Pitch Processing**: Generate smooth pitch contour from anchors, apply vibrato/jitter
2. **Glottal Cycles**: Convert pitch to glottal cycle timing
3. **Harmonic Generation**: Create sine waves for each harmonic with amplitude rolloff
4. **Noise Synthesis**: Generate filtered noise component if specified
5. **Formant Filtering**: Apply spectral envelope using STFT/ISTFT
6. **Post-Processing**: Add amplitude modulation, cross-fade between epochs

### Performance
- **Speed**: ~0.5 seconds to generate 0.5 seconds of audio
- **Quality**: 16 kHz sampling rate, normalized output
- **Memory**: Efficient for sounds up to several seconds

## Differences from R Version

- **Simplified subharmonics**: Basic vocal fry implementation vs. R's complex model
- **Python ecosystem**: Uses NumPy/SciPy instead of R's specialized audio packages
- **Parameter naming**: snake_case instead of camelCase
- **Performance**: Comparable speed with some optimizations

## Troubleshooting

**No sound generated**: Check that `syl_dur_mean > 10` and `pitch_anchors` has valid frequencies

**Distorted output**: Try reducing `temperature`, `jitter_dep`, or `noise_amount`

**Too quiet/loud**: Adjust `formant_strength` or check `exact_formants` amplitude values

**Import errors**: Install missing dependencies: `uv add numpy scipy soundfile sounddevice`

## Contributing

This implementation focuses on core synthesis functionality. Areas for enhancement:

- Advanced vocal fry modeling to match R implementation
- Additional formant presets for different speakers
- Real-time synthesis capabilities
- More sophisticated acoustic analysis tools

## References

- Original R soundgen: https://github.com/tatters/soundgen
- Algorithm documentation: https://cogsci.se/soundgen/algorithm.html  
- Anikin, A. (2019). Soundgen: An open-source tool for synthesizing nonlinguistic vocalizations. *Behavior Research Methods*, 51(2), 778-792.

## License

GPL-compatible license following the original R package.