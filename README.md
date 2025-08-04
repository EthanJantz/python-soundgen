# Python Soundgen

Python implementation of parametric voice synthesis based on the R soundgen package. Generates vocal sounds using a source-filter model with controllable pitch, formants, and voice quality parameters.

## Overview

This package implements parametric synthesis of vocalizations through three components:

1. **Harmonic source** - Glottal oscillation with pitch control and voice effects
2. **Noise component** - Aspiration and breathing sounds  
3. **Formant filtering** - Vocal tract resonance simulation

The synthesis uses a source-filter model where harmonic and noise sources are filtered through formant resonances to create realistic vocal sounds.

## Installation

```bash
git clone <repository-url>
cd python-soundgen

uv sync
# or
pip install numpy scipy matplotlib soundfile sounddevice
```

## Quick Start

```python
import python_soundgen as soundgen

# Generate a vowel sound
sound = soundgen.generate_bout(
    pitch_anchors=[(0, 200), (1, 150)],
    exact_formants='a',
    syl_dur_mean=500
)

# Save and play
import soundfile as sf
sf.write('vowel.wav', sound, 16000)
```

## API Reference

### generate_bout()

Main synthesis function for generating vocalizations.

**Pitch Control**
- `pitch_anchors`: List of (time, Hz) anchor points defining pitch contour
- `pitch_floor`, `pitch_ceiling`: float, pitch bounds in Hz (default: 50, 3500)  
- `vibrato_freq`, `vibrato_dep`: float, vibrato rate (Hz) and depth (semitones)
- `jitter_dep`: float, random pitch variation in semitones

**Formant Specification**
- `exact_formants`: str or list, vowel ('a','e','i','o','u') or FormantSpec objects
- `vocal_tract_length`: float, vocal tract length in cm (default: 15.5)
- `formant_strength`: float, formant amplitude scaling (default: 1.0)

**Voice Quality**
- `temperature`: float, stochastic variation amount (0=deterministic, 0.1+=random)
- `creaky_breathy`: float, voice quality (-1=creaky, +1=breathy, default: 0)
- `noise_amount`: float, proportion with noise/subharmonics (0-100%, default: 0)
- `shimmer_dep`: float, amplitude variation (0-100%, default: 0)

**Timing**
- `syl_dur_mean`: float, syllable duration in ms (default: 500)
- `n_syl`: int, number of syllables (default: 1)
- `pause_dur_mean`: float, pause duration between syllables in ms (default: 200)

**Technical Parameters**
- `sampling_rate`: float, audio sampling rate in Hz (default: 16000)
- `window_length_points`: int, FFT window size (default: 2048)
- `overlap`: float, window overlap percentage (default: 75)
- `add_silence`: float, silence padding in ms (default: 100)

**Returns**: numpy.ndarray, generated audio waveform

## Usage Examples

### Basic Vowel Synthesis

```python
# Generate vowel [a]
sound = soundgen.generate_bout(
    pitch_anchors=[(0, 200), (1, 180)],
    exact_formants='a',
    syl_dur_mean=400
)
```

### Pitch Contours

```python
# Complex pitch curve
sound = soundgen.generate_bout(
    pitch_anchors=[(0, 150), (0.3, 250), (0.7, 200), (1, 120)],
    exact_formants='o',
    syl_dur_mean=800
)
```

### Voice Effects

```python
# Vibrato
sound = soundgen.generate_bout(
    pitch_anchors=[(0, 150), (1, 150)],
    exact_formants='a',
    vibrato_freq=6,
    vibrato_dep=1.0,
    syl_dur_mean=600
)

# Jittery voice with noise
sound = soundgen.generate_bout(
    pitch_anchors=[(0, 140), (1, 130)],
    exact_formants='e',
    jitter_dep=0.8,
    noise_amount=20,
    temperature=0.05
)
```

### Multi-syllable Sounds

```python
# Two syllables
sound = soundgen.generate_bout(
    n_syl=2,
    syl_dur_mean=300,
    pause_dur_mean=100,
    pitch_anchors=[(0, 180), (1, 140)],
    exact_formants='au'
)
```

### Custom Formants

```python
from python_soundgen import FormantSpec

formants = [
    FormantSpec(time=0, freq=800, amp=35, width=100),   # F1
    FormantSpec(time=0, freq=1200, amp=30, width=120),  # F2
    FormantSpec(time=0, freq=2800, amp=25, width=200)   # F3
]

sound = soundgen.generate_bout(
    pitch_anchors=[(0, 180), (1, 160)],
    exact_formants=formants,
    syl_dur_mean=500
)
```

## Formant Reference

### Available Vowels
- `'a'`: [a] - F1=860Hz, F2=1280Hz, F3=2900Hz
- `'e'`: [e] - F1=530Hz, F2=1840Hz, F3=2480Hz  
- `'i'`: [i] - F1=270Hz, F2=2300Hz, F3=3010Hz
- `'o'`: [o] - F1=400Hz, F2=800Hz, F3=2830Hz
- `'u'`: [u] - F1=320Hz, F2=800Hz, F3=2560Hz

### FormantSpec Class
```python
FormantSpec(
    time=0,        # Time point (0-1) or list for time-varying
    freq=800,      # Frequency in Hz or list
    amp=30,        # Amplitude in dB or list  
    width=120      # Bandwidth in Hz or list
)
```

## Utility Functions

```python
# Smooth contour interpolation
contour = soundgen.get_smooth_contour(
    anchors=[(0, 100), (1, 200)],
    length=1000
)

# Harmonic amplitude rolloff
rolloff = soundgen.get_rolloff(
    pitch_per_gc=[200],
    n_harmonics=50,
    rolloff_exp=-12
)

# Spectral envelope for filtering
envelope = soundgen.get_spectral_envelope(
    nr=512, nc=100,
    exact_formants=soundgen.DEFAULT_FORMANTS['a'],
    sampling_rate=16000
)
```

## Technical Details

### Synthesis Algorithm
1. Generate pitch contour from anchor points
2. Apply jitter and vibrato to pitch
3. Create harmonic series with amplitude rolloff
4. Generate noise component if specified  
5. Apply formant filtering via STFT
6. Combine components and apply envelope

### Performance
- Generation speed: ~1x real-time for typical parameters
- Memory usage: ~10MB for 1-second synthesis
- Sampling rates: 8kHz to 48kHz supported

### File I/O
- Supports WAV output via soundfile
- Audio playback via sounddevice
- NumPy array interface for custom processing

## Sample Generation

```bash
# Generate example sounds
python examples/simple_samples.py
```

Sample files are saved to the current directory demonstrating basic vowels, pitch contours, and voice effects.

## Troubleshooting

- **No output**: Check `syl_dur_mean > 10` and valid `pitch_anchors`
- **Distortion**: Reduce `temperature`, `jitter_dep`, or `noise_amount`  
- **Volume issues**: Adjust `formant_strength` parameter
- **Dependencies**: Install with `pip install numpy scipy soundfile sounddevice`

## References

- Original R package: https://cran.r-project.org/web/packages/soundgen/index.html
- Anikin, A. (2019). Soundgen: An open-source tool for synthesizing nonlinguistic vocalizations. *Behavior Research Methods*, 51(2), 778-792.

## License

GPL-compatible license following the original R package.
