"""
Example usage of the Python soundgen library.
Demonstrates various synthesis capabilities and parameter settings.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

try:
    import soundfile as sf
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Note: soundfile and/or sounddevice not available. Audio I/O disabled.")

from . import generate_bout

def basic_vowel_example(play: bool = False, save_path: Optional[str] = None):
    """Generate a basic vowel sound."""
    print("Generating basic vowel [a]...")
    
    sound = generate_bout(
        pitch_anchors=[(0, 200), (1, 150)],
        exact_formants='a',
        syl_dur_mean=1000,
        sampling_rate=16000,
        temperature=0
    )
    
    if play and AUDIO_AVAILABLE:
        sd.play(sound, 16000)
        sd.wait()
    
    if save_path and AUDIO_AVAILABLE:
        sf.write(save_path, sound, 16000)
        print(f"Saved to {save_path}")
    
    return sound

def pitch_contour_example(play: bool = False, save_path: Optional[str] = None):
    """Generate sound with complex pitch contour."""
    print("Generating sound with complex pitch contour...")
    
    sound = generate_bout(
        pitch_anchors=[(0, 100), (0.3, 200), (0.7, 250), (1, 80)],
        exact_formants='aeiou',
        syl_dur_mean=1500,
        sampling_rate=16000,
        vibrato_freq=5,
        vibrato_dep=0.5,
        temperature=0.05
    )
    
    if play and AUDIO_AVAILABLE:
        sd.play(sound, 16000)
        sd.wait()
    
    if save_path and AUDIO_AVAILABLE:
        sf.write(save_path, sound, 16000)
        print(f"Saved to {save_path}")
    
    return sound

def noisy_vocalization_example(play: bool = False, save_path: Optional[str] = None):
    """Generate noisy vocalization with jitter and shimmer."""
    print("Generating noisy vocalization...")
    
    sound = generate_bout(
        pitch_anchors=[(0, 150), (0.5, 200), (1, 100)],
        exact_formants='o',
        syl_dur_mean=800,
        noise_amount=50,
        jitter_dep=2.0,
        shimmer_dep=30,
        rolloff_exp=-15,
        temperature=0.1,
        sampling_rate=16000
    )
    
    if play and AUDIO_AVAILABLE:
        sd.play(sound, 16000)
        sd.wait()
    
    if save_path and AUDIO_AVAILABLE:
        sf.write(save_path, sound, 16000)
        print(f"Saved to {save_path}")
    
    return sound

def multisyllable_example(play: bool = False, save_path: Optional[str] = None):
    """Generate multi-syllable vocalization."""
    print("Generating multi-syllable vocalization...")
    
    sound = generate_bout(
        n_syl=3,
        syl_dur_mean=400,
        pause_dur_mean=150,
        pitch_anchors=[(0, 180), (0.3, 220), (0.7, 200), (1, 160)],
        pitch_anchors_global=[(0, -3), (0.5, 5), (1, -2)],  # semitones
        exact_formants='aiu',
        sampling_rate=16000,
        temperature=0.08
    )
    
    if play and AUDIO_AVAILABLE:
        sd.play(sound, 16000)
        sd.wait()
    
    if save_path and AUDIO_AVAILABLE:
        sf.write(save_path, sound, 16000)
        print(f"Saved to {save_path}")
    
    return sound

def creaky_voice_example(play: bool = False, save_path: Optional[str] = None):
    """Generate creaky voice."""
    print("Generating creaky voice...")
    
    sound = generate_bout(
        pitch_anchors=[(0, 80), (0.5, 120), (1, 70)],
        exact_formants='a',
        syl_dur_mean=1200,
        creaky_breathy=-0.8,  # Creaky
        noise_amount=70,
        g0=75,
        sideband_width_hz=130,
        rolloff_exp=-8,
        sampling_rate=16000,
        temperature=0.1
    )
    
    if play and AUDIO_AVAILABLE:
        sd.play(sound, 16000)
        sd.wait()
    
    if save_path and AUDIO_AVAILABLE:
        sf.write(save_path, sound, 16000)
        print(f"Saved to {save_path}")
    
    return sound

def breathy_voice_example(play: bool = False, save_path: Optional[str] = None):
    """Generate breathy voice with aspiration noise."""
    print("Generating breathy voice...")
    
    sound = generate_bout(
        pitch_anchors=[(0, 150), (0.5, 180), (1, 140)],
        exact_formants='e',
        syl_dur_mean=1000,
        creaky_breathy=0.7,  # Breathy
        breathing_anchors=[(0, -20), (500, -10), (1000, -25)],
        rolloff_exp=-18,
        sampling_rate=16000,
        temperature=0.05
    )
    
    if play and AUDIO_AVAILABLE:
        sd.play(sound, 16000)
        sd.wait()
    
    if save_path and AUDIO_AVAILABLE:
        sf.write(save_path, sound, 16000)
        print(f"Saved to {save_path}")
    
    return sound

def male_female_comparison(play: bool = False, save_path_base: Optional[str] = None):
    """Compare male and female voice characteristics."""
    print("Generating male and female voice comparison...")
    
    # Male voice
    male_sound = generate_bout(
        pitch_anchors=[(0, 120), (0.5, 150), (1, 100)],
        exact_formants='a',
        male_female=-0.5,  # Male adjustment
        syl_dur_mean=800,
        sampling_rate=16000
    )
    
    # Female voice  
    female_sound = generate_bout(
        pitch_anchors=[(0, 120), (0.5, 150), (1, 100)],
        exact_formants='a', 
        male_female=0.5,  # Female adjustment
        syl_dur_mean=800,
        sampling_rate=16000
    )
    
    if play and AUDIO_AVAILABLE:
        print("Playing male voice...")
        sd.play(male_sound, 16000)
        sd.wait()
        print("Playing female voice...")
        sd.play(female_sound, 16000)
        sd.wait()
    
    if save_path_base and AUDIO_AVAILABLE:
        sf.write(f"{save_path_base}_male.wav", male_sound, 16000)
        sf.write(f"{save_path_base}_female.wav", female_sound, 16000)
        print(f"Saved male voice to {save_path_base}_male.wav")
        print(f"Saved female voice to {save_path_base}_female.wav")
    
    return male_sound, female_sound

def animal_like_vocalization(play: bool = False, save_path: Optional[str] = None):
    """Generate animal-like vocalization with subharmonics."""
    print("Generating animal-like vocalization...")
    
    sound = generate_bout(
        pitch_anchors=[(0, 200), (0.3, 350), (0.6, 320), (1, 180)],
        exact_formants=[
            {'f1': {'time': 0, 'freq': 600, 'amp': 35, 'width': 100}},
            {'f2': {'time': 0, 'freq': 1200, 'amp': 30, 'width': 150}},
            {'f3': {'time': 0, 'freq': 2800, 'amp': 20, 'width': 200}}
        ],
        syl_dur_mean=600,
        noise_amount=80,
        g0=150,
        sideband_width_hz=200,
        jitter_dep=1.5,
        rolloff_exp=-10,
        temperature=0.15,
        vocal_tract_length=8,  # Shorter vocal tract
        sampling_rate=16000
    )
    
    if play and AUDIO_AVAILABLE:
        sd.play(sound, 16000)
        sd.wait()
    
    if save_path and AUDIO_AVAILABLE:
        sf.write(save_path, sound, 16000)
        print(f"Saved to {save_path}")
    
    return sound

def plot_spectrogram(sound: np.ndarray, sampling_rate: float = 16000, 
                    title: str = "Spectrogram"):
    """Plot spectrogram of generated sound."""
    try:
        from scipy.signal import spectrogram
        
        f, t, Sxx = spectrogram(sound, sampling_rate, nperseg=1024, noverlap=512)
        
        plt.figure(figsize=(12, 6))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title(title)
        plt.colorbar(label='Power [dB]')
        plt.ylim(0, 4000)  # Focus on speech range
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("matplotlib not available for spectrogram plotting")

def run_all_examples(play: bool = False, save_dir: Optional[str] = None):
    """Run all examples and optionally save audio files."""
    print("Running all soundgen examples...\n")
    
    examples = [
        ("basic_vowel", basic_vowel_example),
        ("pitch_contour", pitch_contour_example), 
        ("noisy_vocalization", noisy_vocalization_example),
        ("multisyllable", multisyllable_example),
        ("creaky_voice", creaky_voice_example),
        ("breathy_voice", breathy_voice_example),
        ("animal_like", animal_like_vocalization)
    ]
    
    sounds = {}
    
    for name, func in examples:
        save_path = f"{save_dir}/{name}.wav" if save_dir else None
        sound = func(play=play, save_path=save_path)
        sounds[name] = sound
        print()
    
    # Male/female comparison
    save_base = f"{save_dir}/voice_comparison" if save_dir else None
    male_sound, female_sound = male_female_comparison(play=play, save_path_base=save_base)
    sounds['male'] = male_sound
    sounds['female'] = female_sound
    
    print("All examples completed!")
    
    # Plot spectrograms if matplotlib available
    try:
        import matplotlib.pyplot as plt
        print("\nGenerating spectrograms...")
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, (name, sound) in enumerate(list(sounds.items())[:8]):
            ax = axes[i]
            
            from scipy.signal import spectrogram
            f, t, Sxx = spectrogram(sound, 16000, nperseg=512, noverlap=256)
            
            im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), 
                              shading='gouraud', cmap='viridis')
            ax.set_ylabel('Frequency [Hz]')
            ax.set_xlabel('Time [sec]')
            ax.set_title(name.replace('_', ' ').title())
            ax.set_ylim(0, 3000)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("matplotlib not available for spectrograms")
    
    return sounds

if __name__ == "__main__":
    # Run examples when script is executed directly
    import os
    
    # Create output directory
    output_dir = "soundgen_examples"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run all examples
    sounds = run_all_examples(play=False, save_dir=output_dir)
    
    print(f"\nExample sounds saved to: {output_dir}/")
    print("To play sounds, set play=True in the function calls.")