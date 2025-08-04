#!/usr/bin/env python3
"""
Simple sound generation - one at a time with proper error handling.
"""

import numpy as np
import python_soundgen as soundgen
import sys

def generate_and_analyze(name, **params):
    """Generate a single sound and analyze it."""
    print(f"Generating {name}...", end=" ", flush=True)
    
    try:
        sound = soundgen.generate_bout(**params)
        
        # Analyze
        duration = len(sound) / params.get('sampling_rate', 16000)
        rms = np.sqrt(np.mean(sound**2))
        peak = np.max(np.abs(sound))
        
        print(f"✅ {duration:.3f}s, RMS={rms:.3f}, Peak={peak:.3f}")
        
        # Save if possible
        try:
            import soundfile as sf
            filename = f"sample_{name}.wav"
            sf.write(filename, sound, params.get('sampling_rate', 16000))
            print(f"   💾 Saved as {filename}")
        except ImportError:
            print(f"   📝 {len(sound)} samples generated")
        
        return sound
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    print("Generating individual sound samples...\n")
    
    # Basic parameters
    base_params = {
        'sampling_rate': 16000,
        'add_silence': 20,
        'temperature': 0
    }
    
    # Test 1: Simple vowel
    print("=== Basic Vowels ===")
    generate_and_analyze(
        "vowel_a", 
        pitch_anchors=[(0, 200), (1, 180)],
        exact_formants='a',
        syl_dur_mean=400,
        **base_params
    )
    
    generate_and_analyze(
        "vowel_i",
        pitch_anchors=[(0, 250), (1, 220)], 
        exact_formants='i',
        syl_dur_mean=400,
        **base_params
    )
    
    # Test 2: Pitch variations
    print("\n=== Pitch Contours ===")
    generate_and_analyze(
        "rising_pitch",
        pitch_anchors=[(0, 100), (1, 200)],
        exact_formants='a', 
        syl_dur_mean=500,
        **base_params
    )
    
    generate_and_analyze(
        "curved_pitch",
        pitch_anchors=[(0, 150), (0.3, 220), (0.7, 180), (1, 120)],
        exact_formants='o',
        syl_dur_mean=600,
        **base_params
    )
    
    # Test 3: Voice effects
    print("\n=== Voice Effects ===")
    generate_and_analyze(
        "vibrato",
        pitch_anchors=[(0, 150), (1, 150)],
        exact_formants='a',
        vibrato_freq=6,
        vibrato_dep=1.0,
        syl_dur_mean=500,
        **base_params
    )
    
    generate_and_analyze(
        "jittery",
        pitch_anchors=[(0, 140), (1, 130)],
        exact_formants='e',
        jitter_dep=0.8,
        noise_amount=20,
        syl_dur_mean=400,
        temperature=0.05,
        sampling_rate=16000,
        add_silence=20
    )
    
    # Test 4: Multi-syllable
    print("\n=== Multi-syllable ===")
    multi_params = base_params.copy()
    multi_params['temperature'] = 0.02  # Override base temperature
    generate_and_analyze(
        "two_syllables",
        n_syl=2,
        syl_dur_mean=250,
        pause_dur_mean=80,
        pitch_anchors=[(0, 170), (1, 130)],
        exact_formants='au',
        **multi_params
    )
    
    print(f"\n🎵 Sound generation complete!")
    print("Check the generated .wav files if soundfile is available.")

if __name__ == "__main__":
    main()