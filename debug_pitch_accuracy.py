"""
Diagnostic script to identify pitch detection accuracy issues.
Run this to see what each module is detecting for your A4 piano sample.
"""

import sys
import numpy as np
import librosa
from pathlib import Path
from modules.crepe_module import extract_crepe_pitch
from modules.pyin_module import hybrid_pyin
from modules.midi_module import hz_to_midi


def debug_pitch_detection(audio_path, expected_note="A4"):
    """
    Debug pitch detection at each stage.
    
    Parameters:
    - audio_path (str): Path to test audio
    - expected_note (str): Expected MIDI note name (e.g., "A4")
    """
    
    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        return
    
    # Convert note name to Hz and MIDI
    expected_hz = librosa.note_to_hz(expected_note)
    expected_midi = librosa.note_to_midi(expected_note)
    
    print(f"\n{'='*60}")
    print(f"PITCH ACCURACY DEBUG")
    print(f"{'='*60}")
    print(f"Expected Note: {expected_note}")
    print(f"Expected Frequency: {expected_hz:.2f} Hz")
    print(f"Expected MIDI: {expected_midi}")
    print(f"{'='*60}\n")
    
    # Step 1: CREPE Detection
    print("📊 STEP 1: CREPE Pitch Detection")
    print("-" * 60)
    crepe_df = extract_crepe_pitch(str(audio_path))
    
    # Filter high-confidence detections
    high_conf = crepe_df[crepe_df["confidence"] > 0.7]
    if len(high_conf) > 0:
        mean_freq = high_conf["frequency"].mean()
        median_freq = high_conf["frequency"].median()
        detected_midi = hz_to_midi(mean_freq)
        detected_note = librosa.midi_to_note(detected_midi)
        
        print(f"  High Confidence Frames: {len(high_conf)}/{len(crepe_df)}")
        print(f"  Mean Frequency: {mean_freq:.2f} Hz → MIDI {detected_midi} ({detected_note})")
        print(f"  Median Frequency: {median_freq:.2f} Hz")
        print(f"  Frequency Range: {high_conf['frequency'].min():.2f} - {high_conf['frequency'].max():.2f} Hz")
        print(f"  Confidence Range: {crepe_df['confidence'].min():.3f} - {crepe_df['confidence'].max():.3f}")
        
        freq_error = mean_freq - expected_hz
        print(f"  ⚠️  Frequency Error: {freq_error:+.2f} Hz ({(freq_error/expected_hz)*100:+.1f}%)")
    else:
        print("  ❌ No high-confidence frames detected!")
        print(f"  Max Confidence: {crepe_df['confidence'].max():.3f}")
    
    # Step 2: Hybrid pYIN + CREPE
    print("\n📊 STEP 2: Hybrid pYIN + CREPE")
    print("-" * 60)
    hybrid_df = hybrid_pyin(str(audio_path), crepe_df)
    
    valid_frames = hybrid_df[~np.isnan(hybrid_df["hybrid_f0"])]
    if len(valid_frames) > 0:
        mean_hybrid_freq = valid_frames["hybrid_f0"].mean()
        detected_midi = hz_to_midi(mean_hybrid_freq)
        detected_note = librosa.midi_to_note(detected_midi)
        
        print(f"  Valid Frames: {len(valid_frames)}/{len(hybrid_df)}")
        print(f"  Mean Hybrid Frequency: {mean_hybrid_freq:.2f} Hz → MIDI {detected_midi} ({detected_note})")
        print(f"  Frequency Range: {valid_frames['hybrid_f0'].min():.2f} - {valid_frames['hybrid_f0'].max():.2f} Hz")
        
        freq_error = mean_hybrid_freq - expected_hz
        print(f"  ⚠️  Frequency Error: {freq_error:+.2f} Hz ({(freq_error/expected_hz)*100:+.1f}%)")
    else:
        print("  ❌ No valid hybrid frames!")
    
    # Step 3: MIDI Conversion
    print("\n📊 STEP 3: MIDI Note Generation")
    print("-" * 60)
    if len(valid_frames) > 0:
        final_midi = hz_to_midi(mean_hybrid_freq)
        final_note = librosa.midi_to_note(final_midi)
        print(f"  Final MIDI Note: {final_midi} ({final_note})")
        
        midi_error = final_midi - expected_midi
        if midi_error == 0:
            print(f"  ✅ CORRECT! Note matches expected value.")
        else:
            print(f"  ❌ INCORRECT! Off by {midi_error:+d} semitones")
            if abs(midi_error) == 12:
                print(f"     → This is an OCTAVE ERROR (detected {abs(midi_error)} semitones away)")
    
    print(f"\n{'='*60}\n")
    
    # Recommendations
    print("🔧 RECOMMENDATIONS:")
    print("-" * 60)
    if len(high_conf) == 0:
        print("  1. ⚠️  CREPE confidence is very low. Try:")
        print("     - Using a different CREPE model capacity ('medium' or 'large')")
        print("     - Checking audio quality (noise, background sounds?)")
        print("     - Using --no-spleeter to skip source separation")
    
    if abs(midi_error) == 12 if 'midi_error' in locals() else False:
        print("  2. 🎵 OCTAVE ERROR detected! Enable Viterbi in CREPE:")
        print("     - Already done in the updated code (viterbi=True)")
    
    if abs(freq_error) > 5 if 'freq_error' in locals() else False:
        print("  3. 📉 Large frequency deviation. Check:")
        print("     - Sample rate consistency (should be 16kHz for CREPE)")
        print("     - If audio is being resampled correctly")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_pitch_accuracy.py <audio_file.wav> [expected_note]")
        print("Example: python debug_pitch_accuracy.py alpha_test/input_audio/piano_a4.wav A4")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    expected_note = sys.argv[2] if len(sys.argv) > 2 else "A4"
    
    debug_pitch_detection(audio_file, expected_note)
