import crepe
import librosa
import numpy as np
import pandas as pd
from pathlib import Path


def extract_crepe_pitch(
    audio_path: str,
    step_size: int = 10,
    model_capacity: str = "full"
):
    """
    Extract pitch and confidence using CREPE.

    Parameters:
    - audio_path (str): Path to monophonic WAV file
    - step_size (int): Time resolution in ms (default: 10 ms)
    - model_capacity (str): CREPE model size (tiny, small, medium, large, full)

    Returns:
    - pandas.DataFrame with columns:
        ['time', 'frequency', 'confidence']
    """

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load audio (CREPE expects 16kHz mono)
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    # CREPE expects float32
    audio = audio.astype(np.float32)

    # Run CREPE without Viterbi (octave correction will be done in pYIN module)
    time, frequency, confidence, _ = crepe.predict(
        audio,
        sr,
        step_size=step_size,
        model_capacity=model_capacity,
        viterbi=True   
    )

    # Package results into DataFrame
    pitch_df = pd.DataFrame({
        "time": time,
        "frequency": frequency,
        "confidence": confidence
    })

    return pitch_df
