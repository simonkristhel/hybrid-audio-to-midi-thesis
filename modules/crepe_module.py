import crepe
import librosa
import numpy as np
import pandas as pd
from pathlib import Path


def extract_crepe_pitch(
    audio_path: str,
    step_size: int = 10,
    model_capacity: str = "full",
    confidence_threshold: float | None = 0.5,
):
    """
    Extract pitch and confidence using CREPE.

    Parameters:
    - audio_path (str): Path to monophonic WAV file
    - step_size (int): Time resolution in ms (default: 10 ms)
    - model_capacity (str): CREPE model size (tiny, small, medium, large, full)
    - confidence_threshold (float | None): Frames below this confidence are
      masked to NaN. Pass None to inspect raw CREPE output.

    Returns:
    - pandas.DataFrame with columns:
        ['time', 'frequency', 'confidence']
      Low-confidence frames are masked to NaN by default so downstream
      statistics and MIDI generation only see voiced CREPE estimates.
    """

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load audio (CREPE expects 16kHz mono)
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    # CREPE expects float32
    audio = audio.astype(np.float32)

    # Run CREPE and keep the confidence so downstream code can decide
    # which frames should count as voiced.
    time, frequency, confidence, _ = crepe.predict(
        audio,
        sr,
        step_size=step_size,
        model_capacity=model_capacity,
        viterbi=True,
    )

    if confidence_threshold is not None:
        frequency = np.where(confidence >= confidence_threshold, frequency, np.nan)

    # Package results into DataFrame
    pitch_df = pd.DataFrame({
        "time": time,
        "frequency": frequency,
        "confidence": confidence
    })

    return pitch_df
