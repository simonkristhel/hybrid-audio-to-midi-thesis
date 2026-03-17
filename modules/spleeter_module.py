import subprocess
import os
from pathlib import Path


def separate_monophonic_audio(
    input_audio_path: str,
    output_dir: str,
    stems: int = 2
):
    """
    Separates monophonic audio from a mixed audio file using Spleeter.

    Parameters:
    - input_audio_path (str): Path to input WAV file
    - output_dir (str): Directory where separated audio will be saved
    - stems (int): Number of stems (default = 2: vocals + accompaniment)

    Returns:
    - Path to separated monophonic audio file
    """

    input_audio_path = Path(input_audio_path)
    output_dir = Path(output_dir)

    if not input_audio_path.exists():
        raise FileNotFoundError(f"Input audio not found: {input_audio_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model = f"spleeter:{stems}stems"

    command = [
        "spleeter",
        "separate",
        "-p", model,
        "-o", str(output_dir),
        str(input_audio_path)
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Spleeter separation failed") from e

    # Expected output structure:
    # output_dir / input_filename / vocals.wav
    base_name = input_audio_path.stem
    separated_path = output_dir / base_name / "vocals.wav"

    if not separated_path.exists():
        raise FileNotFoundError("Separated monophonic track not found.")

    return separated_path
