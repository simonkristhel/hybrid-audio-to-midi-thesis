from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from modules.spleeter_module import separate_monophonic_audio
from modules.crepe_module import extract_crepe_pitch
from modules.pyin_module import hybrid_pyin, pure_pyin
from modules.midi_module import pitch_df_to_midi


def run_system(input_audio_path, use_spleeter=True, ground_truth=None, mode="hybrid"):
    input_audio_path = Path(input_audio_path)

    if not input_audio_path.exists():
        raise FileNotFoundError(f"Input audio not found: {input_audio_path}")

    # =========================
    # OUTPUT FOLDERS
    # =========================
    output_root = Path("alpha_test")
    midi_out_dir = output_root / "output_midi"
    logs_dir = output_root / "logs"
    figures_dir = output_root / "figures"

    midi_out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # STEP 1: INPUT SELECTION
    # =========================
    if use_spleeter:
        print("▶ Step 1: Spleeter – monophonic separation")
        mono_audio_path = separate_monophonic_audio(
            input_audio_path=str(input_audio_path),
            output_dir=str(logs_dir)
        )
    else:
        print("▶ Step 1: Spleeter bypassed (clean monophonic input)")
        mono_audio_path = str(input_audio_path)

    # =========================
    # STEP 2: CREPE
    # =========================
    print("▶ Step 2: CREPE – pitch + confidence extraction")
    crepe_df = extract_crepe_pitch(mono_audio_path)

    # =========================
    # STEP 3: Pitch Tracking
    # =========================
    print(f"▶ Step 3: Pitch tracking mode = {mode}")

    if mode == "hybrid":
        hybrid_df = hybrid_pyin(
            audio_path=mono_audio_path,
            crepe_df=crepe_df
        )

    elif mode == "crepe":
        hybrid_df = crepe_df.rename(
            columns={"frequency": "hybrid_f0"}
        )[["time", "hybrid_f0"]]

    elif mode == "pyin":
        hybrid_df = pure_pyin(mono_audio_path)

    else:
        raise ValueError("Mode must be 'hybrid', 'crepe', or 'pyin'")

    # =========================
    # GENERATE FIGURES
    # =========================
    times = hybrid_df["time"].values
    freqs = hybrid_df["hybrid_f0"].values
    file_stem = input_audio_path.stem

    valid_freqs = freqs[~np.isnan(freqs)]

    mean_freq = np.mean(valid_freqs)
    std_freq = np.std(valid_freqs)

    print("Average detected frequency:", mean_freq)
    print("Min detected frequency:", np.min(valid_freqs))
    print("Max detected frequency:", np.max(valid_freqs))
    print("Standard deviation:", std_freq)

    # =========================
    # MAE + Accuracy Metrics
    # =========================
    if ground_truth is not None:
        mae = np.mean(np.abs(valid_freqs - ground_truth))
        relative_error = (mae / ground_truth) * 100
        cents_error = 1200 * np.log2(mean_freq / ground_truth)

        print("\n📈 Accuracy Metrics")
        print("Ground Truth:", ground_truth)
        print("Mean Absolute Error (Hz):", mae)
        print("Relative Error (%):", relative_error)
        print("Error (cents):", cents_error)

    # =========================
    # STEP 4: MIDI
    # =========================
    print("▶ Step 4: MIDI conversion")
    midi_output_path = midi_out_dir / f"{file_stem}.mid"
    pitch_df_to_midi(hybrid_df, str(midi_output_path))

    print(f"✅ Done! MIDI saved to: {midi_output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert audio to MIDI using pitch tracking")
    parser.add_argument("input_audio", help="Path to input audio file")
    parser.add_argument("--no-spleeter", action="store_true", help="Skip Spleeter monophonic separation")
    parser.add_argument("--ground-truth", type=float, help="Ground truth frequency (Hz)", default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "crepe", "pyin"],
        help="Select pitch tracking mode"
    )

    args = parser.parse_args()
    
    run_system(
        input_audio_path=args.input_audio,
        use_spleeter=not args.no_spleeter,
        ground_truth=args.ground_truth,
        mode=args.mode
    )
