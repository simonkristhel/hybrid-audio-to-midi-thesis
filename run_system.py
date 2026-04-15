"""Legacy single-file runner kept as a thin wrapper around analyze_audio()."""

from pathlib import Path

from evaluate_with_spleeter import analyze_audio


def run_system(
    input_audio_path,
    use_spleeter=True,
    ground_truth=None,
    mode="hybrid",
    output_root="alpha_test",
):
    """
    Backward-compatible single-file entry point.

    The analysis logic now lives in evaluate_with_spleeter.analyze_audio().
    """
    return analyze_audio(
        input_path=Path(input_audio_path),
        mode=mode,
        use_spleeter=use_spleeter,
        ground_truth=ground_truth,
        output_root=output_root,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert audio to MIDI using the shared analysis pipeline")
    parser.add_argument("input_audio", help="Path to input audio file")
    parser.add_argument("--no-spleeter", action="store_true", help="Skip Spleeter monophonic separation")
    parser.add_argument("--ground-truth", type=float, help="Ground truth frequency (Hz)", default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "crepe", "pyin"],
        help="Select pitch tracking mode",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="alpha_test",
        help="Base output folder for MIDI, logs, and figures",
    )

    args = parser.parse_args()

    run_system(
        input_audio_path=args.input_audio,
        use_spleeter=not args.no_spleeter,
        ground_truth=args.ground_truth,
        mode=args.mode,
        output_root=args.output_root,
    )
