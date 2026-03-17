import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from evaluate_with_spleeter import evaluate_with_optional_spleeter, str_to_bool


def load_ground_truth_map(csv_path: str | None) -> dict[str, str]:
    if not csv_path:
        return {}

    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {p}")

    df = pd.read_csv(p)
    required = {"file", "ground_truth"}
    if not required.issubset(df.columns):
        raise ValueError("Ground truth CSV must contain columns: file, ground_truth")

    return {str(r["file"]).strip(): str(r["ground_truth"]).strip() for _, r in df.iterrows()}


def run_batch(
    input_dir: str = "alpha_test/input_audio",
    mode: str = "hybrid",
    output_root: str = "alpha_test",
    run_with_spleeter: bool = True,
    run_without_spleeter: bool = True,
    ground_truth_csv: str | None = None,
) -> Path:
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    wav_files = sorted(input_path.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {input_path}")

    if not run_with_spleeter and not run_without_spleeter:
        raise ValueError("At least one of run_with_spleeter/run_without_spleeter must be true.")

    gt_map = load_ground_truth_map(ground_truth_csv)

    # Create a run-specific output root to avoid overwriting previous batch outputs.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_root = Path(output_root) / "batch_runs" / run_id
    run_output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    flags = []
    if run_without_spleeter:
        flags.append(False)
    if run_with_spleeter:
        flags.append(True)

    for wav_file in wav_files:
        for use_spleeter in flags:
            gt_value = gt_map.get(wav_file.name)
            try:
                result = evaluate_with_optional_spleeter(
                    input_audio_path=str(wav_file),
                    ground_truth=gt_value,
                    mode=mode,
                    use_spleeter=use_spleeter,
                    output_root=str(run_output_root),
                )
                metrics = result.get("metrics") or {}
                rows.append(
                    {
                        "file": wav_file.name,
                        "mode": mode,
                        "use_spleeter": use_spleeter,
                        "ground_truth_input": gt_value,
                        "ground_truth_hz": result.get("ground_truth_hz"),
                        "midi_output_path": result.get("midi_output_path"),
                        "figure_path": result.get("figure_path"),
                        "log_path": result.get("log_path"),
                        "median_detected_hz": metrics.get("median_detected_hz"),
                        "absolute_frequency_error_hz": metrics.get("absolute_frequency_error_hz"),
                        "cents_error": metrics.get("cents_error"),
                        "status": "ok",
                        "error": "",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "file": wav_file.name,
                        "mode": mode,
                        "use_spleeter": use_spleeter,
                        "ground_truth_input": gt_value,
                        "ground_truth_hz": None,
                        "midi_output_path": "",
                        "figure_path": "",
                        "log_path": "",
                        "median_detected_hz": None,
                        "absolute_frequency_error_hz": None,
                        "cents_error": None,
                        "status": "failed",
                        "error": str(exc),
                    }
                )

    summary_df = pd.DataFrame(rows)
    summary_csv = run_output_root / "batch_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n=== Batch Run Summary ===")
    print(f"Input directory: {input_path}")
    print(f"Mode: {mode}")
    print(f"Run output root: {run_output_root}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Total jobs: {len(rows)}")
    print(f"Successful: {(summary_df['status'] == 'ok').sum()}")
    print(f"Failed: {(summary_df['status'] == 'failed').sum()}")

    return summary_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch runner for input_audio with and without Spleeter using existing evaluation logic."
    )
    parser.add_argument("--input_dir", default="alpha_test/input_audio", help="Folder containing .wav files")
    parser.add_argument("--mode", default="hybrid", choices=["pyin", "crepe", "hybrid"], help="Pitch mode")
    parser.add_argument("--output_root", default="alpha_test", help="Base output root")
    parser.add_argument("--run_with_spleeter", type=str_to_bool, default=True, help="Run jobs with Spleeter")
    parser.add_argument("--run_without_spleeter", type=str_to_bool, default=True, help="Run jobs without Spleeter")
    parser.add_argument(
        "--ground_truth_csv",
        default=None,
        help="Optional CSV with columns: file,ground_truth (note or frequency)",
    )

    args = parser.parse_args()

    run_batch(
        input_dir=args.input_dir,
        mode=args.mode,
        output_root=args.output_root,
        run_with_spleeter=args.run_with_spleeter,
        run_without_spleeter=args.run_without_spleeter,
        ground_truth_csv=args.ground_truth_csv,
    )
