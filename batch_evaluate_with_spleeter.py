import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from evaluate_with_spleeter import analyze_audio, str_to_bool


def load_ground_truth_map(csv_path: str | Path | None) -> dict[str, str]:
    """Load optional per-file ground truth values from CSV."""
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


def build_summary_row(
    wav_file: Path,
    mode: str,
    use_spleeter: bool,
    ground_truth_input: str | None,
    result: dict | None = None,
    error: Exception | None = None,
) -> dict:
    """Normalize batch results into one CSV row."""
    if error is not None:
        return {
            "file": wav_file.name,
            "mode": mode,
            "use_spleeter": use_spleeter,
            "ground_truth_input": ground_truth_input,
            "ground_truth_hz": None,
            "midi_output_path": "",
            "figure_path": "",
            "log_path": "",
            "median_detected_hz": None,
            "absolute_frequency_error_hz": None,
            "cents_error": None,
            "status": "failed",
            "error": str(error),
        }

    metrics = (result or {}).get("metrics") or {}
    return {
        "file": wav_file.name,
        "mode": mode,
        "use_spleeter": use_spleeter,
        "ground_truth_input": ground_truth_input,
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


def run_batch_jobs(
    input_dir: str | Path,
    mode: str = "hybrid",
    use_spleeter_flags: list[bool] | tuple[bool, ...] = (True,),
    ground_truth_csv: str | Path | None = None,
    output_root: str | Path = "outputs",
) -> Path:
    """Shared batch implementation used by the new and legacy interfaces."""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    wav_files = sorted(input_path.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {input_path}")

    flags = list(dict.fromkeys(use_spleeter_flags))
    if not flags:
        raise ValueError("At least one use_spleeter setting must be provided.")

    gt_map = load_ground_truth_map(ground_truth_csv)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_root = Path(output_root) / "batch_runs" / run_id
    run_output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for wav_file in wav_files:
        for use_spleeter in flags:
            gt_value = gt_map.get(wav_file.name)
            try:
                result = analyze_audio(
                    input_path=str(wav_file),
                    mode=mode,
                    use_spleeter=use_spleeter,
                    ground_truth=gt_value,
                    output_root=str(run_output_root),
                    verbose=True,
                )
                rows.append(build_summary_row(wav_file, mode, use_spleeter, gt_value, result=result))
            except Exception as exc:
                rows.append(build_summary_row(wav_file, mode, use_spleeter, gt_value, error=exc))

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


def batch_evaluate(
    input_dir: str = "alpha_test/input_audio",
    mode: str = "hybrid",
    use_spleeter: bool = True,
    ground_truth_csv: str | None = None,
    output_root: str = "outputs",
) -> Path:
    """Canonical batch interface for one Spleeter setting."""
    return run_batch_jobs(
        input_dir=input_dir,
        mode=mode,
        use_spleeter_flags=[use_spleeter],
        ground_truth_csv=ground_truth_csv,
        output_root=output_root,
    )


def run_batch(
    input_dir: str = "alpha_test/input_audio",
    mode: str = "hybrid",
    output_root: str = "alpha_test",
    run_with_spleeter: bool = True,
    run_without_spleeter: bool = True,
    ground_truth_csv: str | None = None,
) -> Path:
    """Backward-compatible wrapper that can run both Spleeter settings in one batch."""
    flags: list[bool] = []
    if run_without_spleeter:
        flags.append(False)
    if run_with_spleeter:
        flags.append(True)

    return run_batch_jobs(
        input_dir=input_dir,
        mode=mode,
        use_spleeter_flags=flags,
        ground_truth_csv=ground_truth_csv,
        output_root=output_root,
    )


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
