import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import librosa
import matplotlib
import numpy as np
import pandas as pd

from modules.crepe_module import extract_crepe_pitch
from modules.midi_module import pitch_df_to_midi
from modules.pyin_module import extract_pyin_pitch_details, fuse_pyin_crepe, hybrid_pyin, pure_pyin
from modules.spleeter_module import separate_monophonic_audio

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Manual ground-truth lookup for files that should not depend on filename parsing.
MANUAL_GROUND_TRUTH = {
    "m3_long_straight_a_C4.wav": "C4",
    "m3_long_straight_a_C5.wav": "C5",
    "m3_long_straight_a_F5.wav": "F5",
    "f8_long_straight_a_C4.wav": "C4",
    "f8_long_straight_a_C5.wav": "C5",
    "f8_long_straight_a_F5.wav": "F5",
}


def parse_ground_truth(ground_truth: str | float | None, input_audio_path: Path) -> tuple[float | None, str]:
    """
    Resolve ground truth from:
    1) explicit CLI value
    2) manual filename dictionary
    """
    source = "none"
    value = ground_truth

    if value is None:
        value = MANUAL_GROUND_TRUTH.get(input_audio_path.name)
        if value is not None:
            source = "manual_dict"
    else:
        source = "cli"

    if value is None:
        return None, source

    # Accept numeric frequency strings.
    try:
        hz = float(value)
        if hz <= 0:
            raise ValueError("Ground truth frequency must be > 0")
        return hz, source
    except ValueError:
        pass

    # Otherwise treat as note name.
    try:
        return float(librosa.note_to_hz(value)), source
    except Exception as exc:
        raise ValueError(
            f"Invalid ground truth '{value}'. Use note name (e.g. C4) or frequency (e.g. 261.63)."
        ) from exc


def prepare_output_dirs(output_root: str | Path) -> dict[str, Path]:
    """Create and return the standard output directories for one analysis run."""
    output_root_path = Path(output_root)
    logs_dir = output_root_path / "logs"
    figures_dir = output_root_path / "figures"
    midi_dir = output_root_path / "output_midi"

    logs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    midi_dir.mkdir(parents=True, exist_ok=True)

    return {
        "root": output_root_path,
        "logs": logs_dir,
        "figures": figures_dir,
        "midi": midi_dir,
    }


def record_timing(timings: dict[str, float], label: str, start_time: float) -> None:
    """Store elapsed seconds for one analysis step."""
    timings[label] = perf_counter() - start_time


def timed_call(label: str, func, *args, **kwargs):
    """Run one callable and return its result with elapsed seconds."""
    start_time = perf_counter()
    result = func(*args, **kwargs)
    return label, result, perf_counter() - start_time


def build_parallel_hybrid_pitch_df(mono_audio_path: str, timings: dict[str, float] | None = None) -> pd.DataFrame:
    """Run independent pYIN and CREPE extraction concurrently, then fuse their outputs."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        crepe_future = executor.submit(
            timed_call,
            "hybrid_crepe_extraction",
            extract_crepe_pitch,
            mono_audio_path,
            confidence_threshold=None,
        )
        pyin_future = executor.submit(timed_call, "hybrid_pyin_extraction", extract_pyin_pitch_details, mono_audio_path)

        crepe_label, crepe_df, crepe_elapsed = crepe_future.result()
        pyin_label, pyin_df, pyin_elapsed = pyin_future.result()

    if timings is not None:
        timings[crepe_label] = crepe_elapsed
        timings[pyin_label] = pyin_elapsed

    fusion_start = perf_counter()
    pitch_df = fuse_pyin_crepe(pyin_df=pyin_df, crepe_df=crepe_df)
    if timings is not None:
        timings["hybrid_fusion"] = perf_counter() - fusion_start
        timings["hybrid_parallel_used"] = 1.0

    return pitch_df


def build_pitch_df_for_mode(
    mode: str,
    mono_audio_path: str,
    timings: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Return DataFrame with standardized columns ['time', 'hybrid_f0'].
    """
    if mode == "crepe":
        crepe_df = extract_crepe_pitch(mono_audio_path)
        return crepe_df.rename(columns={"frequency": "hybrid_f0"})[["time", "hybrid_f0"]]

    if mode == "pyin":
        return pure_pyin(mono_audio_path)

    if mode == "hybrid":
        try:
            return build_parallel_hybrid_pitch_df(mono_audio_path, timings=timings)
        except Exception:
            if timings is not None:
                timings["hybrid_sequential_fallback_used"] = 1.0
            crepe_df = extract_crepe_pitch(mono_audio_path, confidence_threshold=None)
            return hybrid_pyin(mono_audio_path, crepe_df)

    raise ValueError("Mode must be one of: pyin, crepe, hybrid")


def compute_pitch_stats(pitch_df: pd.DataFrame) -> dict[str, float]:
    """Return simple descriptive statistics for the detected pitch track."""
    valid_freqs = pitch_df["hybrid_f0"].to_numpy(dtype=float)
    valid_freqs = valid_freqs[np.isfinite(valid_freqs) & (valid_freqs > 0)]

    if len(valid_freqs) == 0:
        return {
            "median_detected_hz": np.nan,
            "mean_detected_hz": np.nan,
            "min_detected_hz": np.nan,
            "max_detected_hz": np.nan,
            "std_detected_hz": np.nan,
        }

    return {
        "median_detected_hz": float(np.median(valid_freqs)),
        "mean_detected_hz": float(np.mean(valid_freqs)),
        "min_detected_hz": float(np.min(valid_freqs)),
        "max_detected_hz": float(np.max(valid_freqs)),
        "std_detected_hz": float(np.std(valid_freqs)),
    }


def compute_metrics(hybrid_df: pd.DataFrame, ground_truth_hz: float | None) -> dict[str, float] | None:
    if ground_truth_hz is None:
        return None

    valid_freqs = hybrid_df["hybrid_f0"].to_numpy(dtype=float)
    valid_freqs = valid_freqs[np.isfinite(valid_freqs) & (valid_freqs > 0)]
    if len(valid_freqs) == 0:
        return {
            "median_detected_hz": np.nan,
            "absolute_frequency_error_hz": np.nan,
            "cents_error": np.nan,
        }

    median_freq = float(np.median(valid_freqs))
    abs_freq_error = float(abs(median_freq - ground_truth_hz))
    cents_error = float(1200.0 * np.log2(max(median_freq, 1e-9) / max(ground_truth_hz, 1e-9)))

    return {
        "median_detected_hz": median_freq,
        "absolute_frequency_error_hz": abs_freq_error,
        "cents_error": cents_error,
    }


def save_pitch_figure(
    pitch_df: pd.DataFrame,
    input_audio_path: Path,
    mode: str,
    used_spleeter: bool,
    figures_dir: Path,
) -> Path:
    tag = "spleeter_on" if used_spleeter else "spleeter_off"
    fig_path = figures_dir / f"{input_audio_path.stem}_{mode}_{tag}.png"

    plt.figure(figsize=(12, 4))
    plt.plot(pitch_df["time"], pitch_df["hybrid_f0"], linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Pitch Track: {input_audio_path.name} | mode={mode} | {tag}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path


def print_analysis_summary(result: dict) -> None:
    """Print a concise console summary for CLI/demo use."""
    print("\n=== Evaluation Summary ===")
    print(f"File processed: {result['input_path']}")
    print(f"Spleeter used: {result['use_spleeter']}")
    print(f"Mode: {result['mode']}")
    print(f"Ground truth source: {result['ground_truth_source']}")
    print(f"Ground truth (Hz): {result['ground_truth_hz']}")
    print(f"MIDI output: {result['midi_output_path']}")
    print(f"Figure output: {result['figure_path']}")
    print(f"Log output: {result['log_path']}")

    pitch_stats = result["pitch_stats"]
    print("Pitch stats:")
    print(f"  median_detected_hz: {pitch_stats['median_detected_hz']:.4f}")
    print(f"  mean_detected_hz: {pitch_stats['mean_detected_hz']:.4f}")
    print(f"  min_detected_hz: {pitch_stats['min_detected_hz']:.4f}")
    print(f"  max_detected_hz: {pitch_stats['max_detected_hz']:.4f}")
    print(f"  std_detected_hz: {pitch_stats['std_detected_hz']:.4f}")

    metrics = result["metrics"]
    if metrics is None:
        print("Metrics: skipped (no ground truth provided/found)")
    else:
        print("Metrics:")
        print(f"  median_detected_hz: {metrics['median_detected_hz']:.4f}")
        print(f"  absolute_frequency_error_hz: {metrics['absolute_frequency_error_hz']:.4f}")
        print(f"  cents_error: {metrics['cents_error']:.4f}")

    timings = result.get("timings") or {}
    if timings:
        print("Timings (seconds):")
        for label, elapsed_seconds in timings.items():
            print(f"  {label}: {elapsed_seconds:.4f}")


def analyze_audio(
    input_path: str | Path,
    mode: str = "hybrid",
    use_spleeter: bool = True,
    ground_truth: str | float | None = None,
    output_root: str | Path = "outputs",
    verbose: bool = True,
) -> dict:
    """
    Analyze one audio file and return structured outputs for UI or CLI use.

    Output files are written under output_root/{logs,figures,output_midi}.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio not found: {input_path}")

    total_start = perf_counter()
    timings: dict[str, float] = {}

    step_start = perf_counter()
    output_dirs = prepare_output_dirs(output_root)
    record_timing(timings, "prepare_output_dirs", step_start)

    logs_dir = output_dirs["logs"]
    figures_dir = output_dirs["figures"]
    midi_dir = output_dirs["midi"]

    # Step 1: Optional source separation
    step_start = perf_counter()
    if use_spleeter:
        mono_audio_path = separate_monophonic_audio(
            input_audio_path=str(input_path),
            output_dir=str(logs_dir),
        )
    else:
        mono_audio_path = str(input_path)
    record_timing(timings, "spleeter_or_input_selection", step_start)

    # Step 2: Pitch extraction by selected mode
    step_start = perf_counter()
    pitch_df = build_pitch_df_for_mode(mode=mode, mono_audio_path=mono_audio_path, timings=timings)
    record_timing(timings, "pitch_extraction", step_start)

    step_start = perf_counter()
    pitch_stats = compute_pitch_stats(pitch_df)
    record_timing(timings, "pitch_statistics", step_start)

    # Step 3: Metrics (optional if ground truth is available)
    step_start = perf_counter()
    gt_hz, gt_source = parse_ground_truth(ground_truth, input_path)
    metrics = compute_metrics(pitch_df, gt_hz)
    record_timing(timings, "metrics", step_start)

    # Step 4: Save MIDI
    step_start = perf_counter()
    tag = "spleeter_on" if use_spleeter else "spleeter_off"
    midi_output_path = midi_dir / f"{input_path.stem}_{mode}_{tag}.mid"
    pitch_df_to_midi(pitch_df, str(midi_output_path))
    record_timing(timings, "save_midi", step_start)

    # Step 5: Save figure
    step_start = perf_counter()
    figure_path = save_pitch_figure(
        pitch_df=pitch_df,
        input_audio_path=input_path,
        mode=mode,
        used_spleeter=use_spleeter,
        figures_dir=figures_dir,
    )
    record_timing(timings, "save_figure", step_start)
    record_timing(timings, "total_before_log", total_start)

    # Step 6: Save a compact metrics log
    step_start = perf_counter()
    log_path = logs_dir / f"{input_path.stem}_{mode}_{tag}_summary.txt"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"input_file={input_path}\n")
        f.write(f"use_spleeter={use_spleeter}\n")
        f.write(f"mode={mode}\n")
        f.write(f"mono_audio_used={mono_audio_path}\n")
        f.write(f"ground_truth_source={gt_source}\n")
        f.write(f"ground_truth_hz={gt_hz}\n")
        f.write(f"midi_output={midi_output_path}\n")
        f.write(f"figure_output={figure_path}\n")
        f.write(f"median_detected_hz={pitch_stats['median_detected_hz']}\n")
        f.write(f"mean_detected_hz={pitch_stats['mean_detected_hz']}\n")
        f.write(f"min_detected_hz={pitch_stats['min_detected_hz']}\n")
        f.write(f"max_detected_hz={pitch_stats['max_detected_hz']}\n")
        f.write(f"std_detected_hz={pitch_stats['std_detected_hz']}\n")
        if metrics is None:
            f.write("metrics=skipped_no_ground_truth\n")
        else:
            f.write(f"median_detected_hz={metrics['median_detected_hz']}\n")
            f.write(f"absolute_frequency_error_hz={metrics['absolute_frequency_error_hz']}\n")
            f.write(f"cents_error={metrics['cents_error']}\n")
        f.write("\n[timings_seconds]\n")
        for timing_label, elapsed_seconds in timings.items():
            f.write(f"{timing_label}={elapsed_seconds:.4f}\n")
    record_timing(timings, "save_log", step_start)
    record_timing(timings, "total", total_start)

    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"save_log={timings['save_log']:.4f}\n")
        f.write(f"total={timings['total']:.4f}\n")

    result = {
        "input_path": str(input_path),
        "mono_audio_path": str(mono_audio_path),
        "mode": mode,
        "use_spleeter": use_spleeter,
        "output_root": str(output_dirs["root"]),
        "ground_truth_input": ground_truth,
        "ground_truth_source": gt_source,
        "ground_truth_hz": gt_hz,
        "midi_output_path": str(midi_output_path),
        "figure_path": str(figure_path),
        "log_path": str(log_path),
        "pitch_stats": pitch_stats,
        "metrics": metrics,
        "timings": timings,
    }

    if verbose:
        print_analysis_summary(result)

    return result


def evaluate_with_optional_spleeter(
    input_audio_path: str,
    ground_truth: str | float | None = None,
    mode: str = "hybrid",
    use_spleeter: bool = True,
    output_root: str = "alpha_test",
):
    """Backward-compatible wrapper around analyze_audio()."""
    return analyze_audio(
        input_path=input_audio_path,
        mode=mode,
        use_spleeter=use_spleeter,
        ground_truth=ground_truth,
        output_root=output_root,
    )


def str_to_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "yes", "y"}:
        return True
    if v in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected boolean value: true/false")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate one audio file with optional Spleeter, pitch extraction, metrics, and MIDI export."
    )
    parser.add_argument("--input", required=True, help="Path to input WAV file")
    parser.add_argument("--ground_truth", default=None, help="Optional note (e.g. C4) or frequency (e.g. 261.63)")
    parser.add_argument("--mode", default="hybrid", choices=["pyin", "crepe", "hybrid"], help="Pitch tracking mode")
    parser.add_argument(
        "--use_spleeter",
        type=str_to_bool,
        default=True,
        help="Whether to run Spleeter first (true/false)",
    )
    parser.add_argument("--output_root", default="alpha_test", help="Output root directory")

    args = parser.parse_args()

    analyze_audio(
        input_path=args.input,
        mode=args.mode,
        use_spleeter=args.use_spleeter,
        ground_truth=args.ground_truth,
        output_root=args.output_root,
    )
