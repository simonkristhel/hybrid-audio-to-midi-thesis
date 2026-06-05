import argparse
import re
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.crepe_module import extract_crepe_pitch
from modules.pyin_module import extract_pyin_pitch_details, fuse_pyin_crepe


def infer_reference_hz(filename: str) -> float | None:
    stem = Path(filename).stem

    hz_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*hz", stem, flags=re.IGNORECASE)
    if hz_match:
        return float(hz_match.group(1))

    note_match = re.search(r"(?:^|[^A-Za-z0-9])([A-Ga-g](?:#|b)?-?[0-8])(?:[^A-Za-z0-9]|$)", stem)
    if note_match:
        note = note_match.group(1)
        note = note[0].upper() + note[1:]
        try:
            return float(librosa.note_to_hz(note))
        except Exception:
            return None

    return None


def load_ground_truth_map(csv_path: Path | None) -> dict[str, float]:
    if csv_path is None:
        return {}

    if not csv_path.exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "file" not in df.columns:
        raise ValueError("Ground truth CSV must contain a file column")
    if "frequency_hz" not in df.columns and "ground_truth" not in df.columns:
        raise ValueError("Ground truth CSV must contain frequency_hz or ground_truth")

    mapping: dict[str, float] = {}
    for _, row in df.iterrows():
        if "frequency_hz" in df.columns and pd.notna(row.get("frequency_hz")):
            ref_hz = float(row["frequency_hz"])
        else:
            label = str(row["ground_truth"]).strip()
            ref_hz = float(label) if re.match(r"^[0-9]+(?:\.[0-9]+)?$", label) else float(librosa.note_to_hz(label))
        mapping[str(row["file"]).strip()] = ref_hz
    return mapping


def legacy_fuse_pyin_crepe(
    pyin_df: pd.DataFrame,
    crepe_df: pd.DataFrame,
    fmin: float = float(librosa.note_to_hz("C1")),
    fmax: float = float(librosa.note_to_hz("C8")),
    confidence_threshold: float = 0.5,
) -> pd.DataFrame:
    """Previous hybrid fusion retained for before/after evaluation."""
    merged = pd.merge_asof(
        pyin_df.sort_values("time"),
        crepe_df.sort_values("time"),
        on="time",
        direction="nearest",
    )

    hybrid_f0 = []
    prev_out = np.nan

    for _, row in merged.iterrows():
        crepe_freq = row.get("frequency", np.nan)
        pyin_freq = row.get("pyin_f0", np.nan)
        c = float(np.clip(float(row.get("confidence", 0.0)), 0.0, 1.0))
        p = float(np.clip(float(row.get("voiced_prob", 0.0)), 0.0, 1.0))

        crepe_valid = np.isfinite(crepe_freq) and (crepe_freq > 0.0)
        pyin_valid = np.isfinite(pyin_freq) and (pyin_freq > 0.0)
        crepe_reliable = crepe_valid and (c >= confidence_threshold)

        if pyin_valid and crepe_reliable:
            disagreement_cents = abs(1200.0 * np.log2(
                max(crepe_freq, pyin_freq) / max(1e-9, min(crepe_freq, pyin_freq))
            ))

            if disagreement_cents > 300.0:
                out = crepe_freq if c >= p else pyin_freq
            else:
                crepe_rel = 0.10 + 0.90 * c
                pyin_rel = 0.55 + 0.45 * p

                if disagreement_cents > 50.0:
                    if c > (p + 0.15):
                        crepe_rel *= 1.4
                    elif p > (c + 0.15):
                        pyin_rel *= 1.3

                crepe_weight = crepe_rel / max(1e-9, (crepe_rel + pyin_rel))
                crepe_weight = float(np.clip(crepe_weight, 0.10, 0.70))
                pyin_weight = 1.0 - crepe_weight

                out = float(np.exp(
                    (crepe_weight * np.log(max(crepe_freq, 1e-9))) +
                    (pyin_weight * np.log(max(pyin_freq, 1e-9)))
                ))

                if np.isfinite(prev_out) and prev_out > 0.0:
                    jump_cents = abs(1200.0 * np.log2(max(out, 1e-9) / max(prev_out, 1e-9)))
                    if jump_cents > 140.0 and c < 0.80:
                        out = (0.75 * pyin_freq) + (0.25 * out)
        elif pyin_valid:
            out = pyin_freq
        elif crepe_reliable:
            out = crepe_freq
        else:
            out = np.nan

        hybrid_f0.append(out)
        if np.isfinite(out) and out > 0.0:
            prev_out = float(np.clip(out, fmin, fmax))
            hybrid_f0[-1] = prev_out

    merged["hybrid_f0"] = hybrid_f0
    series = pd.Series(merged["hybrid_f0"], dtype=float)
    series = series.where(np.isfinite(series) & (series > 0.0))
    series = series.interpolate(limit=1, limit_area="inside")
    valid_mask = series.notna()
    series = series.rolling(window=5, center=True, min_periods=1).median()
    merged["hybrid_f0"] = series.where(valid_mask).clip(lower=fmin, upper=fmax)

    return merged[["time", "hybrid_f0"]]


def build_reference_grid(audio_path: Path, hop_length: int = 256, sr: int = 44100) -> pd.DataFrame:
    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    if np.all(rms <= 0):
        voiced = np.zeros_like(rms, dtype=bool)
    else:
        threshold = 0.02 * float(np.max(rms))
        voiced = rms > threshold

    return pd.DataFrame({"time": times, "gt_voiced": voiced})


def align_to_reference(ref_df: pd.DataFrame, pred_df: pd.DataFrame, pred_col: str, conf_col: str | None = None) -> pd.DataFrame:
    cols = ["time", pred_col]
    if conf_col and conf_col in pred_df.columns:
        cols.append(conf_col)

    merged = pd.merge_asof(
        ref_df.sort_values("time"),
        pred_df[cols].sort_values("time"),
        on="time",
        direction="nearest",
    )
    return merged


def compute_metrics(aligned_df: pd.DataFrame, model: str, ref_hz: float, crepe_conf_threshold: float) -> dict[str, float | str]:
    pred_f0 = aligned_df["pred_f0"].to_numpy(dtype=float)
    gt_voiced = aligned_df["gt_voiced"].to_numpy(dtype=bool)

    if model == "crepe":
        pred_conf = aligned_df.get("pred_conf", pd.Series(np.zeros(len(aligned_df), dtype=float))).to_numpy(dtype=float)
        pred_voiced = np.isfinite(pred_f0) & (pred_f0 > 0) & (pred_conf >= crepe_conf_threshold)
    else:
        pred_voiced = np.isfinite(pred_f0) & (pred_f0 > 0)

    tp = int(np.sum(gt_voiced & pred_voiced))
    fn = int(np.sum(gt_voiced & ~pred_voiced))
    fp = int(np.sum(~gt_voiced & pred_voiced))
    tn = int(np.sum(~gt_voiced & ~pred_voiced))

    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    false_alarm = fp / (fp + tn) if (fp + tn) > 0 else np.nan

    voiced_overlap = gt_voiced & pred_voiced
    if np.any(voiced_overlap):
        pred_voiced_f0 = pred_f0[voiced_overlap]
        cents_err = 1200.0 * np.log2(np.maximum(pred_voiced_f0, 1e-9) / max(ref_hz, 1e-9))
        mae_cents = float(np.mean(np.abs(cents_err)))
        median_mae_cents = float(np.median(np.abs(cents_err)))
        raw_pitch_acc = float(np.mean(np.abs(cents_err) <= 50.0))
    else:
        mae_cents = np.nan
        median_mae_cents = np.nan
        raw_pitch_acc = np.nan

    return {
        "Model": model,
        "MAE_cents": mae_cents,
        "Median_MAE_cents": median_mae_cents,
        "Raw_Pitch_Accuracy": raw_pitch_acc,
        "Voicing_Recall": recall,
        "Voicing_False_Alarm": false_alarm,
    }


def evaluate_file(audio_path: Path, ref_hz: float, crepe_conf_threshold: float) -> list[dict[str, float | str]]:
    crepe_hybrid_df = extract_crepe_pitch(str(audio_path), confidence_threshold=None)
    crepe_df = crepe_hybrid_df.copy()
    crepe_df["frequency"] = np.where(
        crepe_df["confidence"].to_numpy(dtype=float) >= crepe_conf_threshold,
        crepe_df["frequency"].to_numpy(dtype=float),
        np.nan,
    )
    pyin_details_df = extract_pyin_pitch_details(str(audio_path))
    pyin_df = pyin_details_df.rename(columns={"pyin_f0": "hybrid_f0"})[["time", "hybrid_f0"]]
    original_hybrid_df = legacy_fuse_pyin_crepe(pyin_details_df, crepe_df)
    new_hybrid_df = fuse_pyin_crepe(pyin_details_df, crepe_hybrid_df)

    ref_df = build_reference_grid(audio_path)

    per_model = []

    crepe_aligned = align_to_reference(ref_df, crepe_df.rename(columns={"frequency": "pred_f0", "confidence": "pred_conf"}), "pred_f0", "pred_conf")
    per_model.append(compute_metrics(crepe_aligned, "crepe", ref_hz, crepe_conf_threshold))

    pyin_aligned = align_to_reference(ref_df, pyin_df.rename(columns={"hybrid_f0": "pred_f0"}), "pred_f0")
    per_model.append(compute_metrics(pyin_aligned, "pure_pyin", ref_hz, crepe_conf_threshold))

    original_hybrid_aligned = align_to_reference(ref_df, original_hybrid_df.rename(columns={"hybrid_f0": "pred_f0"}), "pred_f0")
    per_model.append(compute_metrics(original_hybrid_aligned, "original_hybrid", ref_hz, crepe_conf_threshold))

    new_hybrid_aligned = align_to_reference(ref_df, new_hybrid_df.rename(columns={"hybrid_f0": "pred_f0"}), "pred_f0")
    per_model.append(compute_metrics(new_hybrid_aligned, "new_hybrid", ref_hz, crepe_conf_threshold))

    return per_model


def print_tables(results_df: pd.DataFrame) -> None:
    pd.options.display.float_format = "{:.4f}".format

    print("\nPer-file metrics")
    print(results_df.to_string(index=False))

    overall = (
        results_df.groupby("Model", as_index=False)[
            ["MAE_cents", "Median_MAE_cents", "Raw_Pitch_Accuracy", "Voicing_Recall", "Voicing_False_Alarm"]
        ]
        .mean(numeric_only=True)
        .sort_values("Model")
    )

    print("\nOverall averages")
    print(overall.to_string(index=False))

    comparison = results_df.pivot_table(
        index="File",
        columns="Model",
        values=["Median_MAE_cents", "Raw_Pitch_Accuracy"],
        aggfunc="first",
    ).sort_index()

    print("\nRequested comparison: Median MAE in cents and Raw Pitch Accuracy")
    print(comparison.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CREPE, pure pYIN, and hybrid pYIN on alpha_test/input_audio")
    parser.add_argument("--input-dir", default="alpha_test/input_audio", help="Directory with input wav files")
    parser.add_argument("--ground-truth-csv", default=None, help="Optional CSV with columns: file,frequency_hz")
    parser.add_argument("--include-files", nargs="*", default=None, help="Optional exact wav filenames to evaluate")
    parser.add_argument("--crepe-conf-threshold", type=float, default=0.5, help="Confidence threshold for CREPE voicing")
    parser.add_argument("--save-csv", default="evaluation/pitch_eval_results.csv", help="Path to save per-file metrics CSV")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    ground_truth_map = load_ground_truth_map(Path(args.ground_truth_csv)) if args.ground_truth_csv else {}

    wav_files = sorted(input_dir.glob("*.wav"))
    if args.include_files:
        include_names = set(args.include_files)
        wav_files = [path for path in wav_files if path.name in include_names]
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {input_dir}")

    rows: list[dict[str, float | str]] = []
    skipped: list[str] = []

    for wav_path in wav_files:
        ref_hz = ground_truth_map.get(wav_path.name)
        if ref_hz is None:
            ref_hz = infer_reference_hz(wav_path.name)

        if ref_hz is None:
            skipped.append(wav_path.name)
            continue

        metrics = evaluate_file(wav_path, ref_hz=ref_hz, crepe_conf_threshold=args.crepe_conf_threshold)
        for metric_row in metrics:
            metric_row["File"] = wav_path.name
            metric_row["Reference_Hz"] = ref_hz
            rows.append(metric_row)

    if not rows:
        raise RuntimeError("No files could be evaluated. Provide parseable filenames or a ground truth CSV.")

    results_df = pd.DataFrame(rows)[
        [
            "File",
            "Model",
            "Reference_Hz",
            "MAE_cents",
            "Median_MAE_cents",
            "Raw_Pitch_Accuracy",
            "Voicing_Recall",
            "Voicing_False_Alarm",
        ]
    ]

    print_tables(results_df)

    save_path = Path(args.save_csv)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(save_path, index=False)
    print(f"\nSaved per-file results to: {save_path}")

    if skipped:
        print("\nSkipped files (no reference pitch inferred):")
        for name in skipped:
            print(f"- {name}")


if __name__ == "__main__":
    main()
