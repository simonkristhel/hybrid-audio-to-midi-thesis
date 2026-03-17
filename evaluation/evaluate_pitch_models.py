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
from modules.pyin_module import hybrid_pyin, pure_pyin


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
    required = {"file", "frequency_hz"}
    if not required.issubset(df.columns):
        raise ValueError("Ground truth CSV must contain columns: file, frequency_hz")

    mapping: dict[str, float] = {}
    for _, row in df.iterrows():
        mapping[str(row["file"]).strip()] = float(row["frequency_hz"])
    return mapping


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
        raw_pitch_acc = float(np.mean(np.abs(cents_err) <= 50.0))
    else:
        mae_cents = np.nan
        raw_pitch_acc = np.nan

    return {
        "Model": model,
        "MAE_cents": mae_cents,
        "Raw_Pitch_Accuracy": raw_pitch_acc,
        "Voicing_Recall": recall,
        "Voicing_False_Alarm": false_alarm,
    }


def evaluate_file(audio_path: Path, ref_hz: float, crepe_conf_threshold: float) -> list[dict[str, float | str]]:
    crepe_df = extract_crepe_pitch(str(audio_path))
    pyin_df = pure_pyin(str(audio_path))
    hybrid_df = hybrid_pyin(str(audio_path), crepe_df)

    ref_df = build_reference_grid(audio_path)

    per_model = []

    crepe_aligned = align_to_reference(ref_df, crepe_df.rename(columns={"frequency": "pred_f0", "confidence": "pred_conf"}), "pred_f0", "pred_conf")
    per_model.append(compute_metrics(crepe_aligned, "crepe", ref_hz, crepe_conf_threshold))

    pyin_aligned = align_to_reference(ref_df, pyin_df.rename(columns={"hybrid_f0": "pred_f0"}), "pred_f0")
    per_model.append(compute_metrics(pyin_aligned, "pure_pyin", ref_hz, crepe_conf_threshold))

    hybrid_aligned = align_to_reference(ref_df, hybrid_df.rename(columns={"hybrid_f0": "pred_f0"}), "pred_f0")
    per_model.append(compute_metrics(hybrid_aligned, "hybrid_pyin", ref_hz, crepe_conf_threshold))

    return per_model


def print_tables(results_df: pd.DataFrame) -> None:
    pd.options.display.float_format = "{:.4f}".format

    print("\nPer-file metrics")
    print(results_df.to_string(index=False))

    overall = (
        results_df.groupby("Model", as_index=False)[
            ["MAE_cents", "Raw_Pitch_Accuracy", "Voicing_Recall", "Voicing_False_Alarm"]
        ]
        .mean(numeric_only=True)
        .sort_values("Model")
    )

    print("\nOverall averages")
    print(overall.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CREPE, pure pYIN, and hybrid pYIN on alpha_test/input_audio")
    parser.add_argument("--input-dir", default="alpha_test/input_audio", help="Directory with input wav files")
    parser.add_argument("--ground-truth-csv", default=None, help="Optional CSV with columns: file,frequency_hz")
    parser.add_argument("--crepe-conf-threshold", type=float, default=0.5, help="Confidence threshold for CREPE voicing")
    parser.add_argument("--save-csv", default="evaluation/pitch_eval_results.csv", help="Path to save per-file metrics CSV")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    ground_truth_map = load_ground_truth_map(Path(args.ground_truth_csv)) if args.ground_truth_csv else {}

    wav_files = sorted(input_dir.glob("*.wav"))
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
