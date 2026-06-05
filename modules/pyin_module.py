from pathlib import Path

import librosa
import numpy as np
import pandas as pd


def extract_pyin_pitch_details(
    audio_path: str,
    fmin: float = librosa.note_to_hz("C1"),
    fmax: float = librosa.note_to_hz("C8"),
) -> pd.DataFrame:
    """Run pYIN and return pitch plus voiced probability for hybrid fusion."""
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    y, sr = librosa.load(audio_path, sr=44100, mono=True)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=8192,
        hop_length=256,
    )

    times = librosa.times_like(f0, sr=sr, hop_length=256)

    return pd.DataFrame({
        "time": times,
        "pyin_f0": f0,
        "voiced_prob": voiced_prob,
    })


def fuse_pyin_crepe(
    pyin_df: pd.DataFrame,
    crepe_df: pd.DataFrame,
    fmin: float = librosa.note_to_hz("C1"),
    fmax: float = librosa.note_to_hz("C8"),
    confidence_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Fuse pYIN and CREPE pitch tracks into the standardized hybrid output.

    Returns:
        DataFrame with columns ['time', 'hybrid_f0']
    """
    # =========================
    # Align CREPE with pYIN
    # =========================
    merged = pd.merge_asof(
        pyin_df.sort_values("time"),
        crepe_df.sort_values("time"),
        on="time",
        direction="nearest",
    )

    crepe_freq = merged["frequency"].to_numpy(dtype=float)
    pyin_freq = merged["pyin_f0"].to_numpy(dtype=float)
    crepe_conf = np.clip(merged.get("confidence", pd.Series(0.0, index=merged.index)).to_numpy(dtype=float), 0.0, 1.0)
    pyin_conf = np.clip(merged.get("voiced_prob", pd.Series(0.0, index=merged.index)).to_numpy(dtype=float), 0.0, 1.0)

    crepe_valid = np.isfinite(crepe_freq) & (crepe_freq > 0.0)
    pyin_valid = np.isfinite(pyin_freq) & (pyin_freq > 0.0)

    def adaptive_threshold(conf: np.ndarray, valid: np.ndarray) -> float:
        valid_conf = conf[valid & np.isfinite(conf)]
        if valid_conf.size == 0:
            return confidence_threshold
        # A segment-local lower-mid percentile keeps weak but consistent tracks
        # available without letting detector tails dominate.
        return float(np.clip(np.nanpercentile(valid_conf, 40), 0.25, 0.85))

    crepe_threshold = adaptive_threshold(crepe_conf, crepe_valid)
    pyin_threshold = adaptive_threshold(pyin_conf, pyin_valid)

    segment_candidates = np.concatenate([crepe_freq[crepe_valid], pyin_freq[pyin_valid]])
    if segment_candidates.size:
        segment_median = float(np.exp(np.nanmedian(np.log(np.maximum(segment_candidates, 1e-9)))))
    else:
        segment_median = np.nan

    if np.isfinite(segment_median) and segment_median > 0.0:
        crepe_outlier = crepe_valid & (np.abs(1200.0 * np.log2(crepe_freq / segment_median)) > 2400.0)
        pyin_outlier = pyin_valid & (np.abs(1200.0 * np.log2(pyin_freq / segment_median)) > 2400.0)
        crepe_valid = crepe_valid & ~crepe_outlier
        pyin_valid = pyin_valid & ~pyin_outlier

    hybrid_f0 = np.full(len(merged), np.nan, dtype=float)

    for i in range(len(merged)):
        c_valid = bool(crepe_valid[i])
        p_valid = bool(pyin_valid[i])
        c_reliable = c_valid and (crepe_conf[i] >= crepe_threshold)
        p_reliable = p_valid and (pyin_conf[i] >= pyin_threshold)

        if c_reliable and p_reliable:
            disagreement_cents = abs(1200.0 * np.log2(
                max(crepe_freq[i], pyin_freq[i]) / max(1e-9, min(crepe_freq[i], pyin_freq[i]))
            ))
            conf_gap = abs(crepe_conf[i] - pyin_conf[i])

            if disagreement_cents <= 100.0 or conf_gap <= 0.20:
                weight_sum = crepe_conf[i] + pyin_conf[i]
                if weight_sum > 1e-9:
                    hybrid_f0[i] = (
                        (pyin_conf[i] * pyin_freq[i]) +
                        (crepe_conf[i] * crepe_freq[i])
                    ) / weight_sum
                else:
                    hybrid_f0[i] = np.nanmean([pyin_freq[i], crepe_freq[i]])
            else:
                hybrid_f0[i] = pyin_freq[i] if pyin_conf[i] >= crepe_conf[i] else crepe_freq[i]
        elif p_reliable:
            hybrid_f0[i] = pyin_freq[i]
        elif c_reliable:
            hybrid_f0[i] = crepe_freq[i]
        elif p_valid:
            hybrid_f0[i] = pyin_freq[i]
        elif c_valid:
            hybrid_f0[i] = crepe_freq[i]

    merged["hybrid_f0"] = np.clip(hybrid_f0, fmin, fmax)

    series = pd.Series(merged["hybrid_f0"], dtype=float)
    series = series.where(np.isfinite(series) & (series > 0.0))
    series = series.interpolate(limit=2, limit_area="inside")
    valid_mask = series.notna()
    series = series.rolling(window=7, center=True, min_periods=1).median()
    merged["hybrid_f0"] = series.where(valid_mask).clip(lower=fmin, upper=fmax)

    return merged[["time", "hybrid_f0"]]


def hybrid_pyin(
    audio_path: str,
    crepe_df: pd.DataFrame,
    fmin: float = librosa.note_to_hz("C1"),
    fmax: float = librosa.note_to_hz("C8"),
    confidence_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Hybrid pYIN pitch estimation guided by CREPE with adaptive weighted fusion.
    Unvoiced regions remain NaN so summary statistics and MIDI generation
    do not treat low-confidence detector tails as real notes.
    """
    pyin_df = extract_pyin_pitch_details(audio_path, fmin=fmin, fmax=fmax)
    return fuse_pyin_crepe(
        pyin_df=pyin_df,
        crepe_df=crepe_df,
        fmin=fmin,
        fmax=fmax,
        confidence_threshold=confidence_threshold,
    )


# ==========================================================
# PURE pYIN (no CREPE, no hybrid logic)
# ==========================================================
def pure_pyin(
    audio_path: str,
    fmin: float = librosa.note_to_hz("C1"),
    fmax: float = librosa.note_to_hz("C8"),
) -> pd.DataFrame:
    """
    Pure pYIN pitch estimation.
    Returns DataFrame with columns ['time', 'hybrid_f0']
    """
    pyin_df = extract_pyin_pitch_details(audio_path, fmin=fmin, fmax=fmax)
    return pyin_df.rename(columns={"pyin_f0": "hybrid_f0"})[["time", "hybrid_f0"]]
