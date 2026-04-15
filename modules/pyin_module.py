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

    hybrid_f0 = []
    prev_out = np.nan

    # =========================
    # Weighted Fusion
    # =========================
    for _, row in merged.iterrows():
        crepe_freq = row.get("frequency", np.nan)
        pyin_freq = row.get("pyin_f0", np.nan)
        crepe_conf = float(row.get("confidence", 0.0))
        pyin_prob = float(row.get("voiced_prob", 0.0))

        c = float(np.clip(crepe_conf, 0.0, 1.0))
        p = float(np.clip(pyin_prob, 0.0, 1.0))

        crepe_valid = np.isfinite(crepe_freq) and (crepe_freq > 0.0)
        pyin_valid = np.isfinite(pyin_freq) and (pyin_freq > 0.0)
        crepe_reliable = crepe_valid and (c >= confidence_threshold)

        out = np.nan

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

                # Preserve pYIN stability by damping sudden jumps unless CREPE is very confident.
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
