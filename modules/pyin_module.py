import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import medfilt


def hybrid_pyin(
    audio_path: str,
    crepe_df: pd.DataFrame,
    fmin: float = librosa.note_to_hz("C1"),
    fmax: float = librosa.note_to_hz("C8"),
    confidence_threshold: float = 0.5
):
    """
    Hybrid pYIN pitch estimation guided by CREPE with adaptive weighted fusion.

    Returns:
        DataFrame with columns ['time', 'hybrid_f0']
    """

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # =========================
    # Load audio
    # =========================
    y, sr = librosa.load(audio_path, sr=44100, mono=True)

    # =========================
    # Run pYIN
    # =========================
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=8192,
        hop_length=256
    )

    times = librosa.times_like(f0, sr=sr, hop_length=256)

    pyin_df = pd.DataFrame({
        "time": times,
        "pyin_f0": f0,
        "voiced_prob": voiced_prob
    })

    # =========================
    # Align CREPE with pYIN
    # =========================
    merged = pd.merge_asof(
        pyin_df.sort_values("time"),
        crepe_df.sort_values("time"),
        on="time",
        direction="nearest"
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

        out = np.nan

        # Requirement 1:
        # If CREPE is unreliable, use pYIN directly.
        if (not crepe_valid) or (c < 0.2):
            out = pyin_freq if pyin_valid else np.nan

        # Use CREPE only when pYIN is unavailable.
        elif crepe_valid and (not pyin_valid):
            out = crepe_freq if c >= (confidence_threshold * 0.6) else np.nan

        # Both available -> disagreement handling + adaptive weighted fusion.
        elif crepe_valid and pyin_valid:
            disagreement_cents = abs(1200.0 * np.log2(
                max(crepe_freq, pyin_freq) / max(1e-9, min(crepe_freq, pyin_freq))
            ))

            # Requirement 2:
            # Large disagreement (>300 cents) -> choose higher-confidence detector.
            if disagreement_cents > 300.0:
                out = crepe_freq if c >= p else pyin_freq
            else:
                # Keep adaptive weighted fusion when detectors are reliable.
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

        # Requirement 3:
        # Never emit NaN or extreme outliers.
        if not np.isfinite(out) or out <= 0.0:
            if pyin_valid:
                out = pyin_freq
            elif crepe_valid:
                out = crepe_freq
            elif np.isfinite(prev_out) and prev_out > 0.0:
                out = prev_out
            else:
                out = fmin

        out = float(np.clip(out, fmin, fmax))

        hybrid_f0.append(out)
        prev_out = out

    merged["hybrid_f0"] = hybrid_f0

    # =========================
    # Light smoothing
    # =========================
    merged["hybrid_f0"] = medfilt(merged["hybrid_f0"], kernel_size=5)

    # Small interpolation only
    merged["hybrid_f0"] = pd.Series(merged["hybrid_f0"]).interpolate(limit=1)
    merged["hybrid_f0"] = merged["hybrid_f0"].ffill().bfill().fillna(fmin)
    merged["hybrid_f0"] = merged["hybrid_f0"].clip(lower=fmin, upper=fmax)

    return merged[["time", "hybrid_f0"]]


# ==========================================================
# PURE pYIN (no CREPE, no hybrid logic)
# ==========================================================
def pure_pyin(
    audio_path: str,
    fmin: float = librosa.note_to_hz("C1"),
    fmax: float = librosa.note_to_hz("C8")
):
    """
    Pure pYIN pitch estimation.
    Returns DataFrame with columns ['time', 'hybrid_f0']
    """

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
        hop_length=256
    )

    times = librosa.times_like(f0, sr=sr, hop_length=256)

    df = pd.DataFrame({
        "time": times,
        "hybrid_f0": f0
    })

    return df
