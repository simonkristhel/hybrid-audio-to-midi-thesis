"""Canonical backend entry points for single-file and batch analysis."""

from evaluate_with_spleeter import analyze_audio
from batch_evaluate_with_spleeter import batch_evaluate

__all__ = ["analyze_audio", "batch_evaluate"]
