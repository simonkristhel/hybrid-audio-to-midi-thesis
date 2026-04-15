from datetime import datetime
from pathlib import Path
import re

import streamlit as st

from evaluate_with_spleeter import analyze_audio

PROJECT_ROOT = Path(__file__).resolve().parent
UI_RUNS_DIR = PROJECT_ROOT / "outputs" / "ui_runs"
ASSETS_DIR = PROJECT_ROOT / "assets"
MODES = ("hybrid", "pyin", "crepe")


def sanitize_filename(filename: str) -> str:
    """Return a filesystem-safe filename stem for uploaded files."""
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._")
    return safe_name or "uploaded_audio.wav"


def create_run_paths(original_name: str) -> tuple[Path, Path]:
    """Create a per-run folder and input path for the uploaded WAV file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = sanitize_filename(original_name)
    run_root = UI_RUNS_DIR / f"{timestamp}_{Path(safe_name).stem}"
    input_dir = run_root / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / safe_name
    return run_root, input_path


def format_metric(value) -> str:
    """Render metrics consistently for the UI."""
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric != numeric:
        return "N/A"
    return f"{numeric:.4f}"


def save_uploaded_file(uploaded_file) -> tuple[Path, Path]:
    """Persist the uploaded file inside an app-managed run folder."""
    run_root, input_path = create_run_paths(uploaded_file.name)
    input_path.write_bytes(uploaded_file.getbuffer())
    return run_root, input_path


def render_local_image(
    image_path: Path,
    caption: str,
    missing_message: str | None = None,
    width: int | None = None,
) -> None:
    """Render a local image file with a visible fallback if it cannot be read."""
    if not image_path.exists():
        st.warning(missing_message or f"Image not found: {image_path.name}")
        return

    try:
        image_bytes = image_path.read_bytes()
    except OSError as exc:
        st.warning(f"Could not read image '{image_path.name}': {exc}")
        return

    output_format = "JPEG" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "PNG"
    st.image(image_bytes, caption=caption, width=width, output_format=output_format)


def render_results(result: dict) -> None:
    """Render the output summary for one completed analysis."""
    metrics = result.get("metrics") or {}
    pitch_stats = result.get("pitch_stats") or {}
    median_hz = metrics.get("median_detected_hz", pitch_stats.get("median_detected_hz"))

    st.subheader("Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Median detected Hz", format_metric(median_hz))
    col2.metric("Absolute frequency error (Hz)", format_metric(metrics.get("absolute_frequency_error_hz")))
    col3.metric("Cents error", format_metric(metrics.get("cents_error")))

    if result.get("ground_truth_hz") is None:
        st.info("Ground truth was not provided, so error metrics are shown as N/A.")

    timings = result.get("timings") or {}
    if timings:
        st.markdown("**Analysis Timings**")
        timing_rows = [
            {"Step": label.replace("_", " ").title(), "Seconds": f"{elapsed_seconds:.4f}"}
            for label, elapsed_seconds in timings.items()
        ]
        st.table(timing_rows)

    st.markdown("**Output Files**")
    output_lines = [
        f"Input file: {result.get('input_path')}",
        f"Mono audio used: {result.get('mono_audio_path')}",
        f"MIDI output: {result.get('midi_output_path')}",
        f"Graph image: {result.get('figure_path')}",
        f"Log file: {result.get('log_path')}",
    ]
    st.code("\n".join(output_lines), language="text")

    figure_path = Path(result["figure_path"])
    render_local_image(
        figure_path,
        caption="Detected pitch over time",
        missing_message="The graph image could not be found on disk.",
    )


def render_how_to_use() -> None:
    """Render the core user flow for running one audio-to-MIDI analysis."""
    st.subheader("How to use?")
    st.markdown(
        """
1. Choose the pitch tracking mode from the sidebar:
   - `hybrid` combines pYIN and CREPE.
   - `pyin` uses the pYIN pitch tracker.
   - `crepe` uses the CREPE pitch tracker.
2. Keep `Use Spleeter preprocessing` checked for songs, recordings with accompaniment, or polyphonic samples. If the audio is monophonic, leave it unchecked.
3. Add an optional ground truth note or frequency, such as `C4` or `261.63`, if you want error metrics.
4. Upload one `.wav` audio file.
5. Click `Analyze Audio` and wait for the results.
6. Review the detected pitch, error metrics, graph, MIDI output path, and log file path.
"""
    )
    st.info(
        "Tip: For a faster first test, use a short WAV file. CREPE and Spleeter can take longer on bigger recordings."
    )


def render_help_tab() -> None:
    """Render brief glossary and usage guidance for thesis demos."""
    mono_img = ASSETS_DIR / "Monophonic-Musical-Texture-Diagram.jpg"
    poly_img = ASSETS_DIR / "Polyphonic-Musical-Texture-Diagram.jpg"
    pitch_img = ASSETS_DIR / "pitch-diagram.gif"
    octave_img = ASSETS_DIR / "octave-diagram.jpg"
    midi_img = ASSETS_DIR / "midi-diagram.webp"

    st.subheader("About")
    st.write(
        "This local interface is for single-file alpha testing of the thesis system. "
        "It runs the existing backend pipeline, saves outputs to an app-managed folder, "
        "and lets you inspect the generated MIDI, graph, and summary information."
    )

    render_how_to_use()

    st.subheader("Help and Glossary")

    with st.expander("Monophonic vs Polyphonic"):
        st.write(
            "Monophonic audio has one main pitch at a time. Polyphonic audio contains multiple "
            "simultaneous pitches, such as singing with accompaniment. The system performs best "
            "when it can isolate a dominant melodic line."
        )

        render_local_image(
            mono_img,
            caption="Monophonic musical texture",
            missing_message="The monophonic reference image could not be found.",
            width=400,
        )
        render_local_image(
            poly_img,
            caption="Polyphonic musical texture",
            missing_message="The polyphonic reference image could not be found.",
            width=400,
        )

    with st.expander("Pitch"):
        st.write(
            "Pitch is the perceived highness or lowness of a sound. In this system, pitch is tracked "
            "frame by frame across time before being converted into MIDI notes."
        )
        render_local_image(
            pitch_img,
            caption="High and Low Pitch",
            missing_message="The pitch reference image could not be found.",
            width=900,
        )

    with st.expander("Frequency"):
        st.write(
            "Frequency is the physical measurement of pitch in Hertz (Hz). Higher Hz values correspond "
            "to higher musical pitches."
        )

    with st.expander("Octave"):
        st.write(
            "An octave is a doubling or halving of frequency. For example, 440 Hz and 880 Hz are one "
            "octave apart even though they share the same note name class."
        )
        render_local_image(
            octave_img,
            caption="octave",
            missing_message="The octave reference image could not be found.",
            width=900,
        )

    with st.expander("MIDI"):
        st.write(
            "MIDI is a symbolic music format. It stores note events such as pitch, timing, and duration, "
            "rather than storing raw audio."
        )
        render_local_image(
            midi_img,
            caption="MIDI file sample",
            missing_message="The midi reference image could not be found.",
            width=600,
        )


    with st.expander("How to Read the Graph"):
        st.write(
            "The horizontal axis is time in seconds. The vertical axis is detected frequency in Hz. "
            "A smoother line usually indicates more stable pitch tracking. Sudden jumps or spikes can "
            "signal unstable detection, octave errors, or noisy input."
        )

    with st.expander("System Limitations"):
        st.write(
            "The system is still an alpha-stage thesis tool. Strong background accompaniment, noisy "
            "recordings, heavy overlap of multiple pitched sources, or imperfect source separation can "
            "reduce accuracy. MIDI output is an approximation of detected pitch events, not a full music score."
        )

def main() -> None:
    st.set_page_config(
        page_title="Hybrid Audio-to-MIDI Thesis System",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Hybrid Audio-to-MIDI Thesis System")
    st.caption(
        "Single-file analysis UI for testing the refactored backend pipeline that extracts pitch, "
        "generates MIDI, and saves graph/log outputs."
    )

    st.sidebar.header("Analysis Settings")
    mode = st.sidebar.selectbox("Pitch tracking mode", MODES, index=0)
    use_spleeter = st.sidebar.checkbox(
        "Use Spleeter preprocessing",
        value=True,
        help="Enable source separation before pitch extraction. This is useful for mixed recordings.",
    )
    ground_truth = st.sidebar.text_input(
        "Optional ground truth",
        value="",
        placeholder="C4 or 261.63",
        help="Optional note name or frequency in Hz for error metrics.",
    ).strip()

    analyze_tab, help_tab = st.tabs(["Analyze", "About / Help / Glossary"])

    with analyze_tab:
        with st.expander("How to use?", expanded=True):
            render_how_to_use()

        uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

        if uploaded_file is not None:
            st.audio(uploaded_file.getvalue(), format="audio/wav")
        else:
            st.info("Upload one .wav file to begin.")

        if st.button("Analyze Audio", disabled=uploaded_file is None):
            if uploaded_file is None:
                st.error("Please upload a WAV file before starting analysis.")
            else:
                try:
                    run_root, input_path = save_uploaded_file(uploaded_file)
                    with st.spinner("Running analysis. This may take longer when CREPE or Spleeter is enabled."):
                        result = analyze_audio(
                            input_path=str(input_path),
                            mode=mode,
                            use_spleeter=use_spleeter,
                            ground_truth=ground_truth or None,
                            output_root=str(run_root),
                            verbose=False,
                        )
                    st.session_state["analysis_result"] = result
                    st.session_state["analysis_run_root"] = str(run_root)
                    st.success("Analysis completed successfully.")
                except Exception as exc:
                    st.session_state.pop("analysis_result", None)
                    st.error(f"Analysis failed: {exc}")

        result = st.session_state.get("analysis_result")
        if result:
            render_results(result)

    with help_tab:
        render_help_tab()


if __name__ == "__main__":
    main()
