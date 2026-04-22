# Hybrid Audio-to-MIDI Thesis System

This project is a local Streamlit interface for testing a hybrid audio-to-MIDI pipeline. It analyzes a WAV file, extracts pitch using pYIN, CREPE, or Hybrid mode, generates a MIDI file, and saves a pitch graph plus summary log.

## Requirements

- Windows, macOS, or Linux
- Python 3.10 recommended
- PowerShell, Command Prompt, Terminal, or any shell
- A web browser

## Setup on Windows Using Command Prompt

After downloading and extracting the ZIP file:

1. Open the extracted project folder.
2. Click the folder address bar at the top of File Explorer.
3. Type `cmd`.
4. Press Enter.

This opens Command Prompt directly inside the project folder.

Then run:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `python` is not recognized, try using `py` instead:

```cmd
py -m venv .venv
.venv\Scripts\activate.bat
py -m pip install --upgrade pip
pip install -r requirements.txt
```

If you already installed the requirements and see an error like `numpy.dtype size changed`, reinstall the pinned dependencies:

```cmd
pip install --force-reinstall -r requirements.txt
```

## Run the App

Each time you reopen the project or start a new terminal session, activate the virtual environment first.

From the project folder, run:

```cmd
.venv\Scripts\activate.bat
streamlit run streamlit_app.py
```

If you are reopening the app later, run the same two commands again from the project folder before launching Streamlit.

The app should open in your browser. If it does not open automatically, copy the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Sample Audio

The `sample_audio/` folder includes six short WAV files that reviewers can use for testing:

| File | Ground truth | Spleeter |
|---|---:|---|
| `Sine Wave 261.63 Hz C4.wav` | `C4` | Off |
| `Grand Piano - 261.63 Hz C4.wav` | `C4` | Off |
| `Solo Violin - 261.63 Hz C4.wav` | `C4` | Off |
| `female_long_straight_a_C4 261.63 Hz.wav` | `C4` | Off |
| `male_long_straight_a_C4 261.63 Hz.wav` | `C4` | Off |
| `female_poly_vocal_piano_C4.wav` | `C4` | On |

The same information is also listed in:

```text
sample_audio/ground_truth.csv
```

For a first quick test, use `Sine Wave 261.63 Hz C4.wav` or `Grand Piano - 261.63 Hz C4.wav` with Spleeter turned off.

## How to Use

1. Choose a pitch tracking mode in the sidebar:
   - `hybrid` combines pYIN and CREPE.
   - `pyin` uses pYIN only.
   - `crepe` uses CREPE only.
2. Enable `Use Spleeter preprocessing` for polyphonic audio, such as songs or recordings with accompaniment. Disable it for monophonic audio.
3. Optionally enter a ground truth note or frequency, such as `C4` or `261.63`.
4. Upload one `.wav` file.
5. Click `Analyze Audio`.
6. Review the detected pitch, timing table, MIDI output path, graph, and log file path.

## Output Files

Generated files are saved under `outputs/` while using the Streamlit app. These files are not included in Git because they are created each time the system runs.

Typical outputs include:

- MIDI file: `.mid`
- Pitch graph: `.png`
- Summary log: `.txt`
- Optional separated Spleeter audio: `vocals.wav` and `accompaniment.wav`

## Notes on Runtime

Runtime depends heavily on the computer running the system. Pitch extraction is the slowest part:

- CREPE uses machine-learning based pitch detection.
- pYIN uses signal-processing based pitch detection.
- Hybrid mode runs both and fuses their outputs.
- Spleeter preprocessing can add more runtime when enabled.

For faster first tests, use a short WAV file and disable Spleeter for monophonic audio.

## Repository Cleanup Notes

The repository ignores generated outputs, cache folders, virtual environments, and local model copies. Do not upload:

- `.venv/` or `thesis_env/`
- `__pycache__/`
- `outputs/`
- `outputs_*/`
- generated alpha-test logs, figures, MIDI files, and batch runs
- `pretrained_models/`

Keep source code, requirements, assets, and small sample inputs needed for reviewers.
