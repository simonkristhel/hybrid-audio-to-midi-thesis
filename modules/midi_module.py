import pretty_midi
import numpy as np
import pandas as pd
from pathlib import Path


def hz_to_midi(frequency):
    """Convert frequency in Hz to MIDI note number with better accuracy."""
    if frequency <= 0:
        return None
    # A4 (440 Hz) = MIDI 69
    midi_note = 69 + 12 * np.log2(frequency / 440.0)
    # Use banker's rounding for more accuracy
    return int(np.round(midi_note))


def pitch_df_to_midi(
    pitch_df: pd.DataFrame,
    output_path: str,
    min_note_duration: float = 0.05
):
    """
    Convert hybrid pitch DataFrame to MIDI file.

    Parameters:
    - pitch_df (DataFrame): columns ['time', 'hybrid_f0']
    - output_path (str): path to output .mid file
    - min_note_duration (float): minimum note length in seconds
    """

    output_path = Path(output_path)

    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Acoustic Grand Piano"))

    times = pitch_df["time"].values
    freqs = pitch_df["hybrid_f0"].values

    current_note = None
    note_start = None

    for i in range(len(freqs)):
        freq = freqs[i]
        time = times[i]
        
        # Get MIDI note, or None if frequency is 0 or invalid
        midi_note = hz_to_midi(freq) if freq > 0 else None
        
        if midi_note is None:
            # Silence detected - end current note if one exists
            if current_note is not None:
                duration = time - note_start
                if duration >= min_note_duration:
                    instrument.notes.append(
                        pretty_midi.Note(
                            velocity=100,
                            pitch=current_note,
                            start=note_start,
                            end=time
                        )
                    )
                current_note = None
        elif midi_note != current_note:
            # Note change detected - end previous note and start new one
            if current_note is not None:
                duration = time - note_start
                if duration >= min_note_duration:
                    instrument.notes.append(
                        pretty_midi.Note(
                            velocity=100,
                            pitch=current_note,
                            start=note_start,
                            end=time
                        )
                    )
            current_note = midi_note
            note_start = time

    # 🔥 ADD THIS BLOCK HERE
    if current_note is not None:
        final_time = times[-1]
        duration = final_time - note_start

        if duration >= min_note_duration:
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=100,
                    pitch=current_note,
                    start=note_start,
                    end=final_time
                )
            )

    midi.instruments.append(instrument)
    midi.write(str(output_path))


    return output_path
