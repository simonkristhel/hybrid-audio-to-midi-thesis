from modules.crepe_module import extract_crepe_pitch
from modules.pyin_module import hybrid_pyin
from modules.midi_module import pitch_df_to_midi

audio_path = "data/separated_audio/test/vocals.wav"

crepe_df = extract_crepe_pitch(audio_path)
hybrid_df = hybrid_pyin(audio_path, crepe_df)

midi_path = pitch_df_to_midi(
    hybrid_df,
    output_path="data/midi_output/test_output.mid"
)

print("MIDI file created at:", midi_path)
