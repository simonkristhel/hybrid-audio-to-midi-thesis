from modules.crepe_module import extract_crepe_pitch

pitch_df = extract_crepe_pitch(
    audio_path="data/separated_audio/test/vocals.wav"
)

print(pitch_df.head())
print("\nTotal frames:", len(pitch_df))
