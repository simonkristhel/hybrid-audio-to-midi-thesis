from modules.crepe_module import extract_crepe_pitch
from modules.pyin_module import hybrid_pyin

audio_path = "data/separated_audio/test/vocals.wav"

crepe_df = extract_crepe_pitch(audio_path)
hybrid_df = hybrid_pyin(audio_path, crepe_df)

print(hybrid_df.head())
print("\nTotal frames:", len(hybrid_df))
