from modules.spleeter_module import separate_monophonic_audio

output = separate_monophonic_audio(
    input_audio_path="data/raw_audio/test.wav",
    output_dir="data/separated_audio"
)

print("Separated file saved at:", output)
