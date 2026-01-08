import os
import soundfile as sf

INPUT_ROOT = "data/sentence_level/reference/LibriSpeech"
OUTPUT_ROOT = "data/sentence_level/reference_wav/LibriSpeech"

for root, dirs, files in os.walk(INPUT_ROOT):
    for file in files:
        if file.lower().endswith(".flac"):
            flac_path = os.path.join(root, file)

            # keep same folder structure
            rel_path = os.path.relpath(root, INPUT_ROOT)
            out_dir = os.path.join(OUTPUT_ROOT, rel_path)
            os.makedirs(out_dir, exist_ok=True)

            wav_path = os.path.join(out_dir, file.replace(".flac", ".wav"))

            audio, sr = sf.read(flac_path)
            sf.write(wav_path, audio, sr)

print(" FLAC to WAV conversion completed!")
