#stage 3:

import whisperx
import torch
import json
import os

# ==============================
# 1. Device
# ==============================
device = "cpu"
print("Using device:", device)

# ==============================
# 2. Load ASR model
# ==============================
print("Loading WhisperX ASR model...")
model = whisperx.load_model(
    "small",
    device,
    compute_type="int8",
    vad_method="silero"
)

# ==============================
# 3. Audio path
# ==============================
audio_path = r"C:\Users\tkber\OneDrive\Desktop\speech_therapy_project\data\sentence_level\reference_wav\LibriSpeech\test-clean\6829\68771\6829-68771-0003.wav"
print("Audio path:", audio_path)

# ==============================
# 4. Transcribe
# ==============================
print("Transcribing...")
result = model.transcribe(audio_path)

print("\n==== RAW SEGMENTS ====\n")
for seg in result["segments"]:
    print(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}")

# ==============================
# 5. Load Alignment Model
# ==============================
print("\nLoading alignment model...")
align_model, metadata = whisperx.load_align_model(
    language_code=result["language"],
    device=device
)

# ==============================
# 6. Perform Alignment
# ==============================
print("Performing forced alignment...")
aligned = whisperx.align(
    result["segments"],
    align_model,
    metadata,
    audio_path,
    device
)

# ==============================
# 7. Extract word & phoneme timestamps
# ==============================
words = []
phonemes = []

print("\n==== WORD LEVEL ALIGNMENT ====\n")

for segment in aligned["segments"]:
    if "words" not in segment:
        continue

    for w in segment["words"]:
        if w.get("start") is None or w.get("end") is None:
            continue

        word_entry = {
            "word": w["word"],
            "start": float(w["start"]),
            "end": float(w["end"])
        }
        words.append(word_entry)

        print(f"[{w['start']:.2f} - {w['end']:.2f}] {w['word']}")

        # phoneme / character level (if available)
        if "chars" in w:
            for ch in w["chars"]:
                if ch.get("start") is None or ch.get("end") is None:
                    continue

                phoneme_entry = {
                    "symbol": ch["char"],
                    "start": float(ch["start"]),
                    "end": float(ch["end"]),
                    "word": w["word"]
                }
                phonemes.append(phoneme_entry)

# ==============================
# 8. Save JSON
# ==============================
output = {
    "audio_path": audio_path,
    "language": result["language"],
    "words": words,
    "phonemes": phonemes
}

os.makedirs("backend/alignment_outputs", exist_ok=True)

out_path = "backend/alignment_outputs/alignment_result.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("\n✅ Alignment JSON saved to:")
print(out_path)

print("\n==== DONE ====")


##stage 2
# import whisperx
# import torch

# # 1. Decide device
# device = "cpu"
# print("Using device:", device)

# # 2. Load model (CPU safe + SILERO VAD)
# print("Loading WhisperX model...")
# model = whisperx.load_model(
#     "small",
#     device,
#     compute_type="int8",
#     vad_method="silero"
# )

# # 3. Pick ONE audio file
# audio_path = r"C:\Users\tkber\OneDrive\Desktop\speech_therapy_project\data\sentence_level\reference_wav\LibriSpeech\test-clean\6829\68771\6829-68771-0003.wav"
# print("Audio path:", audio_path)

# # 4. Transcribe
# print("Transcribing...")
# result = model.transcribe(audio_path)

# # 5. Print result
# print("==== TRANSCRIPTION RESULT ====")
# print("\n==== TRANSCRIPTION SEGMENTS ====\n")
# for seg in result["segments"]:
#     print(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}")









# # import whisperx
# # import torch

# # # 1. Select device
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# # print("Using device:", device)

# # # 2. Load model
# # model = whisperx.load_model(
# #     "small",
# #     device,
# #     compute_type="int8",
# #     vad_method="silero"   # ✅ USE SILERO, NOT PYANNOTE
# # )



# # # 3. Path to ONE test wav (change this to one of your files)
# # audio_path = r"data/sentence_level/reference_wav/LibriSpeech/test-clean/6829/68771/6829-68771-0000.wav"

# # # 4. Transcribe
# # result = model.transcribe(audio_path)

# # print("TRANSCRIPTION RESULT:")
# # print(result["text"])
