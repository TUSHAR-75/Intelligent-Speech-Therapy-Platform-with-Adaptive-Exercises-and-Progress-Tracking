import whisperx
import torch

# 1. Decide device
device = "cpu"
print("Using device:", device)

# 2. Load model (CPU safe + SILERO VAD)
print("Loading WhisperX model...")
model = whisperx.load_model(
    "small",
    device,
    compute_type="int8",
    vad_method="silero"
)

# 3. Pick ONE audio file
audio_path = r"data/sentence_level/reference_wav/LibriSpeech/test-clean/6829/68771/6829-68771-0000.wav"
print("Audio path:", audio_path)

# 4. Transcribe
print("Transcribing...")
result = model.transcribe(audio_path)

# 5. Print result
print("==== TRANSCRIPTION RESULT ====")
print(result["text"])
