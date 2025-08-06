import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
import numpy as np

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the speaker encoder model
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
).to(device)

# Path to your file
file_path = r"C:\Users\HP\Desktop\DS\Audio data\ASR\Mine\mine50.wav"

# Load the audio
signal, sr = torchaudio.load(file_path)

# Make sure it's mono
if signal.shape[0] > 1:
    signal = torch.mean(signal, dim=0, keepdim=True)

# Move signal to device
signal = signal.to(device)

# Make sure sample rate is 16000
if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000).to(device)
    signal = resampler(signal)

# Extract embedding
with torch.no_grad():
    embedding = classifier.encode_batch(signal).squeeze().cpu().numpy()

print("Embedding shape:", embedding.shape)
print(embedding)
# Optionally save embedding
np.save("mine_embedding.npy", embedding)
print("Embedding saved to 'mine_embedding.npy'")
