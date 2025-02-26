import os
from huggingface_hub import hf_hub_download
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Set storage path on Talapas (adjust to your preferred directory)
TALAPAS_STORAGE_PATH = "/scratch/username/librosa_data"  # Replace 'username' with your Talapas username
os.makedirs(TALAPAS_STORAGE_PATH, exist_ok=True)

# Step 1: Download the Librosa example audio file from Hugging Face
def download_librosa_data():
    """Download the small Librosa example file from Hugging Face."""
    file_path = hf_hub_download(
        repo_id="librosa/example",
        filename="nutcracker.mp3",  # Example file; adjust if targeting a different one
        repo_type="dataset",
        cache_dir=TALAPAS_STORAGE_PATH
    )
    print(f"Downloaded Librosa example to: {file_path}")
    return file_path

# Step 2: Custom Dataset Class for Audio Data
class AudioDataset(Dataset):
    """Custom Dataset for loading audio files with Librosa."""
    def __init__(self, audio_file):
        # Load audio file with Librosa
        self.y, self.sr = librosa.load(audio_file, sr=22050)  # y = audio time series, sr = sample rate
        # Convert to spectrogram (example preprocessing)
        self.spectrogram = librosa.feature.melspectrogram(y=self.y, sr=self.sr)
        self.data = torch.tensor(self.spectrogram, dtype=torch.float32)

    def __len__(self):
        # Define length as number of time frames in spectrogram
        return self.data.shape[1]

    def __getitem__(self, idx):
        # Return a single time frame of the spectrogram
        return self.data[:, idx]

# Step 3: Set Up DataLoader
def setup_dataloader(audio_file, batch_size=32):
    """Create a DataLoader for the audio dataset."""
    dataset = AudioDataset(audio_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

# Main execution
if __name__ == "__main__":
    # Download the data
    audio_file = download_librosa_data()

    # Set up the DataLoader
    batch_size = 32  # Adjust as needed
    dataloader = setup_dataloader(audio_file, batch_size=batch_size)

    # Example: Iterate through the DataLoader
    for i, batch in enumerate(dataloader):
        print(f"Batch {i+1} shape: {batch.shape}")
        if i == 2:  # Limit to 3 batches for demo
            break

    print("Data loading complete!")