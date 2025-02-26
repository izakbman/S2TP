import os
import librosa
import librosa.display
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio as aud  # To play audio in the notebook

# Set storage path on Talapas (currently local)
TALAPAS_STORAGE_PATH = "local_data"  
os.makedirs(TALAPAS_STORAGE_PATH, exist_ok=True)

# LibriSpeech ASR dummy dataset from Hugging Face
def download_librispeech_data():
    """Download the LibriSpeech ASR dummy dataset from Hugging Face."""
    dataset = load_dataset(
        "patrickvonplaten/librispeech_asr_dummy",
        "clean",
        split="validation",
        cache_dir=TALAPAS_STORAGE_PATH
    )
    print(f"Downloaded LibriSpeech dataset to: {TALAPAS_STORAGE_PATH}")
    return dataset

# Custom Dataset Class for Audio Data
class LibriSpeechDataset(Dataset):
    """Custom Dataset for loading LibriSpeech audio files with Librosa."""
    def __init__(self, dataset, max_len=None):
        self.dataset = dataset  # Hugging Face dataset object
        self.audio_files = [sample["file"] for sample in dataset]  # List of audio file paths
        self.texts = [sample["text"] for sample in dataset]  # List of transcriptions
        self.max_len = max_len  # Maximum length of audio sequences, if specified

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # Load audio file with Librosa
        audio_path = self.audio_files[idx]
        y, sr = librosa.load(audio_path, sr=16000)  # LibriSpeech is typically 16kHz
        
        # Get the corresponding transcription
        text = self.texts[idx]
        
        return y, sr, text  # Return the waveform (y), sample rate (sr), and transcription (text)

# DataLoader
def setup_dataloader(dataset, batch_size=32, max_len=None):
    """Create a DataLoader for the LibriSpeech dataset."""
    custom_dataset = LibriSpeechDataset(dataset, max_len=max_len)
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader

# Collate function to handle variable-length sequences in the DataLoader
def collate_fn(batch):
    # Pad sequences to the same length
    audio_lengths = [len(item[0]) for item in batch]
    max_len = max(audio_lengths)
    
    padded_audio = []
    texts = []
    for audio, sr, text in batch:
        padding = max_len - len(audio)
        padded_audio.append(F.pad(torch.tensor(audio), (0, padding), mode="constant"))
        texts.append(text)  # Collect the transcriptions
    
    return torch.stack(padded_audio), sr, texts  # Return padded audio, sample rate, and transcriptions

# Visualize the First Audio Sample's Waveform and Spectrogram
def visualize_waveform_and_spectrogram(y, sr=16000, text=None):
    """Visualize the waveform and Mel spectrogram of the audio, and print the transcription."""
    
    # Trim audio for purposes of visualization
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)  # top_db sets the threshold for silence
    y = y_trimmed
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"Audio duration: {duration} seconds")
    
    # Print the real transcription if provided
    if text:
        print(f"Transcription: {text}")
    
    display(aud(y, rate=sr))  # Play the audio in the notebook
    
    # Plot the waveform
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

    # Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8_000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Spectrogram
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sr, fmax=8_000)
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.show()

# Main execution for project milestone
if __name__ == "__main__":
    # Download the dataset
    librispeech_dataset = download_librispeech_data()

    batch_size = 64 
    max_len = 16000
    dataloader = setup_dataloader(librispeech_dataset, batch_size=batch_size, max_len=max_len)
    for i, (audio, sr, texts) in enumerate(dataloader):
        if i == 0:
            first_sample = audio[0].numpy()  # Get the first sample's audio data
            first_text = texts[0]  # Get the first sample's transcription
            visualize_waveform_and_spectrogram(first_sample, sr, text=first_text)  # Visualize and print text
            break
    
    print("Visualization complete!")