import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import Wav2Vec2Processor
from torch.utils.data import Dataset, DataLoader
import librosa
import librosa.display
import joblib
import matplotlib.pyplot as plt
from IPython.display import Audio as aud
import argparse

# Function to set storage path based on Talapas flag
def get_storage_path(use_talapas=False, talapas_base="/home/iboardma"):
    """Set the storage path based on whether Talapas is used."""
    if use_talapas:
        storage_path = os.path.join(talapas_base, "librispeech_cache")
        print(f"Using Talapas storage path: {storage_path}")
    else:
        storage_path = "local_data"
        print(f"Using local storage path: {storage_path}")
    os.makedirs(storage_path, exist_ok=True)  # Create the directory
    return storage_path

# Load Wav2Vec2 processor (global)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# Download LibriSpeech dataset
def download_librispeech_data(split="validation", storage_path="local_data"):
    """Download the LibriSpeech ASR dataset from Hugging Face."""
    dataset = load_dataset(
        "patrickvonplaten/librispeech_asr_dummy",
        "clean",
        split=split,
        cache_dir=storage_path
    )
    dataset = dataset.select(range(min(10, len(dataset))))
    print(f"Downloaded LibriSpeech dataset to: {storage_path}")
    return dataset

# Preprocess audio and tokenize with Wav2Vec2Processor
def preprocess_audio(example):
    audio = example["audio"]["array"]
    sampling_rate = example["audio"]["sampling_rate"]
    
    if sampling_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
    
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.squeeze(0)
    labels = processor.tokenizer(example["text"], return_tensors="pt", padding=True)
    label_ids = labels.input_ids.squeeze(0)
    return {"input_values": input_values, "labels": label_ids, "raw_audio": audio}

# Custom Dataset Class for Audio Data
class LibriSpeechDataset(Dataset):
    def __init__(self, dataset, max_audio_len=None, max_text_len=None):
        self.dataset = dataset.map(preprocess_audio, remove_columns=["audio", "file"])
        self.input_values = [torch.tensor(self.dataset[i]["input_values"]) for i in range(len(self.dataset))]
        self.labels = [torch.tensor(self.dataset[i]["labels"]) for i in range(len(self.dataset))]
        self.raw_audio = [self.dataset[i]["raw_audio"] for i in range(len(self.dataset))]
        self.max_audio_len = max_audio_len or max(v.shape[0] for v in self.input_values)
        self.max_text_len = max_text_len or max(l.shape[0] for l in self.labels)
        self.texts = [sample["text"] for sample in dataset]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.input_values[idx], self.labels[idx], self.raw_audio[idx], self.texts[idx]

# Collate function to pad variable-length sequences
def collate_fn(batch):
    input_values = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    raw_audio = [item[2] for item in batch]
    texts = [item[3] for item in batch]
    
    input_values = [torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in input_values]
    labels = [torch.tensor(l) if not isinstance(l, torch.Tensor) else l for l in labels]
    raw_audio = [torch.tensor(a, dtype=torch.float32) if not isinstance(a, torch.Tensor) else a for a in raw_audio]
    
    max_audio_len = max(v.shape[0] for v in input_values)
    padded_inputs = torch.stack([F.pad(v, (0, max_audio_len - v.shape[0]), mode="constant", value=0) for v in input_values])
    
    max_text_len = max(l.shape[0] for l in labels)
    padded_labels = torch.stack([F.pad(l, (0, max_text_len - l.shape[0]), mode="constant", value=processor.tokenizer.pad_token_id) for l in labels])
    
    max_raw_len = max(a.shape[0] if isinstance(a, torch.Tensor) else len(a) for a in raw_audio)
    padded_audio = torch.stack([
        F.pad(a, (0, max_raw_len - a.shape[0]), mode="constant") if isinstance(a, torch.Tensor)
        else F.pad(torch.tensor(a, dtype=torch.float32), (0, max_raw_len - len(a)), mode="constant")
        for a in raw_audio
    ])
    
    return padded_inputs, padded_labels, padded_audio, texts

# DataLoader setup
def setup_dataloader(dataset, batch_size=32):
    custom_dataset = LibriSpeechDataset(dataset)
    return DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Visualize waveform and spectrogram
def visualize_waveform_and_spectrogram(data, sr=16000, text=None, is_tensor=False):
    if is_tensor or isinstance(data, torch.Tensor):
        y = data.cpu().numpy()
    else:
        y = data
    
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    y = y_trimmed
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"Audio duration: {duration} seconds")
    
    if text:
        print(f"Transcription: {text}")
    
    display(aud(y, rate=sr))
    
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sr, fmax=8000)
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.show()

# Preprocess and save tensors with padding
def preprocess_and_save_tensors(dataset, seed=102):
    dataset_shuffled = dataset.shuffle(seed=seed)
    train_size = int(0.8 * len(dataset))
    valid_size = int(0.1 * len(dataset))
    
    train_dataset = dataset_shuffled.select(range(train_size))
    valid_dataset = dataset_shuffled.select(range(train_size, train_size + valid_size))
    test_dataset = dataset_shuffled.select(range(train_size + valid_size, len(dataset)))
    
    train_data = train_dataset.map(preprocess_audio, remove_columns=["audio", "text", "file"])
    valid_data = valid_dataset.map(preprocess_audio, remove_columns=["audio", "text", "file"])
    test_data = test_dataset.map(preprocess_audio, remove_columns=["audio", "text", "file"])
    
    train_input_values = [torch.tensor(d["input_values"]) for d in train_data]
    train_labels = [torch.tensor(d["labels"]) for d in train_data]
    valid_input_values = [torch.tensor(d["input_values"]) for d in valid_data]
    valid_labels = [torch.tensor(d["labels"]) for d in valid_data]
    test_input_values = [torch.tensor(d["input_values"]) for d in test_data]
    test_labels = [torch.tensor(d["labels"]) for d in test_data]
    
    max_audio_len_train = max(v.shape[0] for v in train_input_values)
    max_label_len_train = max(l.shape[0] for l in train_labels)
    max_audio_len_valid = max(v.shape[0] for v in valid_input_values)
    max_label_len_valid = max(l.shape[0] for l in valid_labels)
    max_audio_len_test = max(v.shape[0] for v in test_input_values)
    max_label_len_test = max(l.shape[0] for l in test_labels)
    
    X_train_tensor = torch.stack([F.pad(v, (0, max_audio_len_train - v.shape[0]), value=0) for v in train_input_values])
    y_train_tensor = torch.stack([F.pad(l, (0, max_label_len_train - l.shape[0]), value=processor.tokenizer.pad_token_id) for l in train_labels])
    X_valid_tensor = torch.stack([F.pad(v, (0, max_audio_len_valid - v.shape[0]), value=0) for v in valid_input_values])
    y_valid_tensor = torch.stack([F.pad(l, (0, max_label_len_valid - l.shape[0]), value=processor.tokenizer.pad_token_id) for l in valid_labels])
    X_test_tensor = torch.stack([F.pad(v, (0, max_audio_len_test - v.shape[0]), value=0) for v in test_input_values])
    y_test_tensor = torch.stack([F.pad(l, (0, max_label_len_test - l.shape[0]), value=processor.tokenizer.pad_token_id) for l in test_labels])
    
    torch.save(X_train_tensor, 'X_train.pt')
    torch.save(y_train_tensor, 'y_train.pt')
    torch.save(X_valid_tensor, 'X_valid.pt')
    torch.save(y_valid_tensor, 'y_valid.pt')
    torch.save(X_test_tensor, 'X_test.pt')
    torch.save(y_test_tensor, 'y_test.pt')
    
    print(f"Train shape: {X_train_tensor.shape}, Valid shape: {X_valid_tensor.shape}, Test shape: {X_test_tensor.shape}")
    return train_dataset, valid_dataset, test_dataset

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process LibriSpeech dataset with optional Talapas storage.")
    parser.add_argument('--talapas', action='store_true', help="Use Talapas storage path instead of local")
    args = parser.parse_args()

    # Set storage path based on argument
    TALAPAS_STORAGE_PATH = get_storage_path(use_talapas=args.talapas, talapas_base="/home/iboardma")
    
    # Create the directory if using hardcoded path (uncomment if sticking with hardcoded)
    # os.makedirs(TALAPAS_STORAGE_PATH, exist_ok=True)

    librispeech_dataset = download_librispeech_data(storage_path=TALAPAS_STORAGE_PATH)
    train_dataset, valid_dataset, test_dataset = preprocess_and_save_tensors(librispeech_dataset)

    batch_size = 64
    dataloader = setup_dataloader(librispeech_dataset, batch_size=batch_size)
    
    for i, (inputs, labels, audio, texts) in enumerate(dataloader):
        if i == 0:
            print("Visualizing raw audio of first sample:")
            first_sample_audio = audio[0]
            first_text = texts[0]
            visualize_waveform_and_spectrogram(first_sample_audio, sr=16000, text=first_text, is_tensor=True)
            break
    
    print("Visualization complete!")
    joblib.dump(processor, 'wav2vec2_processor.pkl')