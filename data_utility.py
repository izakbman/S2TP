import os
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2BertForCTC
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import pickle
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, Audio as aud

def get_storage_path(use_talapas=False, talapas_base="/home/iboardma/projdir/S2TP"):
    """Set the storage path based on whether Talapas is used."""
    if use_talapas:
        storage_path = os.path.join(talapas_base, "librispeech_cache")
        print(f"Using Talapas storage path: {storage_path}")
    else:
        storage_path = "local_data"
        print(f"Using local storage path: {storage_path}")
    os.makedirs(storage_path, exist_ok=True)
    return storage_path

def download_and_process_librispeech(storage_path):
    """Download the dummy LibriSpeech validation split and process into train/valid/test."""
    print("Downloading dummy LibriSpeech validation split...")
    dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
    
    # Process the validation split (4 samples)
    processed_data = []
    print(f"Processing validation split with {len(dataset)} samples...")
    for item in dataset:
        audio_path = item["file"]
        audio_array, sr = librosa.load(audio_path, sr=16000)
        text = item["text"]
        processed_data.append((audio_array, text))
    
    # Manually split 4 samples: 2 train, 1 valid, 1 test
    train_data = processed_data[:2]  # First 2 samples
    valid_data = processed_data[2:3]  # 3rd sample
    test_data = processed_data[3:]   # 4th sample
    
    # Save to pickle files with your naming convention
    split_files = {
        "train": train_data,
        "valid": valid_data,
        "test": test_data
    }
    
    for split_name, data in split_files.items():
        output_file = os.path.join(storage_path, f"{split_name}_data.pkl")
        with open(output_file, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved {split_name} data to {output_file} with {len(data)} samples")
    
    return train_data, valid_data, test_data

def load_librispeech_data(storage_path):
    """Load raw audio arrays and text labels from stored files, or download if missing."""
    split_files = {
        "train": os.path.join(storage_path, "train_data.pkl"),
        "valid": os.path.join(storage_path, "valid_data.pkl"),
        "test": os.path.join(storage_path, "test_data.pkl")
    }
    
    all_exist = all(os.path.exists(file) for file in split_files.values())
    
    if not all_exist:
        print("One or more data files missing. Downloading and processing LibriSpeech...")
        train_data, valid_data, test_data = download_and_process_librispeech(storage_path)
    else:
        print("Loading existing data from pickle files...")
        data_splits = {}
        for split, file_path in split_files.items():
            with open(file_path, "rb") as f:
                data_splits[split] = pickle.load(f)
        train_data, valid_data, test_data = data_splits["train"], data_splits["valid"], data_splits["test"]
    
    # Ensure data is in dictionary format
    for split_data in [train_data, valid_data, test_data]:
        if not split_data:  # Check for empty lists
            raise ValueError(f"Split {split_data} is empty. Check data processing.")
        if not isinstance(split_data[0], dict):
            split_data[:] = [{"audio": item[0], "text": item[1]} for item in split_data]
    
    return train_data, valid_data, test_data

def visualize_waveform_and_spectrogram(audio_array, sr=16000, text=None):
    """Visualize waveform and spectrogram of an audio sample."""
    y_trimmed, _ = librosa.effects.trim(audio_array, top_db=20)
    duration = librosa.get_duration(y=y_trimmed, sr=sr)
    print(f"Audio duration: {duration} seconds")
    
    if text:
        print(f"Transcription: {text}")
    
    display(aud(y_trimmed, rate=sr))
    
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y_trimmed, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()
    
    S = librosa.feature.melspectrogram(y=y_trimmed, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sr, fmax=8000)
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load and process LibriSpeech into respective splits.")
    parser.add_argument('--talapas', action='store_true', help="Use Talapas storage path instead of local")
    parser.add_argument('--display', action='store_true', help="Display first example waveform and spectrogram")
    args = parser.parse_args()
    
    storage_path = get_storage_path(use_talapas=args.talapas)
    
    # Load or download the dummy dataset
    train_data, valid_data, test_data = load_librispeech_data(storage_path)
    
    if args.display:
        first_example = train_data[0]
        visualize_waveform_and_spectrogram(first_example["audio"], text=first_example["text"])
    
    print(f"Dataset ready! Train: {len(train_data)} samples, Valid: {len(valid_data)} samples, Test: {len(test_data)} samples")