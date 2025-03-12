import os
import torch
from datasets import load_dataset
from transformers import Wav2Vec2Processor
import librosa
import librosa.display
import matplotlib.pyplot as plt
import argparse
from IPython.display import display, Audio as aud
import numpy as np

# Function to set storage path based on Talapas flag
def get_storage_path(use_talapas=False, talapas_base="/home/iboardma/projdir/S2TP"):
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

# Download and split LibriSpeech dataset
def download_and_split_librispeech_data(storage_path="local_data"):
    """Download and split the LibriSpeech dataset into train, validation, and test."""
    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation", trust_remote_code=True)
    dataset = dataset.shuffle(seed=42)  # Shuffle the dataset
    
    train_size = int(0.8 * len(dataset))
    valid_size = int(0.1 * len(dataset))
    
    # Split into train, valid, test
    train_dataset = dataset.select(range(train_size))
    valid_dataset = dataset.select(range(train_size, train_size + valid_size))
    test_dataset = dataset.select(range(train_size + valid_size, len(dataset)))
    
    return train_dataset, valid_dataset, test_dataset

# Preprocess audio and tokenize with Wav2Vec2Processor
def preprocess_audio(example):
    audio = example["audio"]["array"]
    # Process the audio input using the processor, padding to the longest sequence in the batch
    inputs = processor(audio, return_tensors="pt", padding="longest")
    input_values = inputs.input_values.squeeze(0)
    
    # Tokenize the text labels
    labels = processor.tokenizer(example["text"], return_tensors="pt", padding="longest")
    label_ids = labels.input_ids.squeeze(0)
    
    return {"input_values": input_values, "labels": label_ids}

# Function to visualize waveform and spectrogram
def visualize_waveform_and_spectrogram(data, sr=16000, text=None):

    y = np.array(data['array'])
    
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

# Preprocess and save tensors with padding handled by processor
def preprocess_and_save_tensors(train_dataset, valid_dataset, test_dataset, storage_path="local_data", display=False):
    
    if display:
        print("Displaying first audio example: ")
        first_example = train_dataset[0]
        visualize_waveform_and_spectrogram(first_example['audio'], text=first_example["text"])

        
    train_data = train_dataset.map(preprocess_audio, remove_columns=["audio", "file", "text"])
    valid_data = valid_dataset.map(preprocess_audio, remove_columns=["audio", "file", "text"])
    test_data = test_dataset.map(preprocess_audio, remove_columns=["audio", "file", "text"])


    # Save the processed tensors as files
    torch.save([d["input_values"] for d in train_data], os.path.join(storage_path, 'train_input_values.pt'))
    torch.save([d["labels"] for d in train_data], os.path.join(storage_path, 'train_labels.pt'))
    torch.save([d["input_values"] for d in valid_data], os.path.join(storage_path, 'valid_input_values.pt'))
    torch.save([d["labels"] for d in valid_data], os.path.join(storage_path, 'valid_labels.pt'))
    torch.save([d["input_values"] for d in test_data], os.path.join(storage_path, 'test_input_values.pt'))
    torch.save([d["labels"] for d in test_data], os.path.join(storage_path, 'test_labels.pt'))
    for d in train_data: 
        print(d)
    print(f"Data saved to: {storage_path}")
    return train_data, valid_data, test_data

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process LibriSpeech dataset with optional Talapas storage.")
    parser.add_argument('--talapas', action='store_true', help="Use Talapas storage path instead of local")
    parser.add_argument('--display', action='store_true', help="Display first example waveform and spectrogram")
    args = parser.parse_args()

    # Set storage path based on argument
    storage_path = get_storage_path(use_talapas=args.talapas, talapas_base="/home/iboardma/projdir/S2TP")
    
    # Download and split dataset
    train_dataset, valid_dataset, test_dataset = download_and_split_librispeech_data(storage_path)

    # Preprocess and save tensors
    preprocess_and_save_tensors(train_dataset, valid_dataset, test_dataset, storage_path, display=args.display)
    
    print("Preprocessing and saving complete!")

