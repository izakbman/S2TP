import torch.nn as nn
import numpy as np
import torch
import os
import pickle
import argparse
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from torch.utils.data import DataLoader
from jiwer import wer, cer
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# Load saved raw data from pickle files
def load_data(file_path):
    """Load raw audio arrays and labels from pickle files."""
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Preprocessing functions
def preprocess_data_raw(data, processor, sampling_rate=16000):
    """Preprocess for SimpleCTCModel with raw audio."""
    input_values_list = []
    labels_list = []
    for audio_array, text in data:
        audio_array = audio_array.astype(np.float32) / np.max(np.abs(audio_array) + 1e-9)  # Normalize raw audio
        inputs = torch.tensor(audio_array, dtype=torch.float32)  # Raw audio tensor
        labels = processor.tokenizer(text, return_tensors="pt", padding=False).input_ids.squeeze(0)
        input_values_list.append(inputs)
        labels_list.append(labels)
    return input_values_list, labels_list

def preprocess_data_wav2vec(data, processor, sampling_rate=16000):
    """Preprocess for Wav2Vec2 with its processor."""
    input_values_list = []
    labels_list = []
    for audio_array, text in data:
        audio_array = audio_array.astype(np.float32)
        inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt", padding=False).input_values.squeeze(0)
        labels = processor.tokenizer(text, return_tensors="pt", padding=False).input_ids.squeeze(0)
        input_values_list.append(inputs)
        labels_list.append(labels)
    return input_values_list, labels_list

# Load datasets
storage_path = "librispeech_cache"
test_data = load_data(os.path.join(storage_path, "test_data.pkl"))
train_data = load_data(os.path.join(storage_path, "train_data.pkl"))

# Define SimpleCTCModel
class SimpleCTCModel(nn.Module):
    def __init__(self, vocab_size=32, feature_dim=64):
        super(SimpleCTCModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, feature_dim, kernel_size=10, stride=5, padding=3),
            nn.ReLU()
        )
        self.fc = nn.Linear(feature_dim, vocab_size)
    
    def forward(self, input_values, labels=None):
        input_values = input_values.unsqueeze(1)  # [batch_size, 1, sequence_length]
        features = self.feature_extractor(input_values)  # [batch_size, feature_dim, reduced_length]
        features = features.transpose(1, 2)  # [batch_size, reduced_length, feature_dim]
        logits = self.fc(features)  # [batch_size, reduced_length, vocab_size]
        
        if labels is not None:
            input_lengths = torch.tensor([int(inputs.shape[-1] / 5) for inputs in input_values], dtype=torch.long, device=device)
            label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long, device=device)
            loss = nn.CTCLoss(blank=0)(logits.transpose(0, 1), labels, input_lengths, label_lengths)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

# Custom collate function
def custom_collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0.0).to(device)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=processor.tokenizer.pad_token_id).to(device)
    return inputs_padded, labels_padded

# Greedy decode for SimpleCTCModel
def greedy_decode(predicted_ids, blank_id=0):
    decoded = []
    prev = None
    for t in predicted_ids:
        curr = t.item()
        if curr != blank_id and curr != prev:
            decoded.append(curr)
        prev = curr
    return decoded

# Argument parser
parser = argparse.ArgumentParser(description="Evaluate ASR models on LibriSpeech test set")
parser.add_argument("--mymodel", action="store_true", help="Evaluate SimpleCTCModel")
parser.add_argument("--wav2vec", action="store_true", help="Evaluate Wav2Vec2 model")
args = parser.parse_args()

# Prepare data
batch_size = 4
if args.mymodel:
    train_inputs_raw, train_labels_raw = preprocess_data_raw(train_data, processor)
    test_inputs_raw, test_labels_raw = preprocess_data_raw(test_data, processor)
    train_dataset_raw = list(zip(train_inputs_raw, train_labels_raw))
    test_dataset_raw = list(zip(test_inputs_raw, test_labels_raw))
    train_loader_raw = DataLoader(train_dataset_raw, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    test_loader_raw = DataLoader(test_dataset_raw, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

if args.wav2vec:
    test_inputs_wav2vec, test_labels_wav2vec = preprocess_data_wav2vec(test_data, processor)
    test_dataset_wav2vec = list(zip(test_inputs_wav2vec, test_labels_wav2vec))
    test_loader_wav2vec = DataLoader(test_dataset_wav2vec, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

# Train and evaluate SimpleCTCModel
if args.mymodel:
    print("Training SimpleCTCModel...")
    model = SimpleCTCModel(vocab_size=processor.tokenizer.vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 10
    
    model.train()
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        for inputs, labels in train_loader_raw:
            optimizer.zero_grad()
            outputs = model(inputs, labels=labels)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss / len(train_loader_raw):.4f}")

    # Evaluate SimpleCTCModel
    model.eval()
    all_predicted_transcriptions_mymodel = []
    all_actual_transcriptions_mymodel = []
    with torch.no_grad():
        for inputs, labels in test_loader_raw:
            outputs = model(inputs)
            logits = outputs["logits"]
            predicted_ids = torch.argmax(logits, dim=-1)
            
            for i in range(predicted_ids.size(0)):
                pred_ids = greedy_decode(predicted_ids[i], blank_id=processor.tokenizer.pad_token_id)
                pred_text = processor.tokenizer.decode(pred_ids)
                all_predicted_transcriptions_mymodel.append(pred_text)
            
            labels_cpu = [label.cpu().tolist() for label in labels]
            all_actual_transcriptions_mymodel.extend(processor.batch_decode(labels_cpu))

    cer_score_mymodel = cer(all_actual_transcriptions_mymodel, all_predicted_transcriptions_mymodel)
    wer_score_mymodel = wer(all_actual_transcriptions_mymodel, all_predicted_transcriptions_mymodel)
    print(f"\nSimpleCTCModel - CER: {cer_score_mymodel:.4f}, WER: {wer_score_mymodel:.4f}")

# Evaluate Wav2Vec2 model
if args.wav2vec:
    print("Evaluating Wav2Vec2 model...")
    wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    wav2vec_model.eval()
    all_predicted_transcriptions_wav2vec = []
    all_actual_transcriptions_wav2vec = []
    
    with torch.no_grad():
        for inputs, labels in test_loader_wav2vec:
            outputs = wav2vec_model(inputs).logits  # Wav2Vec2 outputs logits directly
            predicted_ids = torch.argmax(outputs, dim=-1)
            
            # Wav2Vec2 handles CTC decoding internally via processor
            pred_texts = processor.batch_decode(predicted_ids)
            all_predicted_transcriptions_wav2vec.extend(pred_texts)
            
            labels_cpu = [label.cpu().tolist() for label in labels]
            all_actual_transcriptions_wav2vec.extend(processor.batch_decode(labels_cpu))

    cer_score_wav2vec = cer(all_actual_transcriptions_wav2vec, all_predicted_transcriptions_wav2vec)
    wer_score_wav2vec = wer(all_actual_transcriptions_wav2vec, all_predicted_transcriptions_wav2vec)
    print(f"\nWav2Vec2 - CER: {cer_score_wav2vec:.4f}, WER: {wer_score_wav2vec:.4f}")

# Compare results
if args.mymodel and args.wav2vec:
    print("\nComparison:")
    print(f"SimpleCTCModel - CER: {cer_score_mymodel:.4f}, WER: {wer_score_mymodel:.4f}")
    print(f"Wav2Vec2       - CER: {cer_score_wav2vec:.4f}, WER: {wer_score_wav2vec:.4f}")