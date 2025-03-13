import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from transformers import Wav2Vec2Processor
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

# Load datasets
storage_path = "librispeech_cache"
train_data = load_data(os.path.join(storage_path, "train_data.pkl"))
test_data = load_data(os.path.join(storage_path, "test_data.pkl"))

# Preprocess all data into lists of tensors
def preprocess_data(data, processor, sampling_rate=16000):
    input_values_list = []
    labels_list = []
    for audio_array, text in data:
        audio_array = audio_array.astype(np.float32)
        inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt", padding=False).input_values.squeeze(0)
        labels = processor.tokenizer(text, return_tensors="pt", padding=False).input_ids.squeeze(0)  # Tokenized IDs
        
        input_values_list.append(inputs)
        labels_list.append(labels)
    return input_values_list, labels_list

# Process train and test data
train_inputs, train_labels = preprocess_data(train_data, processor)
test_inputs, test_labels = preprocess_data(test_data, processor)

# Very basic custom model for CTC
class BasicCTCModel(nn.Module):
    def __init__(self, vocab_size=32, feature_dim=128):
        super(BasicCTCModel, self).__init__()
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
            input_lengths = torch.full((input_values.size(0),), logits.size(1), dtype=torch.long, device=device)
            label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long, device=device)
            loss = nn.CTCLoss(blank=0)(logits.transpose(0, 1), labels, input_lengths, label_lengths)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

# Custom collate function to pad and move to device
def custom_collate_fn(batch):
    """Pad inputs and labels, ensuring labels remain tokenized tensors."""
    inputs, labels = zip(*batch)
    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0.0).to(device)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=processor.tokenizer.pad_token_id).to(device)
    return inputs_padded, labels_padded

# Create datasets and dataloaders
train_dataset = list(zip(train_inputs, train_labels))
test_dataset = list(zip(test_inputs, test_labels))

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

# Initialize basic model
model = BasicCTCModel(vocab_size=processor.tokenizer.vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 20
model.train()
for epoch in tqdm(range(num_epochs)):
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss / len(train_loader):.4f}")

# Evaluation on full test set
model.eval()
all_predicted_ids = []
all_actual_transcriptions = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        logits = outputs["logits"]
        predicted_ids = torch.argmax(logits, dim=-1)
        all_predicted_ids.extend(predicted_ids.cpu().tolist())  # Move to CPU for decoding
        all_actual_transcriptions.extend(processor.batch_decode(labels))

# Decode predictions
all_predicted_transcriptions = processor.batch_decode(all_predicted_ids)

# Calculate CER and WER
cer_score = cer(all_actual_transcriptions, all_predicted_transcriptions)
wer_score = wer(all_actual_transcriptions, all_predicted_transcriptions)

# Print results
print(f"\nCharacter Error Rate (CER): {cer_score:.4f}")
print(f"Word Error Rate (WER): {wer_score:.4f}")

# Print sample transcriptions
print("\nSample transcriptions (first 5):")
for i in range(min(5, len(all_actual_transcriptions))):
    print(f"Actual: {all_actual_transcriptions[i]}")
    print(f"Predicted: {all_predicted_transcriptions[i]}\n")