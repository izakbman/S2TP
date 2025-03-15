import torch.nn as nn
import numpy as np
import torch
import os
import pickle
import argparse
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from torch.utils.data import DataLoader
from jiwer import wer, cer
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt\

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
# Note: You have two identical preprocess_data_raw definitions; keeping the one with lengths
def preprocess_data_raw(data, processor, sampling_rate=16000):
    """Preprocess for SimpleCTCModel with raw audio, including original lengths."""
    input_values_list = []
    labels_list = []
    input_lengths_list = []  # Store original lengths before padding
    for audio_array, text in data:
        audio_array = audio_array.astype(np.float32) / np.max(np.abs(audio_array) + 1e-9)  # Normalize raw audio
        inputs = torch.tensor(audio_array, dtype=torch.float32)  # Raw audio tensor
        labels = processor.tokenizer(text, return_tensors="pt", padding=False).input_ids.squeeze(0)
        input_values_list.append(inputs)
        labels_list.append(labels)
        input_lengths_list.append(inputs.size(0))  # Original length of audio
    return input_values_list, labels_list, input_lengths_list

# Assuming preprocess_data_wav2vec is missing; adding it for completeness
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

class SimpleCTCModel(nn.Module):
    def __init__(self, vocab_size=32, feature_dim=64):
        super(SimpleCTCModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, feature_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(feature_dim)
        )
        self.fc = nn.Linear(feature_dim, vocab_size)
    
    def forward(self, input_values, input_lengths=None, labels=None):
        input_values = input_values.unsqueeze(1)  # [batch_size, 1, sequence_length]
        features = self.feature_extractor(input_values)  # [batch_size, feature_dim, reduced_length]
        features = features.transpose(1, 2)  # [batch_size, reduced_length, feature_dim]
        logits = self.fc(features)  # [batch_size, reduced_length, vocab_size]
        
        if labels is not None and input_lengths is not None:
            # Adjust input_lengths for Conv1d: floor((L + 2*padding - kernel_size) / stride) + 1
            reduced_lengths = torch.tensor([(length + 2 * 1 - 4) // 2 + 1 for length in input_lengths], 
                                         dtype=torch.long, device=device)
            reduced_lengths = reduced_lengths.clamp(min=1)  # Ensure no zero/negative lengths
            label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long, device=device)
            
            # CTCLoss expects [T, N, C]
            log_probs = F.log_softmax(logits.transpose(0, 1), dim=-1)  # [reduced_length, batch_size, vocab_size]
            loss = nn.CTCLoss(blank=0)(log_probs, labels, reduced_lengths, label_lengths)
            return {"loss": loss, "logits": logits}
        
        return {"logits": logits}

# Custom collate function
def custom_collate_fn(batch):
    inputs, labels, input_lengths = zip(*batch)  # Unpack 3 values
    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0.0).to(device)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=processor.tokenizer.pad_token_id).to(device)
    input_lengths_tensor = torch.tensor(input_lengths, dtype=torch.long).to(device)
    return inputs_padded, labels_padded, input_lengths_tensor

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
parser = argparse.ArgumentParser(description="Train or evaluate ASR models on LibriSpeech test set")
parser.add_argument("--mymodel", action="store_true", help="Use SimpleCTCModel")
parser.add_argument("--wav2vec", action="store_true", help="Use Wav2Vec2 model")
parser.add_argument("--eval", action="store_true", help="Skip training and evaluate only, using saved weights if available")
args = parser.parse_args()

# Ensure evaluation folder exists
eval_folder = "evaluation"
os.makedirs(eval_folder, exist_ok=True)

# Prepare data
batch_size = 4
sr = 16000
if args.mymodel:
    train_inputs_raw, train_labels_raw, train_lengths_raw = preprocess_data_raw(train_data, processor, sr)
    test_inputs_raw, test_labels_raw, test_lengths_raw = preprocess_data_raw(test_data, processor, sr)
    train_dataset_raw = list(zip(train_inputs_raw, train_labels_raw, train_lengths_raw))
    test_dataset_raw = list(zip(test_inputs_raw, test_labels_raw, test_lengths_raw))
    train_loader_raw = DataLoader(train_dataset_raw, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    test_loader_raw = DataLoader(test_dataset_raw, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

if args.wav2vec:
    test_inputs_wav2vec, test_labels_wav2vec = preprocess_data_wav2vec(test_data, processor, sr)
    test_dataset_wav2vec = list(zip(test_inputs_wav2vec, test_labels_wav2vec))
    test_loader_wav2vec = DataLoader(test_dataset_wav2vec, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

# Train and evaluate SimpleCTCModel
if args.mymodel:
    model_weights_path = os.path.join(eval_folder, "simple_ctc_model.pth")
    model = SimpleCTCModel(vocab_size=processor.tokenizer.vocab_size).to(device)
    
    if args.eval:
        if os.path.exists(model_weights_path):
            print(f"Loading saved weights for SimpleCTCModel from {model_weights_path}")
            model.load_state_dict(torch.load(model_weights_path))
        else:
            print(f"No saved weights found at {model_weights_path}")
            response = input("Would you like to train the model now? (yes/no): ").strip().lower()
            if response != "yes":
                print("Skipping SimpleCTCModel evaluation due to missing weights.")
                args.mymodel = False
            else:
                args.eval = False  # Switch to training mode
    
    if not args.eval:
        print("Training SimpleCTCModel...")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        num_epochs = 10
        
        model.train()
        for epoch in tqdm(range(num_epochs)):
            total_loss = 0
            for inputs, labels, input_lengths in train_loader_raw:
                optimizer.zero_grad()
                outputs = model(inputs, labels=labels, input_lengths=input_lengths)
                loss = outputs["loss"]
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss / len(train_loader_raw):.4f}")
        
        # Save weights after training
        torch.save(model.state_dict(), model_weights_path)
        print(f"Saved trained weights to {model_weights_path}")

    # Evaluate SimpleCTCModel
    if args.mymodel:  # Only eval if still active
        model.eval()
        all_predicted_transcriptions_mymodel = []
        all_actual_transcriptions_mymodel = []
        with torch.no_grad():
            for inputs, labels, _ in test_loader_raw:
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

        # Save evaluation plot
        plt.figure(figsize=(6, 4))
        plt.bar(["CER", "WER"], [cer_score_mymodel, wer_score_mymodel], color=['blue', 'orange'])
        plt.title("SimpleCTCModel Evaluation Metrics")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.savefig(os.path.join(eval_folder, "simple_ctc_metrics.png"))
        plt.close()

# Evaluate Wav2Vec2 model
if args.wav2vec:
    print("Evaluating Wav2Vec2 model...")
    wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    wav2vec_model.eval()
    all_predicted_transcriptions_wav2vec = []
    all_actual_transcriptions_wav2vec = []
    
    with torch.no_grad():
        for inputs, labels in test_loader_wav2vec:
            outputs = wav2vec_model(inputs).logits
            predicted_ids = torch.argmax(outputs, dim=-1)
            pred_texts = processor.batch_decode(predicted_ids)
            all_predicted_transcriptions_wav2vec.extend(pred_texts)
            
            labels_cpu = [label.cpu().tolist() for label in labels]
            all_actual_transcriptions_wav2vec.extend(processor.batch_decode(labels_cpu))

    cer_score_wav2vec = cer(all_actual_transcriptions_wav2vec, all_predicted_transcriptions_wav2vec)
    wer_score_wav2vec = wer(all_actual_transcriptions_wav2vec, all_predicted_transcriptions_wav2vec)
    print(f"\nWav2Vec2 - CER: {cer_score_wav2vec:.4f}, WER: {wer_score_wav2vec:.4f}")

    # Save evaluation plot
    plt.figure(figsize=(6, 4))
    plt.bar(["CER", "WER"], [cer_score_wav2vec, wer_score_wav2vec], color=['blue', 'orange'])
    plt.title("Wav2Vec2 Evaluation Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.savefig(os.path.join(eval_folder, "wav2vec_metrics.png"))
    plt.close()

# Compare results
if args.mymodel and args.wav2vec:
    print("\nComparison:")
    print(f"SimpleCTCModel - CER: {cer_score_mymodel:.4f}, WER: {wer_score_mymodel:.4f}")
    print(f"Wav2Vec2       - CER: {cer_score_wav2vec:.4f}, WER: {wer_score_wav2vec:.4f}")
    
    # Save comparison plot
    plt.figure(figsize=(8, 5))
    models = ["SimpleCTCModel", "Wav2Vec2"]
    cer_scores = [cer_score_mymodel, cer_score_wav2vec]
    wer_scores = [wer_score_mymodel, wer_score_wav2vec]
    x = np.arange(len(models))
    width = 0.35
    plt.bar(x - width/2, cer_scores, width, label="CER", color="blue")
    plt.bar(x + width/2, wer_scores, width, label="WER", color="orange")
    plt.xlabel("Models")
    plt.ylabel("Score")
    plt.title("Model Comparison: CER and WER")
    plt.xticks(x, models)
    plt.legend()
    plt.ylim(0, 1)
    plt.savefig(os.path.join(eval_folder, "model_comparison.png"))
    plt.close()