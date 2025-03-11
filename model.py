import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import joblib
from transformers import Wav2Vec2Processor

# Define a simple Transformer-based CTC model
class SpeechTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SpeechTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=8, dim_feedforward=hidden_size * 4, dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)  # Output vocab size

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        x = self.embedding(x)  # (batch_size, seq_len, hidden_size)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, hidden_size) for Transformer
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, hidden_size)
        x = self.fc(x)  # (batch_size, seq_len, num_classes)
        return x

# Load processor for vocab size and decoding
processor = joblib.load('wav2vec2_processor.pkl')
vocab_size = processor.tokenizer.vocab_size

random_seeds = [102, 203, 304, 405, 506, 607]
batch_size = 16  # Smaller batch size for audio data (memory-intensive)

# Load and create DataLoaders
train_loaders = {}
valid_loaders = {}
test_loaders = {}

for seed in random_seeds:
    X_train_tensor = torch.load(f'X_train_seed{seed}.pt')
    y_train_tensor = torch.load(f'y_train_seed{seed}.pt')
    X_valid_tensor = torch.load(f'X_valid_seed{seed}.pt')
    y_valid_tensor = torch.load(f'y_valid_seed{seed}.pt')
    X_test_tensor = torch.load(f'X_test_seed{seed}.pt')
    y_test_tensor = torch.load(f'y_test_seed{seed}.pt')

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loaders[seed] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loaders[seed] = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loaders[seed] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Seed {seed} - Train batches: {len(train_loaders[seed])}, "
          f"Valid batches: {len(valid_loaders[seed])}, Test batches: {len(test_loaders[seed])}")

# Train and evaluate
num_epochs = 20  # Fewer epochs for demo (speech models are slow)
results = {'ctc_loss': []}

for seed in random_seeds:
    print(f"\nTraining with seed {seed}...")
    torch.manual_seed(seed)
    model = SpeechTransformer(
        input_size=X_train_tensor.shape[-1],  # Feature size from Wav2Vec2
        hidden_size=256,
        num_layers=4,
        num_classes=vocab_size
    )
    criterion = nn.CTCLoss(blank=processor.tokenizer.pad_token_id, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loaders[seed]:
            optimizer.zero_grad()
            outputs = model(inputs)  # (batch_size, seq_len, num_classes)
            outputs = outputs.log_softmax(2)  # CTC expects log probs
            
            input_lengths = torch.full((inputs.size(0),), inputs.size(1), dtype=torch.long)
            target_lengths = torch.tensor([t[t != processor.tokenizer.pad_token_id].size(0) for t in targets], dtype=torch.long)
            
            loss = criterion(outputs.permute(1, 0, 2), targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Seed {seed}, Epoch [{epoch+1}/{num_epochs}], "
                  f"Train CTC Loss: {running_loss/len(train_loaders[seed]):.4f}")

    # Evaluate on test set (simplified: CTC loss only)
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loaders[seed]:
            outputs = model(inputs).log_softmax(2)
            input_lengths = torch.full((inputs.size(0),), inputs.size(1), dtype=torch.long)
            target_lengths = torch.tensor([t[t != processor.tokenizer.pad_token_id].size(0) for t in targets], dtype=torch.long)
            loss = criterion(outputs.permute(1, 0, 2), targets, input_lengths, target_lengths)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loaders[seed])
    results['ctc_loss'].append(avg_test_loss)
    print(f"Seed {seed} - Test CTC Loss: {avg_test_loss:.4f}")

# Average results
avg_ctc_loss = np.mean(results['ctc_loss'])
print(f"\nAverage CTC Loss across 6 seeds: {avg_ctc_loss:.4f}")