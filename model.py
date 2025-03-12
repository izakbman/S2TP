from transformers import AutoProcessor, Wav2Vec2BertForCTC
from datasets import load_dataset
import torch
import torch
from torch.utils.data import Dataset, DataLoader
import os

#dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation", trust_remote_code=True)
#dataset = dataset.sort("id")
#sampling_rate = dataset.features["audio"].sampling_rate

#processor = AutoProcessor.from_pretrained("hf-audio/wav2vec2-bert-CV16-en")
#model = Wav2Vec2BertForCTC.from_pretrained("hf-audio/wav2vec2-bert-CV16-en")

# audio file is decoded on the fly
#inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

class MyDataset(Dataset):
    def __init__(self, input_data, labels, device='cuda'):
        """
        Args:
            input_data (torch.Tensor): Tensor of input features (X_train, X_test, etc.)
            labels (torch.Tensor): Tensor of corresponding labels
            device (str): The device to load data on, default is 'cuda'
        """
        self.input_data = input_data
        self.labels = labels
        self.device = device

    def __len__(self):
        """Returns the size of the dataset"""
        return len(self.input_data)

    def __getitem__(self, idx):
        """Fetches a single sample and moves to the specified device"""
        input_value = self.input_data[idx].to(self.device)
        label = self.labels[idx].to(self.device)
        print(f"Data loaded in dataloaders on device: {self.device}")
        return input_value, label

# Function to load the .pt files
def load_tensor_data(file_path):
    """
    Load tensor data from a .pt file.
    """
    if os.path.exists(file_path):
        return torch.load(file_path, weights_only = True)
    else:
        raise FileNotFoundError(f"File not found: {file_path}")


# Loading tensors from .pt files
X_train = load_tensor_data("librispeech_cache/X_train.pt")
y_train = load_tensor_data("librispeech_cache/y_train.pt")
X_test = load_tensor_data("librispeech_cache/X_test.pt")
y_test = load_tensor_data("librispeech_cache/y_test.pt")
X_valid = load_tensor_data("librispeech_cache/X_valid.pt")
y_valid = load_tensor_data("librispeech_cache/y_valid.pt")

# Choose device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create Dataset objects
train_dataset = MyDataset(X_train, y_train, device=device)
test_dataset = MyDataset(X_test, y_test, device=device)
valid_dataset = MyDataset(X_valid, y_valid, device=device)

# Create DataLoader objects
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)


if __name__ == "__main__":
# Example of iterating through the DataLoader
    for inputs, labels in train_loader:
        # Inputs and labels are already on the GPU
        print(inputs.shape, labels.shape)  # Process the data, for example, by passing it through a model
        break  # Just show the first batch


"""

X_train = torch.load
print(inputs)
with torch.no_grad():
    logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)

# transcribe speech
transcription = processor.batch_decode(predicted_ids)
transcription[0]



inputs["labels"] = processor(text=dataset[0]["text"], return_tensors="pt").input_ids

# compute loss
loss = model(**inputs).loss
round(loss.item(), 2)

"""