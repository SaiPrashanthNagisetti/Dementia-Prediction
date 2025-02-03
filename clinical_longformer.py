import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, LongformerForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Ensure GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model
MODEL_PATH = "longformer-base-4096"  # Change this to your local path if necessary
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = LongformerForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2).to(device)  # Adjust num_labels for your dataset

# Define Dataset Class
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

# Load Dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    texts = data["text"].tolist()
    labels = data["class"].tolist()  # Ensure column names match your dataset
    return texts, labels

# Prepare Data
train_texts, train_labels = load_data("dataset_train_updated.csv")
test_texts, test_labels = load_data("dataset_test_updated.csv")

# Create Dataset Objects
MAX_LEN = 512  # Adjust if needed
BATCH_SIZE = 16

train_dataset = CustomDataset(train_texts, train_labels, tokenizer, MAX_LEN)
test_dataset = CustomDataset(test_texts, test_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define Training Loop
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

def train_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(data_loader)

def eval_model(model, data_loader, device):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return accuracy_score(true_labels, preds)

# Training
EPOCHS = 3
for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    test_accuracy = eval_model(model, test_loader, device)
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the Model
model.save_pretrained("longformer_finetuned")
tokenizer.save_pretrained("longformer_finetuned")

