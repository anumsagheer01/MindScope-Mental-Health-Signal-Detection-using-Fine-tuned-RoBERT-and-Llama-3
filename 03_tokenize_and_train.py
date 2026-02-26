import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
from sklearn.metrics import classification_report

# Load data
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")

# Sample without groupby
samples_train = []
for label in train_df['label'].unique():
    samples_train.append(train_df[train_df['label'] == label].sample(1000, random_state=42))
train_df = pd.concat(samples_train).reset_index(drop=True)

samples_val = []
for label in val_df['label'].unique():
    samples_val.append(val_df[val_df['label'] == label].sample(200, random_state=42))
val_df = pd.concat(samples_val).reset_index(drop=True)

print(f"Train size: {len(train_df)} | Val size: {len(val_df)}")

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

class MentalHealthDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df['body'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = MentalHealthDataset(train_df, tokenizer)
val_dataset = MentalHealthDataset(val_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=5)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3
scheduler = get_scheduler("linear", optimizer=optimizer,
                          num_warmup_steps=0,
                          num_training_steps=num_epochs * len(train_loader))

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % 10 == 0:
            print(f"Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)

    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            preds.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    print(f"\n=== Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f} ===")
    print(classification_report(true_labels, preds,
          target_names=['depression', 'ADHD', 'OCD', 'ptsd', 'aspergers']))

model.save_pretrained("mindscope_model")
tokenizer.save_pretrained("mindscope_model")
print("Model saved!")