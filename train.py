import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import joblib

# -----------------------------
# Configuration (Fast Training)
# -----------------------------
SEQ_LEN = 24       # sequence length for LSTM
EPOCHS = 1         # 1 epoch for fast training
LR = 0.001
BATCH_SIZE = 16    # smaller batch
HIDDEN_SIZE = 50
USE_GPU = torch.cuda.is_available()
device = torch.device("cuda" if USE_GPU else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("data/energy.csv", sep=";", na_values=["?"])
data.dropna(inplace=True)

# Combine Date and Time into datetime column if present
if 'Date' in data.columns and 'Time' in data.columns:
    data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], dayfirst=True)
    data.drop(['Date', 'Time'], axis=1, inplace=True)

# Use only numeric column
values = data["Global_active_power"].astype(float).values.reshape(-1, 1)

# Scale values
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

# -----------------------------
# Create Sequences
# -----------------------------
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

X_np, y_np = create_sequences(scaled, SEQ_LEN)

# -----------------------------
# Reduce dataset for super fast training
# -----------------------------
X_np = X_np[:1000]
y_np = y_np[:1000]

# Convert to torch tensors
X_tensor = torch.tensor(X_np, dtype=torch.float32)
y_tensor = torch.tensor(y_np, dtype=torch.float32)

# -----------------------------
# DataLoader
# -----------------------------
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# LSTM Model
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMModel().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# Training Loop (Fast)
# -----------------------------
for epoch in range(EPOCHS):
    epoch_loss = 0
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batch_X.size(0)

    epoch_loss /= len(loader.dataset)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.6f}")

# -----------------------------
# Save Model & Scaler
# -----------------------------
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/lstm_model.pt")
print("Model saved to model/lstm_model.pt")

joblib.dump(scaler, "model/scaler.save")
print("Scaler saved to model/scaler.save")
