"""
Train_model.py
--------------
Trains a CNN-LSTM Hybrid model on the Bonn University EEG dataset.

CNN extracts local features from each signal window.
LSTM learns temporal patterns across the sequence.
Together they achieve higher accuracy than CNN alone.

DATASET STRUCTURE (inside dataset/):
  Z/  - normal eyes open    -> label 0
  O/  - normal eyes closed  -> label 0
  N/  - abnormal            -> label 1
  F/  - abnormal            -> label 1
  S/  - seizure             -> label 1
"""

import os, glob, json, pickle
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = "dataset"
MODEL_PATH  = "model/cnn_eeg.pt"
SCALER_PATH = "model/scaler.pkl"
META_PATH   = "model/meta.json"

FS       = 173.61
WINDOW   = 512
STEP     = 128
EPOCHS   = 50
BATCH    = 64
LR       = 1e-3

NORMAL_SETS   = ["Z", "O"]
ABNORMAL_SETS = ["N", "F", "S"]

os.makedirs("model", exist_ok=True)

# ── Signal helpers ────────────────────────────────────────────────────────────
def bandpass(sig, lo=0.5, hi=40.0, fs=FS, order=4):
    nyq = fs / 2
    b, a = butter(order, [lo/nyq, hi/nyq], btype="band")
    return filtfilt(b, a, sig)

def load_txt(path):
    try:
        return np.loadtxt(path)
    except Exception:
        return None

def window_signal(sig, win=WINDOW, step=STEP):
    segs = []
    for s in range(0, len(sig) - win + 1, step):
        segs.append(sig[s:s+win])
    return segs

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading Bonn EEG dataset...")
X, y = [], []

for folder, label in [(s,0) for s in NORMAL_SETS] + [(s,1) for s in ABNORMAL_SETS]:
    path = os.path.join(DATA_DIR, folder)
    if not os.path.isdir(path):
        print(f"  [WARN] Not found: {path} — skipping")
        continue
    files = glob.glob(os.path.join(path, "*.txt"))
    print(f"  {folder}: {len(files)} files, label={label}")
    for f in files:
        sig = load_txt(f)
        if sig is None: continue
        sig = bandpass(sig)
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)
        for seg in window_signal(sig):
            X.append(seg)
            y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)
print(f"\nTotal windows: {len(X)} | Normal: {(y==0).sum()} | Abnormal: {(y==1).sum()}")

# ── Split & scale ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)
print("✔ Scaler saved")

# Reshape for CNN-LSTM: (N, 1, WINDOW)
X_train_t = torch.tensor(X_train[:, np.newaxis, :], dtype=torch.float32)
X_test_t  = torch.tensor(X_test[:, np.newaxis, :],  dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=BATCH)

# ── CNN-LSTM Hybrid Model ─────────────────────────────────────────────────────
class CNN_LSTM(nn.Module):
    """
    CNN-LSTM Hybrid Architecture:
    - CNN layers extract local time-frequency features from the signal
    - LSTM layer learns sequential/temporal patterns across the extracted features
    - Dense layers make the final Normal/Abnormal decision
    """
    def __init__(self, window=512, lstm_hidden=128, lstm_layers=2):
        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv1d(1,  32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
        )

        # Calculate CNN output size
        cnn_out_len = window // 8  # 3 MaxPool1d(2) layers = divide by 8

        # LSTM temporal learner
        self.lstm = nn.LSTM(
            input_size  = 128,        # CNN output channels
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            dropout     = 0.3,
            bidirectional = True      # Bidirectional = looks forward AND backward
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # x shape: (batch, 1, window)

        # CNN feature extraction
        cnn_out = self.cnn(x)
        # cnn_out shape: (batch, 128, window//8)

        # Reshape for LSTM: (batch, seq_len, features)
        lstm_in = cnn_out.permute(0, 2, 1)
        # lstm_in shape: (batch, window//8, 128)

        # LSTM temporal learning
        lstm_out, _ = self.lstm(lstm_in)
        # lstm_out shape: (batch, window//8, hidden*2)

        # Take last timestep output
        last_out = lstm_out[:, -1, :]
        # last_out shape: (batch, hidden*2)

        # Classification
        return self.classifier(last_out)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
print("\nCNN-LSTM Hybrid Architecture:")
print("=" * 50)
print("CNN Block 1: Conv1D(32) + BN + ReLU + MaxPool + Dropout")
print("CNN Block 2: Conv1D(64) + BN + ReLU + MaxPool + Dropout")
print("CNN Block 3: Conv1D(128) + BN + ReLU + MaxPool + Dropout")
print("LSTM: Bidirectional(hidden=128, layers=2) + Dropout")
print("Dense: Linear(256->128) + Linear(128->64) + Linear(64->2)")
print("=" * 50)

model = CNN_LSTM(window=WINDOW).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5)

# ── Training loop ─────────────────────────────────────────────────────────────
print("\nTraining CNN-LSTM Hybrid Model...\n")
best_acc  = 0
best_epoch = 0

for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    all_preds = []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)

    acc = accuracy_score(y_test_t.numpy(), all_preds)
    scheduler.step(1 - acc)

    if acc > best_acc:
        best_acc   = acc
        best_epoch = epoch
        torch.save(model.state_dict(), MODEL_PATH)

    print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc*100:.2f}% {'<-- Best!' if acc == best_acc else ''}")

# ── Final evaluation ──────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Best Accuracy : {best_acc*100:.2f}%")
print(f"Best Epoch    : {best_epoch}/{EPOCHS}")
print(f"Model saved   : {MODEL_PATH}")

# Load best model for final report
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
all_preds = []
with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb.to(device)).argmax(1).cpu().numpy()
        all_preds.extend(preds)

print("\nClassification Report:")
print(classification_report(y_test_t.numpy(), all_preds,
      target_names=["Normal", "Abnormal"]))

print("Confusion Matrix:")
cm = confusion_matrix(y_test_t.numpy(), all_preds)
print(f"  True Normal  correctly identified : {cm[0][0]}")
print(f"  True Abnormal correctly identified: {cm[1][1]}")
print(f"  False positives (Normal as Abnormal): {cm[0][1]}")
print(f"  False negatives (Abnormal as Normal): {cm[1][0]}")

# Save metadata
meta = {
    "accuracy":   round(best_acc * 100, 2),
    "window":     WINDOW,
    "step":       STEP,
    "fs":         FS,
    "classes":    ["Normal", "Abnormal"],
    "framework":  "pytorch",
    "model_type": "CNN-LSTM Hybrid",
    "architecture": "CNN(32,64,128) + BiLSTM(128x2) + Dense(128,64,2)"
}
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n✔ Metadata saved to {META_PATH}")
print("Training complete!")