"""
Train_model.py
--------------
Trains a 1-D CNN on the Bonn University EEG dataset using PyTorch.
Saves model to model/cnn_eeg.pt

DATASET STRUCTURE (inside dataset/):
  Z/  – normal eyes open    → label 0
  O/  – normal eyes closed  → label 0
  N/  – abnormal            → label 1
  F/  – abnormal            → label 1
  S/  – seizure             → label 1
"""

import os, glob, json, pickle
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
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
EPOCHS   = 40
BATCH    = 64
LR       = 1e-3

NORMAL_SETS   = ["Z", "O"]
ABNORMAL_SETS = ["N", "F", "S"]

os.makedirs("model", exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
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

# Reshape → (N, 1, WINDOW) for PyTorch Conv1d
X_train = torch.tensor(X_train[:, np.newaxis, :], dtype=torch.float32)
X_test  = torch.tensor(X_test[:, np.newaxis, :],  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test,  dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=BATCH)

# ── CNN Model ─────────────────────────────────────────────────────────────────
class EEG_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1,  32, kernel_size=5, padding=2), nn.BatchNorm1d(32),  nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.25),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64),  nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.25),
            nn.Conv1d(64,128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Dropout(0.3),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.fc(self.net(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = EEG_CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# ── Training loop ─────────────────────────────────────────────────────────────
print("\nTraining CNN...\n")
best_acc = 0

for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    all_preds = []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)

    acc = accuracy_score(y_test.numpy(), all_preds)
    scheduler.step(1 - acc)

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), MODEL_PATH)

    print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc*100:.2f}%")

print(f"\nBest Accuracy: {best_acc*100:.2f}%")
print(f"Model saved → {MODEL_PATH}")

meta = {
    "accuracy": round(best_acc * 100, 2),
    "window": WINDOW,
    "step": STEP,
    "fs": FS,
    "classes": ["Normal", "Abnormal"],
    "framework": "pytorch"
}
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print("Training complete!")