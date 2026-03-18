"""
app.py  –  Flask backend for EEG Brain Signal Classifier (PyTorch version)
"""

import os, json, pickle, io, base64
import numpy as np
from scipy.signal import butter, filtfilt
from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

os.makedirs("uploads", exist_ok=True)

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

MODEL_PATH  = "model/cnn_eeg.pt"
SCALER_PATH = "model/scaler.pkl"
META_PATH   = "model/meta.json"

model, scaler, meta = None, None, {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_artefacts():
    global model, scaler, meta
    if os.path.exists(MODEL_PATH):
        model = EEG_CNN().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("✔ CNN model loaded (PyTorch)")
    else:
        print("⚠ Model not found — run Train_model.py first")
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print("✔ Scaler loaded")
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)

load_artefacts()

FS     = meta.get("fs", 173.61)
WINDOW = meta.get("window", 512)
STEP   = meta.get("step", 128)

def bandpass(sig, lo=0.5, hi=40.0, fs=FS, order=4):
    nyq = fs / 2
    b, a = butter(order, [lo/nyq, hi/nyq], btype="band")
    return filtfilt(b, a, sig)

def preprocess(raw):
    sig = bandpass(raw)
    sig = (sig - sig.mean()) / (sig.std() + 1e-8)
    return sig

def segment(sig):
    segs = []
    for s in range(0, len(sig) - WINDOW + 1, STEP):
        segs.append(sig[s:s+WINDOW])
    return np.array(segs, dtype=np.float32) if segs else None

def plot_signal(raw, filtered, prediction, confidence):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), facecolor="#0d1117")
    accent = "#00e5a0" if prediction == "Normal" else "#ff4f6e"
    for ax, data, title in zip(axes, [raw[:1000], filtered[:1000]], ["Raw EEG Signal", "Filtered Signal (0.5-40 Hz)"]):
        ax.set_facecolor("#161b22")
        ax.plot(data, color=accent, linewidth=0.8, alpha=0.9)
        ax.set_title(title, color="#c9d1d9", fontsize=10, pad=6)
        ax.tick_params(colors="#8b949e", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
    fig.suptitle(f"Prediction: {prediction}  ({confidence:.1f}% confidence)", color=accent, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

@app.route("/")
def index():
    return render_template("index.html", model_ready=(model is not None))

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded. Run Train_model.py first."}), 503
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400
    allowed = {".txt", ".csv", ".eeg", ".dat"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported file type '{ext}'."}), 400
    try:
        content = file.read().decode("utf-8", errors="ignore")
        lines   = [l.strip() for l in content.splitlines() if l.strip()]
        raw     = np.array([float(v.replace(",", ".")) for l in lines for v in l.split() if v], dtype=np.float32)
    except Exception as e:
        return jsonify({"error": f"Could not parse file: {e}"}), 400
    if len(raw) < WINDOW:
        return jsonify({"error": f"Signal too short (need >= {WINDOW} samples)."}), 400
    filtered = preprocess(raw)
    segs = segment(filtered)
    if segs is None:
        return jsonify({"error": "Segmentation failed."}), 500
    flat = scaler.transform(segs)
    flat = torch.tensor(flat[:, np.newaxis, :], dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(flat)
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    mean_prob  = float(probs.mean())
    label      = "Abnormal" if mean_prob >= 0.5 else "Normal"
    confidence = mean_prob * 100 if label == "Abnormal" else (1 - mean_prob) * 100
    chart = plot_signal(raw, filtered, label, confidence)
    return jsonify({
        "prediction":  label,
        "confidence":  round(confidence, 2),
        "probability": round(mean_prob, 4),
        "segments":    int(len(segs)),
        "samples":     int(len(raw)),
        "chart":       chart,
        "model_acc":   meta.get("accuracy", "N/A")
    })

@app.route("/status")
def status():
    return jsonify({"model_loaded": model is not None, "model_accuracy": meta.get("accuracy", "N/A"), "window": WINDOW, "fs": FS})

if __name__ == "__main__":
    import os
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port, debug=False)