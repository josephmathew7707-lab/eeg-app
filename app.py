"""
app.py - Flask backend for EEG Brain Signal Classifier (PyTorch)
No login system - direct access
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
import matplotlib.patches as mpatches

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
os.makedirs("uploads", exist_ok=True)

# ── CNN Model ─────────────────────────────────────────────────────────────────
class CNN_LSTM(nn.Module):
    """CNN-LSTM Hybrid: CNN extracts local features, LSTM learns temporal patterns"""
    def __init__(self, window=512, lstm_hidden=128, lstm_layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1,  32, kernel_size=5, padding=2), nn.BatchNorm1d(32),  nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.25),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64),  nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.25),
            nn.Conv1d(64,128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.25),
        )
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            dropout=0.3, bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64),             nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        cnn_out     = self.cnn(x)
        lstm_in     = cnn_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_in)
        return self.classifier(lstm_out[:, -1, :])

EEG_CNN = CNN_LSTM


MODEL_PATH  = "model/cnn_eeg.pt"
SCALER_PATH = "model/scaler.pkl"
META_PATH   = "model/meta.json"
model, scaler, meta = None, None, {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_artefacts():
    global model, scaler, meta
    if os.path.exists(MODEL_PATH):
        model = EEG_CNN(window=WINDOW).to(device)
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

FS     = 173.61
WINDOW = 512
STEP   = 128
load_artefacts()
FS     = meta.get("fs", 173.61)
WINDOW = meta.get("window", 512)
STEP   = meta.get("step", 128)

# ── Signal helpers ─────────────────────────────────────────────────────────────
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

# ── Abnormality detection ──────────────────────────────────────────────────────
def detect_abnormalities(raw, filtered, probs):
    display_len   = min(1000, len(raw))
    raw_disp      = raw[:display_len]
    filt_disp     = filtered[:display_len]
    mean_val      = float(np.mean(np.abs(raw)))
    max_val       = float(np.max(np.abs(raw)))
    std_val       = float(np.std(raw))
    threshold     = mean_val + 2.5 * std_val
    mean_prob     = float(np.mean(probs))
    is_abnormal   = mean_prob >= 0.5

    stats = {
        "mean_amplitude": round(mean_val, 2),
        "max_amplitude":  round(max_val, 2),
        "std_deviation":  round(std_val, 2),
        "spike_count":    0,
        "hf_ratio":       0.0,
        "threshold":      round(threshold, 2)
    }

    # If CNN says Normal — return empty findings immediately
    if not is_abnormal:
        return [], [], stats

    # Spike detection
    spike_mask = np.abs(raw_disp) > threshold
    spike_regions = []
    in_spike = False
    start = 0
    for i, s in enumerate(spike_mask):
        if s and not in_spike:
            in_spike = True; start = i
        elif not s and in_spike:
            in_spike = False
            if i - start > 2:
                spike_regions.append([max(0, start-5), min(display_len, i+5)])
    if in_spike:
        spike_regions.append([start, display_len])
    merged = []
    for r in spike_regions:
        if merged and r[0] - merged[-1][1] < 20:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    spike_regions = merged

    diff     = np.diff(filt_disp)
    hf_ratio = float(np.sum(np.abs(diff) > np.std(diff) * 2) / len(diff))
    pos_energy = float(np.sum(raw[raw > 0] ** 2))
    neg_energy = float(np.sum(raw[raw < 0] ** 2))
    asymmetry  = abs(pos_energy - neg_energy) / (pos_energy + neg_energy + 1e-8)

    stats["spike_count"] = len(spike_regions)
    stats["hf_ratio"]    = round(hf_ratio * 100, 1)

    findings = []
    if len(spike_regions) > 0:
        count = len(spike_regions)
        findings.append({
            "type": "Sudden brain activity spikes detected",
            "simple": "Your brain signal shows sudden sharp jumps called spikes. This happens when a large group of brain cells fire at the same time — which is not normal. This pattern is commonly seen in people who have epilepsy or seizure disorders.",
            "what_it_means": "These spikes are highlighted in red on the chart above. A normal brain signal should look like gentle waves — but here we can see sharp sudden peaks.",
            "what_to_do": "Please consult a neurologist (brain doctor) as soon as possible. Avoid driving or operating heavy machinery until checked by a doctor.",
            "detail": f"{count} abnormal spike region(s) detected in your brain signal.",
            "technical": f"Epileptiform spike-wave discharges: {count} region(s), threshold={threshold:.1f}uV",
            "severity": "High" if count > 5 else "Moderate",
            "color": "#ff4f6e"
        })
    if max_val > 3 * mean_val:
        findings.append({
            "type": "Unusually strong brain signal bursts",
            "simple": f"Some parts of your brain signal are much stronger than normal — {(max_val/mean_val):.1f}x higher than your average. Think of it like someone suddenly shouting in a quiet room.",
            "what_it_means": "When the signal becomes extremely large, brain cells may be firing too strongly or in an uncontrolled way.",
            "what_to_do": "This needs to be reviewed by a neurologist. It could be related to epilepsy, brain injury, or other neurological conditions.",
            "detail": f"Peak amplitude {max_val:.1f} uV — {(max_val/mean_val):.1f}x above baseline.",
            "technical": f"High-amplitude transients: peak={max_val:.1f}uV, ratio={max_val/mean_val:.1f}x",
            "severity": "High",
            "color": "#ff8c42"
        })
    if hf_ratio > 0.15:
        findings.append({
            "type": "Rapid irregular brain activity",
            "simple": "Your brain signal is changing very quickly and irregularly. Normally brain signals have a smooth steady rhythm — but here the signal is jumping up and down very fast.",
            "what_it_means": "This kind of rapid irregular activity can happen during or just before a seizure.",
            "what_to_do": "Tell your doctor about any recent episodes of shaking, blacking out, or confusion. A full clinical EEG test is recommended.",
            "detail": f"Fast irregular activity detected in {hf_ratio*100:.1f}% of the signal.",
            "technical": f"High-frequency burst activity: HF ratio={hf_ratio*100:.1f}%",
            "severity": "Moderate",
            "color": "#ffd166"
        })
    if asymmetry > 0.3:
        findings.append({
            "type": "Uneven brain signal pattern",
            "simple": "Your brain signal is not balanced — one side is much stronger than the other. A healthy brain signal should be roughly balanced.",
            "what_it_means": "This imbalance can indicate that one part of the brain is more active than it should be.",
            "what_to_do": "Mention this to your doctor. It may need further investigation with a full clinical EEG.",
            "detail": f"Signal imbalance: {asymmetry*100:.1f}% asymmetry index.",
            "technical": f"Signal asymmetry: index={asymmetry*100:.1f}%",
            "severity": "Low",
            "color": "#a0c4ff"
        })
    if len(findings) == 0:
        findings.append({
            "type": "Irregular brain wave pattern",
            "simple": "Your brain signal does not follow the normal wave pattern. Even though the signal strength looks okay, the shape and rhythm of the waves is irregular.",
            "what_it_means": "Our AI detected that the overall pattern of your brain signal is different from normal. This could be an early sign of a neurological condition.",
            "what_to_do": "Please visit a neurologist for a proper clinical EEG test.",
            "detail": "CNN detected abnormal waveform morphology despite normal amplitude range.",
            "technical": f"Diffuse abnormal pattern: mean_prob={mean_prob:.4f}",
            "severity": "Moderate",
            "color": "#ff8c42"
        })

    return spike_regions, findings, stats

# ── Chart ──────────────────────────────────────────────────────────────────────
def plot_signal(raw, filtered, prediction, confidence, spike_regions):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), facecolor="#0d1117")
    accent      = "#00e5a0" if prediction == "Normal" else "#ff4f6e"
    is_abnormal = prediction == "Abnormal"
    display_len = min(1000, len(raw))

    for ax, data, title in zip(
        axes,
        [raw[:display_len], filtered[:display_len]],
        ["Raw EEG Signal", "Filtered Signal (0.5-40 Hz)"]
    ):
        ax.set_facecolor("#161b22")
        ax.plot(data, color=accent, linewidth=0.8, alpha=0.9, zorder=2)

        if is_abnormal and spike_regions:
            for (s, e) in spike_regions:
                ax.axvspan(s, e, color="#ff4f6e", alpha=0.18, zorder=1)
            fs, fe = spike_regions[0]
            mid = (fs + fe) // 2
            if fe <= len(data) and fs < len(data):
                peak_val = float(np.max(np.abs(data[fs:min(fe, len(data))])))
                sign = 1 if data[mid] > 0 else -1
                ax.annotate("Spike", xy=(mid, data[mid]),
                    xytext=(mid+30, data[mid]+sign*peak_val*0.4),
                    arrowprops=dict(arrowstyle="->", color="#ff4f6e", lw=1.2),
                    color="#ff4f6e", fontsize=7, fontweight="bold")

        if is_abnormal:
            thr = float(np.mean(np.abs(raw)) + 2.5 * np.std(raw))
            ax.axhline( thr, color="#ffd166", linewidth=0.6, linestyle="--", alpha=0.5)
            ax.axhline(-thr, color="#ffd166", linewidth=0.6, linestyle="--", alpha=0.5)
            handles = [
                mpatches.Patch(color="#ff4f6e", alpha=0.4, label="Abnormal region"),
                mpatches.Patch(color="#ffd166", alpha=0.7, label="Spike threshold"),
            ]
            ax.legend(handles=handles, loc="upper right", fontsize=6,
                      facecolor="#0d1117", edgecolor="#30363d", labelcolor="#c9d1d9")

        ax.set_title(title, color="#c9d1d9", fontsize=10, pad=6)
        ax.tick_params(colors="#8b949e", labelsize=7)
        ax.set_ylabel("Amplitude (uV)", color="#8b949e", fontsize=7)
        ax.set_xlabel("Sample index",   color="#8b949e", fontsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    spike_text = f"  |  Spikes: {len(spike_regions)}" if is_abnormal else ""
    fig.suptitle(f"Prediction: {prediction}  |  Confidence: {confidence:.1f}%{spike_text}",
                 color=accent, fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=130)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ── Routes ─────────────────────────────────────────────────────────────────────
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
        raw     = np.array([float(v.replace(",", ".")) for l in lines
                            for v in l.split() if v], dtype=np.float32)
    except Exception as e:
        return jsonify({"error": f"Could not parse file: {e}"}), 400
    if len(raw) < WINDOW:
        return jsonify({"error": f"Signal too short (need >= {WINDOW} samples)."}), 400

    filtered = preprocess(raw)
    segs     = segment(filtered)
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

    spike_regions, findings, stats = detect_abnormalities(raw, filtered, probs)
    chart = plot_signal(raw, filtered, label, confidence, spike_regions)

    return jsonify({
        "prediction":    label,
        "confidence":    round(confidence, 2),
        "probability":   round(mean_prob, 4),
        "segments":      int(len(segs)),
        "samples":       int(len(raw)),
        "chart":         chart,
        "model_acc":     meta.get("accuracy", "N/A"),
        "findings":      findings,
        "stats":         stats,
        "spike_regions": spike_regions,
    })

@app.route("/status")
def status():
    return jsonify({
        "model_loaded":   model is not None,
        "model_type":     meta.get("model_type", "CNN-LSTM Hybrid"),
        "model_accuracy": meta.get("accuracy", "N/A"),
        "window": WINDOW,
        "fs":     FS
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)