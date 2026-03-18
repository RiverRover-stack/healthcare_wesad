import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy.signal import butter, filtfilt, welch

# ─── Setup Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(r"d:\WESAD")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
FS = 700  # 700 Hz chest sampling rate

# ─── Helper Functions ─────────────────────────────────────────────────────────

def butter_bandpass(signal, lowcut, highcut, fs=FS, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def butter_lowpass(signal, cutoff, fs=FS, order=4):
    nyq = 0.5 * fs
    freq = cutoff / nyq
    b, a = butter(order, freq, btype='low')
    return filtfilt(b, a, signal)

def z_score_normalize(signal):
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-10)

# ─── Synthetic Signal Generators ─────────────────────────────────────────────

def make_ecg_signal(duration=2.0, fs=FS):
    """Generate a realistic noisy ECG-like signal."""
    t = np.linspace(0, duration, int(duration * fs))
    # Clean cardiac oscillation + harmonics
    clean = (0.8 * np.sin(2 * np.pi * 1.2 * t)
             + 0.3 * np.sin(2 * np.pi * 2.4 * t)
             + 0.1 * np.sin(2 * np.pi * 0.15 * t))  # baseline wander
    # Add wideband noise
    noise = 0.5 * np.random.randn(len(t))
    return t, clean + noise, clean

def make_eda_signal(duration=60.0, fs=FS):
    """Generate EDA signal: slow tonic component + phasic bursts."""
    t = np.linspace(0, duration, int(duration * fs))
    tonic = 2.0 + 0.5 * np.sin(2 * np.pi * 0.02 * t)  # slow drift
    phasic = np.zeros_like(t)
    # SCR bursts at random times
    scr_times = [10, 22, 35, 48, 55]
    for st in scr_times:
        idx = int(st * fs)
        if idx < len(t) - int(3 * fs):
            burst = np.zeros(len(t))
            window = int(3 * fs)
            envelope = np.exp(-np.linspace(0, 4, window))
            burst[idx:idx + window] = 0.4 * envelope
            phasic += burst
    noise = 0.02 * np.random.randn(len(t))
    return t, tonic + phasic + noise, tonic, phasic

def make_two_subjects_signal(duration=30.0, fs=FS):
    """Two subjects with different baselines for normalization demo."""
    t = np.linspace(0, duration, int(duration * fs))
    subj_a = 5.0 + 1.5 * np.sin(2 * np.pi * 0.1 * t) + 0.3 * np.random.randn(len(t))
    subj_b = 12.0 + 0.8 * np.sin(2 * np.pi * 0.15 * t) + 0.2 * np.random.randn(len(t))
    return t, subj_a, subj_b

# ─── Seaborn / Matplotlib Style ──────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
COLORS = sns.color_palette("muted")
C_RAW   = COLORS[3]   # muted red for noisy/raw
C_CLEAN = COLORS[0]   # muted blue for filtered/clean
C_A     = COLORS[0]
C_B     = COLORS[2]
C_BASE  = "#4C7FA3"   # baseline window color
C_STRESS = "#C0504D"  # stress window color
C_BUFF  = "#888888"

# ─── Figure Layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 22))
fig.suptitle(
    "WESAD Pipeline: Data Preprocessing & Feature Engineering\n(Synthetic EDA – Student Demonstration)",
    fontsize=15, fontweight='bold', y=0.99
)

gs = fig.add_gridspec(5, 2, hspace=0.65, wspace=0.35,
                      left=0.07, right=0.97, top=0.95, bottom=0.04)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1  –  Raw vs. Filtered ECG  (Signal Denoising)
# ─────────────────────────────────────────────────────────────────────────────
ax1a = fig.add_subplot(gs[0, 0])
ax1b = fig.add_subplot(gs[0, 1])

t_ecg, noisy_ecg, clean_ecg = make_ecg_signal()
filtered_ecg = butter_bandpass(noisy_ecg, lowcut=0.5, highcut=40.0)

# Downsample for display (plot every 4th sample)
step = 4
t_d = t_ecg[::step]

ax1a.plot(t_d, noisy_ecg[::step], color=C_RAW, linewidth=0.8, label="Raw ECG")
ax1a.set_title("Step 1a – Raw ECG Signal\n(High Noise, Baseline Wander)", fontsize=10, fontweight='bold')
ax1a.set_xlabel("Time (s)")
ax1a.set_ylabel("Amplitude (a.u.)")
ax1a.legend(fontsize=8)
ax1a.text(0.02, 0.92, "[!] Noisy", transform=ax1a.transAxes, color='red', fontsize=8)

ax1b.plot(t_d, filtered_ecg[::step], color=C_CLEAN, linewidth=0.9, label="Filtered ECG (0.5–40 Hz)")
ax1b.set_title("Step 1b – After Butterworth Bandpass Filter\n(SNR Enhanced)", fontsize=10, fontweight='bold')
ax1b.set_xlabel("Time (s)")
ax1b.set_ylabel("Amplitude (a.u.)")
ax1b.legend(fontsize=8)
ax1b.text(0.02, 0.92, "[OK] Clean", transform=ax1b.transAxes, color='green', fontsize=8)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2  –  Z-Score Normalization  (Both Subjects: Before & After)
# ─────────────────────────────────────────────────────────────────────────────
ax2a = fig.add_subplot(gs[1, 0])
ax2b = fig.add_subplot(gs[1, 1])

t_sub, subj_a, subj_b = make_two_subjects_signal()
step2 = 10
t_s = t_sub[::step2]

ax2a.plot(t_s, subj_a[::step2], color=C_A, linewidth=0.9, label="Subject A (baseline ~5)")
ax2a.plot(t_s, subj_b[::step2], color=C_B, linewidth=0.9, label="Subject B (baseline ~12)")
ax2a.set_title("Step 2a – Before Normalization\n(Inter-Subject Bias / Domain Shift)", fontsize=10, fontweight='bold')
ax2a.set_xlabel("Time (s)")
ax2a.set_ylabel("Raw Signal Value")
ax2a.legend(fontsize=8)

norm_a = z_score_normalize(subj_a)
norm_b = z_score_normalize(subj_b)

ax2b.plot(t_s, norm_a[::step2], color=C_A, linewidth=0.9, label="Subject A (normalized)")
ax2b.plot(t_s, norm_b[::step2], color=C_B, linewidth=0.9, label="Subject B (normalized)")
ax2b.axhline(0, color='gray', linestyle='--', linewidth=0.8, label="Global Mean = 0")
ax2b.set_title("Step 2b – After Z-Score Normalization\n(Domain Shift Removed)", fontsize=10, fontweight='bold')
ax2b.set_xlabel("Time (s)")
ax2b.set_ylabel("Z-Score")
ax2b.legend(fontsize=8)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3  –  Windowing and Purity Labels
# ─────────────────────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2, :])

total_sec = 120
labels_raw = np.zeros(total_sec)
labels_raw[0:35]   = 0  # Baseline
labels_raw[35:40]  = -1  # Transition (buffer)
labels_raw[40:80]  = 1  # Stress
labels_raw[80:85]  = -1  # Transition
labels_raw[85:120] = 0  # Baseline

t_label = np.arange(total_sec)
window_size = 20   # 20 sec display windows (represents 60s in reality)
step_size   = 10
win_starts = np.arange(0, total_sec - window_size, step_size)

# Draw colored background for states
for i, t in enumerate(t_label):
    color = C_BASE if labels_raw[i] == 0 else (C_STRESS if labels_raw[i] == 1 else C_BUFF)
    ax3.axvspan(t, t + 1, color=color, alpha=0.25)

# Draw window boundaries
for i, ws in enumerate(win_starts):
    we = ws + window_size
    window_labels = labels_raw[ws:we]
    valid = window_labels[window_labels != -1]
    if len(valid) == 0:
        continue
    purity = np.sum(valid == valid[0]) / len(valid)
    ec = 'black' if purity >= 0.8 else 'orange'
    ax3.add_patch(mpatches.FancyBboxPatch(
        (ws, 0.1), window_size, 0.8,
        boxstyle="round,pad=0.5", linewidth=1.5,
        edgecolor=ec, facecolor='none'
    ))
    label_txt = "OK" if purity >= 0.8 else "X"
    ax3.text(ws + window_size / 2, 0.92, label_txt, ha='center', fontsize=8,
             color='green' if purity >= 0.8 else 'orange')

# Transition buffer hatch
for i, t in enumerate(t_label):
    if labels_raw[i] == -1:
        ax3.axvspan(t, t + 1, color=C_BUFF, alpha=0.55, hatch='///')

ax3.set_xlim(0, total_sec)
ax3.set_ylim(-0.05, 1.1)
ax3.set_xlabel("Time (seconds)")
ax3.set_yticks([])
ax3.set_title(
    "Step 3 - Windowing & Purity Check  (Window=60s, Overlap=50%, Buffer=5s, Purity>=80%)\n"
    "Blue=Baseline  |  Red=Stress  |  Gray///=Transition Buffer  |  OK=Accepted Window  |  X=Rejected",
    fontsize=10, fontweight='bold'
)
baseline_patch = mpatches.Patch(color=C_BASE, alpha=0.4, label='Baseline')
stress_patch   = mpatches.Patch(color=C_STRESS, alpha=0.4, label='Stress')
buffer_patch   = mpatches.Patch(color=C_BUFF, alpha=0.5, label='Transition Buffer')
ax3.legend(handles=[baseline_patch, stress_patch, buffer_patch], loc='lower right', fontsize=8)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4  –  Feature Engineering  (4 Feature Categories)
# ─────────────────────────────────────────────────────────────────────────────

# --- 4a  Statistical: Histogram of EDA Window Means ---
ax4a = fig.add_subplot(gs[3, 0])

n_windows = 80
baseline_means = np.random.normal(loc=-0.2, scale=0.4, size=n_windows)
stress_means   = np.random.normal(loc=0.6,  scale=0.5, size=n_windows)

ax4a.hist(baseline_means, bins=15, color=C_BASE, alpha=0.7, label='Baseline',  edgecolor='white')
ax4a.hist(stress_means,   bins=15, color=C_STRESS, alpha=0.7, label='Stress', edgecolor='white')
ax4a.axvline(np.mean(baseline_means), color=C_BASE,   linestyle='--', linewidth=1.5)
ax4a.axvline(np.mean(stress_means),   color=C_STRESS, linestyle='--', linewidth=1.5)
ax4a.set_title("Feature Category 1: Statistical\n(EDA Window Mean per Class)", fontsize=9, fontweight='bold')
ax4a.set_xlabel("Mean EDA (Z-Score)")
ax4a.set_ylabel("Count")
ax4a.legend(fontsize=8)

# --- 4b  Temporal: First Derivative ---
ax4b = fig.add_subplot(gs[3, 1])

dur_t = 5.0
t_tmp = np.linspace(0, dur_t, int(dur_t * 100))
sig_b = 0.3 * np.sin(2 * np.pi * 0.5 * t_tmp) + 0.05 * np.random.randn(len(t_tmp))
sig_s = 0.8 * np.sin(2 * np.pi * 0.8 * t_tmp) + 0.1 * np.random.randn(len(t_tmp))
deriv_b = np.diff(sig_b) * 100
deriv_s = np.diff(sig_s) * 100

ax4b.plot(t_tmp[1:], deriv_b, color=C_BASE,   linewidth=0.9, alpha=0.8, label='Baseline Slope')
ax4b.plot(t_tmp[1:], deriv_s, color=C_STRESS, linewidth=0.9, alpha=0.8, label='Stress Slope')
ax4b.axhline(0, color='gray', linestyle='--', linewidth=0.7)
ax4b.set_title("Feature Category 2: Temporal\n(First Derivative / Slope – Rate of Change)", fontsize=9, fontweight='bold')
ax4b.set_xlabel("Time (s)")
ax4b.set_ylabel("dSignal/dt")
ax4b.legend(fontsize=8)

# --- 4c  Frequency: PSD (Welch) ---
ax4c = fig.add_subplot(gs[4, 0])

sig_base_freq = np.random.randn(4096) * 0.5 + 0.2 * np.sin(2 * np.pi * 0.1 * np.arange(4096) / 100)
sig_stress_freq = np.random.randn(4096) * 0.9 + 0.5 * np.sin(2 * np.pi * 0.3 * np.arange(4096) / 100)

f_b, psd_b = welch(sig_base_freq, fs=100, nperseg=256)
f_s, psd_s = welch(sig_stress_freq, fs=100, nperseg=256)

ax4c.semilogy(f_b, psd_b, color=C_BASE,   linewidth=1.2, label='Baseline PSD')
ax4c.semilogy(f_s, psd_s, color=C_STRESS, linewidth=1.2, label='Stress PSD')
ax4c.axvspan(0, 0.04, alpha=0.1, color='gray', label='VLF Band')
ax4c.axvspan(0.04, 0.15, alpha=0.1, color='blue', label='LF Band')
ax4c.axvspan(0.15, 0.4, alpha=0.1, color='green', label='HF Band')
ax4c.set_xlim(0, 0.5)
ax4c.set_title("Feature Category 3: Frequency\n(Power Spectral Density – Welch Method)", fontsize=9, fontweight='bold')
ax4c.set_xlabel("Frequency (Hz)")
ax4c.set_ylabel("Power Spectral Density")
ax4c.legend(fontsize=7, loc='upper right')

# --- 4d  EDA-Specific: Tonic + Phasic + SCR Peaks ---
ax4d = fig.add_subplot(gs[4, 1])

t_eda, eda_raw, eda_tonic, eda_phasic = make_eda_signal(duration=60.0)
step_e = 14  # downsample for display
t_e = t_eda[::step_e]

ax4d.plot(t_e, (eda_tonic + eda_phasic)[::step_e], color='steelblue',
          linewidth=1.0, label='EDA Signal (Tonic+Phasic)', alpha=0.9)
ax4d.plot(t_e, eda_tonic[::step_e], color='darkorange',
          linewidth=1.4, linestyle='--', label='Tonic Component (SCL)')
ax4d.fill_between(t_e, eda_tonic[::step_e],
                  (eda_tonic + eda_phasic)[::step_e],
                  color='lightblue', alpha=0.4, label='Phasic Component (SCR)')

# Mark SCR peaks
scr_peak_times = [10, 22, 35, 48, 55]
for st in scr_peak_times:
    idx = st * FS
    if idx < len(eda_raw):
        peak_val = (eda_tonic + eda_phasic)[idx]
        ax4d.plot(st, peak_val, 'rv', markersize=8, zorder=5)
        ax4d.text(st, peak_val + 0.05, 'SCR', ha='center', fontsize=6.5, color='red')

ax4d.set_title("Feature Category 4: EDA-Specific\n(Tonic/Phasic Decomposition + SCR Peaks)", fontsize=9, fontweight='bold')
ax4d.set_xlabel("Time (s)")
ax4d.set_ylabel("EDA (μS)")
ax4d.legend(fontsize=7)

# ─────────────────────────────────────────────────────────────────────────────
# Save & Display
# ─────────────────────────────────────────────────────────────────────────────
output_path = OUTPUT_DIR / "eda_visualization.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n[DONE] Plot saved to: {output_path}")
plt.show()
