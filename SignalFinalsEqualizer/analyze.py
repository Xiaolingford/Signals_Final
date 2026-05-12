"""
analyze.py
==========
Generates the report figures and metrics for the voice denoiser demo.

Outputs:
  filter_response.png   — magnitude/phase of notch cascade + low-pass, pole-zero
  ss_analysis.png       — spectral subtraction: noise PSD, before/after spectra,
                          spectrograms
  demo_compare.png      — before / after spectrograms for a test WAV
  metrics.txt           — SNR / hum / noise-floor numbers for each filter stage

Usage:
    python analyze.py                                    # default file
    python analyze.py path/to/file.wav [noise_only.wav]
    python analyze.py test_voice_snr15_fan_hum.wav noise_fan_only.wav

Optional metrics:
  If `pesq` and `pystoi` are installed AND a clean reference is available at
  `test_voice_clean.wav`, the report adds PESQ (quality) and STOI
  (intelligibility) scores.  Install with:
      pip install pesq pystoi
"""

import os
import sys
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Optional speech metrics (graceful fallback if not installed)
try:
    from pesq import pesq
    _HAS_PESQ = True
except ImportError:
    _HAS_PESQ = False
try:
    from pystoi import stoi
    _HAS_STOI = True
except ImportError:
    _HAS_STOI = False

# ---------------------------------------------------------------------------
# Mirror config from equalizer.py
# ---------------------------------------------------------------------------
FS          = 16000
NOTCH_FREQS = [60.0, 120.0, 180.0]
NOTCH_Q     = 30.0
LP_CUTOFF   = 7000.0
LP_TAPS     = 101
SS_ALPHA    = 2.0
SS_BETA     = 0.02
SS_SMOOTH   = 0.7
SPEC_NFFT   = 1024
HOP         = SPEC_NFFT // 2

# Build notch cascade and low-pass
NOTCH_STAGES = []
for f0 in NOTCH_FREQS:
    if f0 < FS / 2 - 1.0:
        b, a = signal.iirnotch(f0, NOTCH_Q, fs=FS)
        NOTCH_STAGES.append((b, a))

LP_B = signal.firwin(LP_TAPS, LP_CUTOFF, fs=FS, window='hamming')
LP_A = np.array([1.0])

WIN = np.sqrt(np.hanning(SPEC_NFFT)).astype(np.float32)


# ---------------------------------------------------------------------------
# Offline spectral subtractor (Wiener-style — mirrors equalizer.SpectralSubtractor)
# ---------------------------------------------------------------------------
def spectral_subtract_offline(
        x_speech: np.ndarray,
        x_noise: np.ndarray,
        alpha: float = SS_ALPHA,
        beta: float  = SS_BETA,
        smooth: float = SS_SMOOTH,
) -> np.ndarray:
    """
    Offline Wiener-style spectral subtraction with temporal smoothing of
    the gain.  Matches the live SpectralSubtractor in equalizer.py.
    """
    def stft_frames(x):
        n = len(x)
        for start in range(0, n - SPEC_NFFT + 1, HOP):
            frame = x[start:start + SPEC_NFFT] * WIN
            yield np.fft.rfft(frame)

    # 1. Estimate noise magnitude spectrum
    noise_mags = [np.abs(X) for X in stft_frames(x_noise.astype(np.float32))]
    if not noise_mags:
        return x_speech.copy()
    noise_mag = np.mean(noise_mags, axis=0).astype(np.float32)
    noise_pow = alpha * (noise_mag ** 2) + 1e-12

    # 2. Process speech frame by frame (overlap-add) with smoothed Wiener gain
    n         = len(x_speech)
    out_buf   = np.zeros(n + SPEC_NFFT, dtype=np.float32)
    gain_prev = np.ones(SPEC_NFFT // 2 + 1, dtype=np.float32)

    for start in range(0, n - SPEC_NFFT + 1, HOP):
        frame = x_speech[start:start + SPEC_NFFT].astype(np.float32) * WIN
        X     = np.fft.rfft(frame)
        mag   = np.abs(X)
        phase = np.angle(X)

        snr_post = (mag ** 2) / noise_pow
        gain_raw = snr_post / (1.0 + snr_post)
        gain     = smooth * gain_prev + (1 - smooth) * gain_raw
        gain_prev = gain
        gain     = np.maximum(gain, beta)

        mag_clean = mag * gain
        X_clean   = mag_clean * np.exp(1j * phase)
        y_frame   = np.fft.irfft(X_clean).real.astype(np.float32) * WIN

        out_buf[start:start + SPEC_NFFT] += y_frame

    return out_buf[:n]


# ---------------------------------------------------------------------------
# Offline pass of all three filter stages
# ---------------------------------------------------------------------------
def process_offline(x: np.ndarray,
                    noise: np.ndarray | None = None,
                    use_notch: bool = True,
                    use_lowpass: bool = True,
                    use_ss: bool = True) -> np.ndarray:
    """Apply filters in the same order as equalizer.py's `process()`."""
    y = x.astype(np.float32)

    if use_ss and noise is not None:
        y = spectral_subtract_offline(y, noise.astype(np.float32))

    if use_notch:
        for b, a in NOTCH_STAGES:
            zi = signal.lfilter_zi(b, a) * y[0]
            y, _ = signal.lfilter(b, a, y, zi=zi)

    if use_lowpass:
        zi = signal.lfilter_zi(LP_B, LP_A) * y[0]
        y, _ = signal.lfilter(LP_B, LP_A, y, zi=zi)

    return y


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def power_at(x: np.ndarray, freq: float, fs: int = FS, bw: float = 1.0) -> float:
    w = np.hanning(len(x))
    X = np.abs(np.fft.rfft(x * w)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / fs)
    return 10 * np.log10(X[(f >= freq - bw) & (f <= freq + bw)].sum() + 1e-12)


def speech_band_snr_db(x: np.ndarray, fs: int = FS) -> float:
    """Crude SNR: speech band (200-3000 Hz) vs out-of-band (> 4000 Hz)."""
    X   = np.abs(np.fft.rfft(x)) ** 2
    f   = np.fft.rfftfreq(len(x), 1 / fs)
    sig = X[(f >= 200) & (f <= 3000)].sum() + 1e-12
    noi = X[f > 4000].sum() + 1e-12
    return 10 * np.log10(sig / noi)


def _align_to_reference(reference: np.ndarray, signal_to_align: np.ndarray,
                        max_lag: int = 2000) -> np.ndarray:
    """
    Time-align `signal_to_align` to `reference` via cross-correlation.
    Filters (especially IIR notches and the STFT pipeline) introduce small
    group delays; without alignment, segmental SNR penalizes that delay as
    if it were noise.  Returns the shifted signal, same length as input.
    """
    n = min(len(reference), len(signal_to_align), 8 * max_lag)
    a = reference[:n] - np.mean(reference[:n])
    b = signal_to_align[:n] - np.mean(signal_to_align[:n])
    corr = np.correlate(a, b, mode='full')
    lag = np.argmax(corr) - (len(b) - 1)   # positive = b lags a (shift left)
    if abs(lag) > max_lag:
        return signal_to_align  # suspicious; refuse to shift
    if lag > 0:
        out = np.concatenate([signal_to_align[lag:],
                              np.zeros(lag, dtype=signal_to_align.dtype)])
    elif lag < 0:
        out = np.concatenate([np.zeros(-lag, dtype=signal_to_align.dtype),
                              signal_to_align[:lag]])
    else:
        out = signal_to_align
    return out


def segmental_snr_db(clean: np.ndarray, processed: np.ndarray,
                     fs: int = FS, frame_ms: float = 25.0,
                     align: bool = True) -> float:
    """
    Segmental SNR: SNR computed per frame, then averaged.
    Captures non-stationary speech better than full-signal SNR.
    If `align`, the processed signal is shifted to match the clean reference
    via cross-correlation first (compensates for filter group delay).
    """
    n = min(len(clean), len(processed))
    clean = clean[:n].astype(np.float64)
    processed = processed[:n].astype(np.float64)
    if align:
        processed = _align_to_reference(clean, processed)
    frame = int(fs * frame_ms / 1000)
    snrs = []
    for i in range(0, n - frame, frame):
        c = clean[i:i + frame]
        p = processed[i:i + frame]
        sig_e = np.sum(c ** 2) + 1e-12
        err_e = np.sum((c - p) ** 2) + 1e-12
        # Only count frames with speech energy (skip silence)
        if sig_e > 1e-6:
            snrs.append(10 * np.log10(sig_e / err_e))
    if not snrs:
        return float('nan')
    # Clip to a sensible range (PESQ-style)
    snrs = np.clip(snrs, -10, 35)
    return float(np.mean(snrs))


def noise_floor_percentile_db(x: np.ndarray, fs: int = FS,
                               frame_ms: float = 25.0,
                               percentile: int = 10) -> float:
    """
    Estimate the noise floor as the 10th-percentile frame energy (dB).
    This isolates quiet/pause frames, which approximate noise-only.
    Better than averaging over the speech band (which mixes speech in).
    """
    frame = int(fs * frame_ms / 1000)
    energies = []
    for i in range(0, len(x) - frame, frame):
        e = np.mean(x[i:i + frame] ** 2)
        energies.append(e)
    if not energies:
        return float('nan')
    return 10 * np.log10(np.percentile(energies, percentile) + 1e-12)


# ---------------------------------------------------------------------------
# Figure 1 — Filter frequency responses
# ---------------------------------------------------------------------------
def plot_filter_response(outpath: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Filter Frequency Responses', fontsize=14)

    # Cascaded notch magnitude
    ax = axes[0, 0]
    # Combine cascade: multiply frequency responses
    w_total = None
    h_total = None
    for b, a in NOTCH_STAGES:
        w, h = signal.freqz(b, a, worN=8192, fs=FS)
        if h_total is None:
            w_total, h_total = w, h
        else:
            h_total = h_total * h
    ax.plot(w_total, 20 * np.log10(np.abs(h_total) + 1e-12))
    ax.set_title(f'IIR Notch Cascade @ {NOTCH_FREQS} Hz  (Q = {NOTCH_Q:g})')
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('|H(f)| (dB)')
    ax.set_xlim(0, 300); ax.set_ylim(-60, 5); ax.grid(alpha=0.3)
    for f0 in NOTCH_FREQS:
        ax.axvline(f0, color='red', ls='--', alpha=0.4)

    # Pole-zero plot of first notch (representative)
    ax = axes[0, 1]
    b, a = NOTCH_STAGES[0]
    z, p, _k = signal.tf2zpk(b, a)
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color='gray', lw=0.7)
    ax.scatter(z.real, z.imag, marker='o', s=80, facecolors='none',
               edgecolors='C0', label='zeros')
    ax.scatter(p.real, p.imag, marker='x', s=80, color='C3', label='poles')
    ax.set_title(f'Notch pole-zero plot @ {NOTCH_FREQS[0]:g} Hz (z-plane)')
    ax.set_xlabel('Re'); ax.set_ylabel('Im')
    ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend()

    # Low-pass magnitude
    w, h = signal.freqz(LP_B, LP_A, worN=8192, fs=FS)
    ax = axes[1, 0]
    ax.plot(w, 20 * np.log10(np.abs(h) + 1e-12))
    ax.set_title(f'FIR Low-pass (Hamming, {LP_TAPS} taps)  cutoff = {LP_CUTOFF:g} Hz')
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('|H(f)| (dB)')
    ax.set_xlim(0, FS / 2); ax.set_ylim(-100, 5); ax.grid(alpha=0.3)
    ax.axvline(LP_CUTOFF, color='green', ls='--', alpha=0.5, label=f'{LP_CUTOFF:g} Hz')
    ax.legend(loc='lower left')

    # FIR impulse response
    ax = axes[1, 1]
    ax.stem(np.arange(LP_TAPS) - LP_TAPS // 2, LP_B, basefmt=' ', markerfmt='.')
    ax.set_title('FIR impulse response h[n]  (windowed sinc)')
    ax.set_xlabel('n'); ax.set_ylabel('h[n]'); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f'  wrote {outpath}')


# ---------------------------------------------------------------------------
# Figure 2 — Spectral subtraction analysis
# ---------------------------------------------------------------------------
def plot_ss_analysis(x_noisy: np.ndarray, x_noise: np.ndarray, outpath: str):
    x_clean = spectral_subtract_offline(x_noisy, x_noise)
    f_axis  = np.fft.rfftfreq(SPEC_NFFT, 1 / FS)

    # Noise PSD estimate
    noise_mags = []
    for start in range(0, len(x_noise) - SPEC_NFFT + 1, HOP):
        frame = x_noise[start:start + SPEC_NFFT].astype(np.float32) * WIN
        noise_mags.append(np.abs(np.fft.rfft(frame)))
    noise_mag_mean = np.mean(noise_mags, axis=0) if noise_mags else np.zeros(len(f_axis))

    def mean_spec(x):
        mags = []
        for start in range(0, len(x) - SPEC_NFFT + 1, HOP):
            frame = x[start:start + SPEC_NFFT].astype(np.float32) * WIN
            mags.append(np.abs(np.fft.rfft(frame)))
        return np.mean(mags, axis=0) if mags else np.zeros(len(f_axis))

    ms_noisy = mean_spec(x_noisy)
    ms_clean = mean_spec(x_clean)

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f'Spectral Subtraction Analysis  (α={SS_ALPHA}, β={SS_BETA}, '
        f'smooth={SS_SMOOTH})',
        fontsize=13,
    )
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    # TL — Noise PSD
    ax = fig.add_subplot(gs[0, 0])
    ax.semilogy(f_axis, noise_mag_mean ** 2 + 1e-20, color='orange')
    ax.set_title('Estimated noise PSD (from calibration clip)')
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Power (linear)')
    ax.set_xlim(0, FS / 2); ax.grid(alpha=0.3, which='both')
    for harm in [150, 300, 450]:
        ax.axvline(harm, color='red', ls=':', alpha=0.5, lw=0.8)
    ax.text(155, noise_mag_mean.max() ** 2 * 0.6,
            'fan harmonics', color='red', fontsize=8, rotation=90, va='top')

    # TR — Before/After mean spectra
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(f_axis, 20 * np.log10(ms_noisy + 1e-12), label='Noisy input',  lw=1.2)
    ax.plot(f_axis, 20 * np.log10(ms_clean + 1e-12), label='After SS',     lw=1.2)
    ax.plot(f_axis, 20 * np.log10(noise_mag_mean * SS_BETA + 1e-12),
            '--', color='gray', lw=0.8, label=f'Floor (β={SS_BETA})')
    ax.set_title('Mean magnitude spectrum: before vs. after')
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('|X(f)| (dB)')
    ax.set_xlim(0, FS / 2); ax.set_ylim(-100, 10); ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # BL — Spectrogram before
    ax = fig.add_subplot(gs[1, 0])
    f_s, t_s, Sxx = signal.spectrogram(
        x_noisy, fs=FS, nperseg=SPEC_NFFT, noverlap=HOP, scaling='spectrum'
    )
    Sdb = 10 * np.log10(Sxx + 1e-12)
    ax.pcolormesh(t_s, f_s, Sdb, vmin=-80, vmax=-10, cmap='magma', shading='auto')
    ax.set_title('BEFORE spectral subtraction')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Frequency (Hz)')
    for f0 in NOTCH_FREQS + [150.0, 300.0]:
        ax.axhline(f0, color='cyan', ls=':', alpha=0.4, lw=0.6)

    # BR — Spectrogram after
    ax = fig.add_subplot(gs[1, 1])
    f_s, t_s, Sxx = signal.spectrogram(
        x_clean, fs=FS, nperseg=SPEC_NFFT, noverlap=HOP, scaling='spectrum'
    )
    Sdb = 10 * np.log10(Sxx + 1e-12)
    ax.pcolormesh(t_s, f_s, Sdb, vmin=-80, vmax=-10, cmap='magma', shading='auto')
    ax.set_title('AFTER spectral subtraction')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Frequency (Hz)')

    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f'  wrote {outpath}')


# ---------------------------------------------------------------------------
# Figure 3 — Before / after comparison (spectrogram)
# ---------------------------------------------------------------------------
def plot_demo_compare(wav_path: str, noise: np.ndarray | None, outpath: str):
    fs_in, x = wavfile.read(wav_path)
    x = x.astype(np.float32) / 32767.0
    if x.ndim > 1:
        x = x.mean(axis=1)
    if fs_in != FS:
        x = signal.resample_poly(x, FS, fs_in)

    y = process_offline(x, noise, use_ss=(noise is not None))

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    nperseg = 512
    for ax, sig_, title in [
        (axes[0], x, 'BEFORE  (input)'),
        (axes[1], y, 'AFTER   (spectral sub + notch cascade + low-pass)'),
    ]:
        f, t, Sxx = signal.spectrogram(sig_, fs=FS, nperseg=nperseg,
                                       noverlap=nperseg // 2, scaling='spectrum')
        Sdb = 10 * np.log10(Sxx + 1e-12)
        ax.pcolormesh(t, f, Sdb, vmin=-80, vmax=-10, cmap='magma', shading='auto')
        for f0 in NOTCH_FREQS:
            ax.axhline(f0, color='cyan', ls='--', alpha=0.4, lw=0.6)
        ax.axhline(LP_CUTOFF, color='lime', ls='--', alpha=0.6, lw=0.8)
        ax.set_ylabel('Frequency (Hz)'); ax.set_title(title)
    axes[1].set_xlabel('Time (s)')
    fig.suptitle(f'Spectrogram: {os.path.basename(wav_path)}')
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f'  wrote {outpath}')
    return x, y


# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------
def write_metrics(x: np.ndarray,
                  noise: np.ndarray | None,
                  wav_path: str,
                  outpath: str,
                  clean_ref: np.ndarray | None = None):
    # Stage outputs
    y_ss_only  = process_offline(x, noise, use_ss=True,  use_notch=False, use_lowpass=False) \
                 if noise is not None else x
    y_notch    = process_offline(x, None,  use_ss=False, use_notch=True,  use_lowpass=False)
    y_full     = process_offline(x, noise, use_ss=(noise is not None),
                                 use_notch=True, use_lowpass=True)

    def row(label, b, a):
        return (f"{'  ' + label:<32}{b:>12.1f}{a:>12.1f}{a - b:>+10.1f}")

    header = f"{'Metric':<34}{'RAW (dB)':>12}{'AFTER (dB)':>12}{'Δ (dB)':>10}"
    sep    = '-' * 68

    lines = [
        f'Metrics for: {wav_path}',
        f'Sample rate : {FS} Hz',
        f'Length      : {len(x) / FS:.2f} s',
        '',
        '=== Stage 1: Spectral Subtraction (broadband fan noise) ===',
        header, sep,
        row('Speech-band SNR (200-3kHz)', speech_band_snr_db(x),     speech_band_snr_db(y_ss_only)),
        row('Noise floor (10th pctile)',  noise_floor_percentile_db(x),
                                          noise_floor_percentile_db(y_ss_only)),
        row('Power @ 150 Hz (fan BPF)',   power_at(x, 150.0),        power_at(y_ss_only, 150.0)),
        row('Power @ 300 Hz (fan 2H)',    power_at(x, 300.0),        power_at(y_ss_only, 300.0)),
        '',
        '=== Stage 2: IIR Notch Cascade (60/120/180 Hz hum) ===',
        header, sep,
        row('Power @ 60 Hz (hum)',        power_at(x, 60.0),         power_at(y_notch, 60.0)),
        row('Power @ 120 Hz (2nd harm)',  power_at(x, 120.0),        power_at(y_notch, 120.0)),
        row('Power @ 180 Hz (3rd harm)',  power_at(x, 180.0),        power_at(y_notch, 180.0)),
        row('Power @ 250 Hz (control)',   power_at(x, 250.0),        power_at(y_notch, 250.0)),
        '',
        '=== Combined: SS + Notch + Low-pass ===',
        header, sep,
        row('Speech-band SNR (200-3kHz)', speech_band_snr_db(x),     speech_band_snr_db(y_full)),
        row('Power @ 60 Hz (hum)',        power_at(x, 60.0),         power_at(y_full, 60.0)),
        row('Noise floor (10th pctile)',  noise_floor_percentile_db(x),
                                          noise_floor_percentile_db(y_full)),
        row('Power @ 8 kHz (hiss)',       power_at(x, min(8000, FS/2 - 100)),
                                          power_at(y_full, min(8000, FS/2 - 100))),
        '',
    ]

    # Optional reference-based metrics
    if clean_ref is not None:
        # Truncate both to shortest length for fair comparison
        n = min(len(clean_ref), len(x), len(y_full))
        clean_n = clean_ref[:n]
        raw_n   = x[:n]
        proc_n  = y_full[:n]

        lines.append('=== Reference-based (vs. clean_voice) ===')
        lines.append(f"{'Metric':<34}{'RAW':>12}{'AFTER':>12}{'Δ':>10}")
        lines.append(sep)
        lines.append(row('Segmental SNR (dB)',
                         segmental_snr_db(clean_n, raw_n),
                         segmental_snr_db(clean_n, proc_n)))

        if _HAS_PESQ:
            try:
                p_raw  = pesq(FS, clean_n, raw_n,  'wb')
                p_proc = pesq(FS, clean_n, proc_n, 'wb')
                lines.append(f"  {'PESQ (wideband, 1.0-4.5)':<32}{p_raw:>12.2f}{p_proc:>12.2f}{p_proc - p_raw:>+10.2f}")
            except Exception as e:
                lines.append(f'  PESQ: error ({e})')
        else:
            lines.append('  PESQ: not installed (pip install pesq)')

        if _HAS_STOI:
            try:
                s_raw  = stoi(clean_n, raw_n,  FS, extended=False)
                s_proc = stoi(clean_n, proc_n, FS, extended=False)
                lines.append(f"  {'STOI (intelligibility, 0-1)':<32}{s_raw:>12.3f}{s_proc:>12.3f}{s_proc - s_raw:>+10.3f}")
            except Exception as e:
                lines.append(f'  STOI: error ({e})')
        else:
            lines.append('  STOI: not installed (pip install pystoi)')
        lines.append('')

    lines.append(f'SS α={SS_ALPHA}  β={SS_BETA}  smooth={SS_SMOOTH}  NFFT={SPEC_NFFT}')

    text = '\n'.join(lines)
    with open(outpath, 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')
    print(f'  wrote {outpath}')
    print()
    print(text)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _load_wav_to_fs(path: str) -> np.ndarray:
    fs_in, x = wavfile.read(path)
    x = x.astype(np.float32) / 32767.0
    if x.ndim > 1:
        x = x.mean(axis=1)
    if fs_in != FS:
        x = signal.resample_poly(x, FS, fs_in).astype(np.float32)
    return x


if __name__ == '__main__':
    # Resolve paths relative to THIS script unless they're absolute.
    # This way `python subfolder/analyze.py` from the parent folder still
    # finds the WAVs next to the script.
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(SCRIPT_DIR, p)

    wav       = _resolve(sys.argv[1] if len(sys.argv) > 1 else 'test_voice_snr15_fan_hum.wav')
    noise_wav = _resolve(sys.argv[2] if len(sys.argv) > 2 else 'noise_fan_only.wav')

    if not os.path.exists(wav):
        sys.exit(f'WAV not found: {wav}\n  (run generate_test_files.py first)')

    # Load noise reference (optional)
    noise = None
    if os.path.exists(noise_wav):
        noise = _load_wav_to_fs(noise_wav)
        print(f'Using noise reference: {noise_wav}')
    else:
        print(f'Noise reference not found ({noise_wav}) — skipping spectral subtraction.')
        print('  Run generate_test_files.py to create noise_fan_only.wav')

    # Load clean reference if available — enables PESQ / STOI / SegSNR
    clean_ref = None
    clean_path = _resolve('test_voice_clean.wav')
    if os.path.exists(clean_path):
        clean_ref = _load_wav_to_fs(clean_path)
        print(f'Using clean reference: {clean_path}  (enables PESQ/STOI/SegSNR)')

    print('\nGenerating analysis figures...')
    plot_filter_response(_resolve('filter_response.png'))

    x = _load_wav_to_fs(wav)

    if noise is not None:
        plot_ss_analysis(x, noise, _resolve('ss_analysis.png'))

    x_orig, _y_full = plot_demo_compare(wav, noise, _resolve('demo_compare.png'))
    write_metrics(x_orig, noise, wav, _resolve('metrics.txt'), clean_ref=clean_ref)
    print('\nDone.')