"""
analyze.py
==========
Generates the report figures and metrics for the equalizer / denoiser demo.

Outputs:
  filter_response.png   — magnitude/phase of notch + low-pass, pole-zero plot
  ss_analysis.png       — spectral subtraction: noise PSD, before/after spectra,
                          musical-noise floor, α/β sensitivity sweep
  demo_compare.png      — before / after spectrograms for a test WAV
  metrics.txt           — SNR before/after for all three filter stages

Usage:
    python analyze.py                                     # default file
    python analyze.py path/to/file.wav [noise_only.wav]
    python analyze.py test_5551234_snr15_fan_hum.wav noise_fan_only.wav
"""

import os
import sys
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Mirror config from equalizer.py
# ---------------------------------------------------------------------------
FS          = 8000
NOTCH_FREQ  = 60.0
NOTCH_Q     = 30.0
LP_CUTOFF   = 3400.0
LP_TAPS     = 101
SS_ALPHA    = 2.0
SS_BETA     = 0.02
SPEC_NFFT   = 512      # must match equalizer.py (and be even)
HOP         = SPEC_NFFT // 2

NOTCH_B, NOTCH_A = signal.iirnotch(NOTCH_FREQ, NOTCH_Q, fs=FS)
LP_B = signal.firwin(LP_TAPS, LP_CUTOFF, fs=FS, window='hamming')
LP_A = np.array([1.0])

WIN = np.sqrt(np.hanning(SPEC_NFFT)).astype(np.float32)  # sqrt-Hanning


# ---------------------------------------------------------------------------
# Offline spectral subtractor  (mirrors equalizer.SpectralSubtractor)
# ---------------------------------------------------------------------------
def spectral_subtract_offline(
        x_speech: np.ndarray,
        x_noise: np.ndarray,
        alpha: float = SS_ALPHA,
        beta: float  = SS_BETA,
) -> np.ndarray:
    """
    Offline spectral subtraction using the sqrt-Hanning overlap-add STFT.

    x_speech  : signal to clean (float32, already normalised to [-1, 1])
    x_noise   : noise-only reference for estimating noise PSD (may be shorter)
    returns   : cleaned float32 signal, same length as x_speech
    """
    def stft_frames(x):
        """Yield windowed FFT frames with 50% overlap."""
        n = len(x)
        for start in range(0, n - SPEC_NFFT + 1, HOP):
            frame = x[start:start + SPEC_NFFT] * WIN
            yield np.fft.rfft(frame)

    # 1. Estimate noise magnitude spectrum
    noise_mags = [np.abs(X) for X in stft_frames(x_noise.astype(np.float32))]
    if not noise_mags:
        return x_speech.copy()
    noise_mag = np.mean(noise_mags, axis=0).astype(np.float32)

    # 2. Process speech signal frame by frame (overlap-add)
    n       = len(x_speech)
    out_buf = np.zeros(n + SPEC_NFFT, dtype=np.float32)

    for i, start in enumerate(range(0, n - SPEC_NFFT + 1, HOP)):
        frame = x_speech[start:start + SPEC_NFFT].astype(np.float32) * WIN
        X     = np.fft.rfft(frame)
        mag   = np.abs(X)
        phase = np.angle(X)

        mag_clean = np.maximum(mag - alpha * noise_mag, beta * noise_mag)
        X_clean   = mag_clean * np.exp(1j * phase)
        y_frame   = np.fft.irfft(X_clean).real.astype(np.float32) * WIN

        out_buf[start:start + SPEC_NFFT] += y_frame

    return out_buf[:n]


# ---------------------------------------------------------------------------
# Offline pass of all three filters
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
        zi = signal.lfilter_zi(NOTCH_B, NOTCH_A) * y[0]
        y, _ = signal.lfilter(NOTCH_B, NOTCH_A, y, zi=zi)

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


def band_snr_db(x: np.ndarray, fs: int = FS) -> float:
    X   = np.abs(np.fft.rfft(x)) ** 2
    f   = np.fft.rfftfreq(len(x), 1 / fs)
    sig = X[(f >= 200) & (f <= 2000)].sum() + 1e-12
    noi = X[f > 2500].sum() + 1e-12
    return 10 * np.log10(sig / noi)


def broadband_noise_floor_db(x: np.ndarray, fs: int = FS,
                              flo: float = 100.0, fhi: float = 3000.0) -> float:
    """Mean power spectral density across a broad band (dB/bin)."""
    X = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / fs)
    sel = (f >= flo) & (f <= fhi)
    return 10 * np.log10(X[sel].mean() + 1e-12)


# ---------------------------------------------------------------------------
# Figure 1 — Filter frequency responses
# ---------------------------------------------------------------------------
def plot_filter_response(outpath: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Filter Frequency Responses', fontsize=14)

    # Notch magnitude
    w, h = signal.freqz(NOTCH_B, NOTCH_A, worN=8192, fs=FS)
    ax = axes[0, 0]
    ax.plot(w, 20 * np.log10(np.abs(h) + 1e-12))
    ax.set_title(f'IIR Notch @ {NOTCH_FREQ:g} Hz  (Q = {NOTCH_Q:g})')
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('|H(f)| (dB)')
    ax.set_xlim(0, 500); ax.set_ylim(-60, 5); ax.grid(alpha=0.3)
    ax.axvline(60, color='red', ls='--', alpha=0.5, label='60 Hz')
    ax.legend(loc='lower right')

    # Notch pole-zero
    ax = axes[0, 1]
    z, p, _k = signal.tf2zpk(NOTCH_B, NOTCH_A)
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color='gray', lw=0.7)
    ax.scatter(z.real, z.imag, marker='o', s=80, facecolors='none',
               edgecolors='C0', label='zeros')
    ax.scatter(p.real, p.imag, marker='x', s=80, color='C3', label='poles')
    ax.set_title('Notch pole-zero plot (z-plane)')
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
    """
    Four-panel figure explaining and evaluating spectral subtraction:
      TL: Estimated noise PSD (from calibration clip)
      TR: Before/after mean power spectra
      BL: Spectrogram before SS
      BR: Spectrogram after  SS
    """
    x_clean = spectral_subtract_offline(x_noisy, x_noise)
    f_axis  = np.fft.rfftfreq(SPEC_NFFT, 1 / FS)

    # Noise PSD estimate
    noise_mags = []
    for start in range(0, len(x_noise) - SPEC_NFFT + 1, HOP):
        frame = x_noise[start:start + SPEC_NFFT].astype(np.float32) * WIN
        noise_mags.append(np.abs(np.fft.rfft(frame)))
    noise_mag_mean = np.mean(noise_mags, axis=0) if noise_mags else np.zeros(len(f_axis))

    # Mean magnitude spectra (noisy vs. clean)
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
        f'Spectral Subtraction Analysis  (α={SS_ALPHA}, β={SS_BETA})',
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
    for harm in [60, 150, 300]:
        ax.axhline(harm, color='cyan', ls=':', alpha=0.5, lw=0.8)

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
    nperseg = 256
    for ax, sig_, title in [
        (axes[0], x, 'BEFORE  (input)'),
        (axes[1], y, 'AFTER   (spectral sub + notch + low-pass)'),
    ]:
        f, t, Sxx = signal.spectrogram(sig_, fs=FS, nperseg=nperseg,
                                       noverlap=nperseg // 2, scaling='spectrum')
        Sdb = 10 * np.log10(Sxx + 1e-12)
        ax.pcolormesh(t, f, Sdb, vmin=-80, vmax=-10, cmap='magma', shading='auto')
        ax.axhline(60,        color='cyan', ls='--', alpha=0.6, lw=0.8)
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
                  outpath: str):
    # Stage outputs
    y_ss_only  = process_offline(x, noise, use_ss=True,  use_notch=False, use_lowpass=False) \
                 if noise is not None else x
    y_notch    = process_offline(x, None,  use_ss=False, use_notch=True,  use_lowpass=False)
    y_full     = process_offline(x, noise, use_ss=(noise is not None),
                                 use_notch=True, use_lowpass=True)

    def row(label, b, a):
        return (f"{'  ' + label:<32}{b:>12.1f}{a:>12.1f}{a - b:>+10.1f}")

    header = (
        f"{'Metric':<34}{'RAW (dB)':>12}{'AFTER (dB)':>12}{'Δ (dB)':>10}"
    )
    sep = '-' * 68

    lines = [
        f'Metrics for: {wav_path}',
        f'Sample rate : {FS} Hz',
        f'Length      : {len(x) / FS:.2f} s',
        '',
        '=== Stage 1: Spectral Subtraction (broadband fan noise) ===',
        header, sep,
        row('Band SNR (200-2000 Hz)',     band_snr_db(x),          band_snr_db(y_ss_only)),
        row('Noise floor (100-3000 Hz)',  broadband_noise_floor_db(x),
                                          broadband_noise_floor_db(y_ss_only)),
        row('Power @ 150 Hz (fan BPF)',   power_at(x, 150.0),      power_at(y_ss_only, 150.0)),
        row('Power @ 300 Hz (2nd harm)',  power_at(x, 300.0),      power_at(y_ss_only, 300.0)),
        '',
        '=== Stage 2: IIR Notch (60 Hz hum) ===',
        header, sep,
        row('Power @ 60 Hz (hum)',        power_at(x, 60.0),       power_at(y_notch, 60.0)),
        row('Power @ 100 Hz (control)',   power_at(x, 100.0),      power_at(y_notch, 100.0)),
        '',
        '=== Combined: SS + Notch + Low-pass ===',
        header, sep,
        row('Band SNR (200-2000 Hz)',     band_snr_db(x),          band_snr_db(y_full)),
        row('Power @ 60 Hz (hum)',        power_at(x, 60.0),       power_at(y_full, 60.0)),
        row('Noise floor (100-3000 Hz)',  broadband_noise_floor_db(x),
                                          broadband_noise_floor_db(y_full)),
        row('Power @ 3000 Hz (hiss)',     power_at(x, 3000.0),     power_at(y_full, 3000.0)),
        '',
        f'SS α={SS_ALPHA}  β={SS_BETA}  NFFT={SPEC_NFFT}  HOP={HOP}',
    ]

    text = '\n'.join(lines)
    with open(outpath, 'w') as fh:
        fh.write(text + '\n')
    print(f'  wrote {outpath}')
    print()
    print(text)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    wav   = sys.argv[1] if len(sys.argv) > 1 else 'test_5551234_snr15_fan_hum.wav'
    noise_wav = sys.argv[2] if len(sys.argv) > 2 else 'noise_fan_only.wav'

    if not os.path.exists(wav):
        sys.exit(f'WAV not found: {wav}\n  (run generate_test_files.py first)')

    # Load noise reference (optional — skip if missing)
    noise = None
    if os.path.exists(noise_wav):
        fs_n, raw_n = wavfile.read(noise_wav)
        noise = raw_n.astype(np.float32) / 32767.0
        if noise.ndim > 1:
            noise = noise.mean(axis=1)
        if fs_n != FS:
            noise = signal.resample_poly(noise, FS, fs_n).astype(np.float32)
        print(f'Using noise reference: {noise_wav}')
    else:
        print(f'Noise reference not found ({noise_wav}) — skipping spectral subtraction.')
        print('  Run generate_test_files.py to create noise_fan_only.wav')

    print('\nGenerating analysis figures...')
    plot_filter_response('filter_response.png')

    fs_in, x = wavfile.read(wav)
    x = x.astype(np.float32) / 32767.0
    if x.ndim > 1:
        x = x.mean(axis=1)
    if fs_in != FS:
        x = signal.resample_poly(x, FS, fs_in).astype(np.float32)

    if noise is not None:
        plot_ss_analysis(x, noise, 'ss_analysis.png')

    x_orig, y_full = plot_demo_compare(wav, noise, 'demo_compare.png')
    write_metrics(x_orig, noise, wav, 'metrics.txt')
    print('\nDone.')