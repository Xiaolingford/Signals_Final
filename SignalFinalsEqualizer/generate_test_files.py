"""
generate_test_files.py
======================
Generates noisy speech test WAV files for the audio equalizer / denoiser demo.

Reads a clean voice recording from `clean_voice.wav` (in the same folder)
and produces noise-contaminated variants for benchmarking.

File formats produced:
  test_voice_clean.wav                 — reference (resampled to 16 kHz, mono)
  test_voice_snr<N>.wav                — + white noise
  test_voice_snr<N>_hum.wav            — + white noise + 60 Hz mains hum (+ harmonics)
  test_voice_snr<N>_fan.wav            — + white noise + broadband fan noise
  test_voice_snr<N>_fan_hum.wav        — full "worst case" file
  noise_fan_only.wav                   — pure fan noise (for calibration demo)

Broadband fan noise model
--------------------------
Real PC / HVAC fan noise is *pink noise* (power ∝ 1/f), not white.
We generate it by shaping white noise with a 1/sqrt(f) spectral envelope,
then adding narrow tonal components at the blade-pass frequency and its first
two harmonics (typical desktop fan: 120–200 Hz with harmonics).

Each noise file uses an INDEPENDENT random seed so the calibration clip is
NOT a perfect match to the noise in the test files (more realistic).

Usage:
    python SignalFinalsEqualizer/generate_test_files.py                  # default demo set
    python SignalFinalsEqualizer/generate_test_files.py 15               # one file at SNR=15 dB
    python SignalFinalsEqualizer/generate_test_files.py 15 --fan --hum   # custom combo
"""

import os
import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FS               = 16000      # Hz — wideband speech (was 8 kHz for telephony)
CLEAN_VOICE_FILE = 'clean_voice.wav'

# 60 Hz hum + harmonics (real mains interference has harmonics)
HUM_FREQ    = 60.0
HUM_AMP     = 0.15
HUM_HARM    = [1.0, 0.5, 0.3]   # relative amps of 60, 120, 180 Hz

# Fan noise parameters
FAN_BPF     = 150.0    # blade-pass fundamental (Hz)
FAN_TONAL   = 0.08     # amplitude of each tonal harmonic
FAN_PINK    = 0.15     # overall pink-noise amplitude scale


# ---------------------------------------------------------------------------
# Voice loader
# ---------------------------------------------------------------------------
def load_clean_voice(path: str = CLEAN_VOICE_FILE, fs: int = FS) -> np.ndarray:
    """
    Load a clean voice recording, convert to mono float32 in [-1, 1],
    and resample to `fs` if needed.
    """
    if not os.path.exists(path):
        sys.exit(
            f'\nERROR: Clean voice file not found: {path}\n'
            f'  Record ~5–10 s of speech in a quiet room, save as "{path}"\n'
            f'  in this folder, then re-run this script.\n'
            f'  Any sample rate works (will resample to {fs} Hz).\n'
        )

    fs_in, data = wavfile.read(path)
    data = data.astype(np.float32)

    # Stereo -> mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # int -> float in [-1, 1]
    if np.issubdtype(data.dtype, np.integer) or data.max() > 1.5:
        # Heuristic for various PCM bit depths
        data = data / np.max(np.abs(data) + 1e-12) * 0.9

    # Resample if needed
    if fs_in != fs:
        print(f'  resampling clean voice: {fs_in} Hz → {fs} Hz')
        data = resample_poly(data, fs, fs_in).astype(np.float32)

    # Normalize to ~ -3 dBFS to leave headroom for added noise
    peak = np.max(np.abs(data)) + 1e-12
    data = (data / peak * 0.7).astype(np.float32)

    print(f'  loaded clean voice: {len(data) / fs:.2f} s, {fs} Hz, peak={np.max(np.abs(data)):.3f}')
    return data


# ---------------------------------------------------------------------------
# Noise generators
# ---------------------------------------------------------------------------
def add_awgn(sig: np.ndarray, snr_db: float, rng=None) -> np.ndarray:
    """Add additive white Gaussian noise at the requested SNR (dB)."""
    rng = rng or np.random.default_rng()
    sig_power   = np.mean(sig ** 2) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=sig.shape).astype(np.float32)
    return sig + noise


def add_hum(sig: np.ndarray, fs: int = FS,
            freq: float = HUM_FREQ, amp: float = HUM_AMP) -> np.ndarray:
    """
    Inject mains-frequency hum WITH harmonics at 2× and 3× the fundamental.
    Real power-line interference is rarely a pure sinusoid.
    """
    t = np.arange(len(sig)) / fs
    hum = np.zeros_like(sig, dtype=np.float32)
    for k, weight in enumerate(HUM_HARM, start=1):
        hum += (amp * weight * np.sin(2 * np.pi * freq * k * t)).astype(np.float32)
    return sig + hum


def make_pink_noise(n_samples: int, fs: int = FS, rng=None) -> np.ndarray:
    """
    Generate normalised pink (1/f) noise via spectral shaping.

    Algorithm:
      1. White noise in frequency domain (complex Gaussian per bin).
      2. Multiply by 1/sqrt(f) envelope  →  PSD ∝ 1/f (pink).
      3. Zero DC, then real IFFT.
      4. Normalise to unit RMS.
    """
    rng = rng or np.random.default_rng()
    n = n_samples
    white_rfft = (rng.standard_normal(n // 2 + 1)
                + 1j * rng.standard_normal(n // 2 + 1)).astype(np.complex64)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    freqs[0] = 1.0          # avoid div-by-zero at DC; we zero it below
    pink_rfft = white_rfft / np.sqrt(freqs)
    pink_rfft[0] = 0.0      # remove DC
    pink = np.fft.irfft(pink_rfft, n=n).astype(np.float32)
    rms  = np.sqrt(np.mean(pink ** 2)) + 1e-12
    return pink / rms


def make_fan_noise(n_samples: int, fs: int = FS,
                   bpf: float = FAN_BPF,
                   tonal_amp: float = FAN_TONAL,
                   pink_amp: float = FAN_PINK,
                   rng=None) -> np.ndarray:
    """
    Simulate desktop-fan noise:
      • Pink broadband component  (captures the "whoosh")
      • Tonal component at BPF, 2×BPF, 3×BPF  (blade harmonics)
    """
    rng  = rng or np.random.default_rng()
    t    = np.arange(n_samples) / fs
    pink = make_pink_noise(n_samples, fs=fs, rng=rng) * pink_amp

    harmonics = np.zeros(n_samples, dtype=np.float32)
    for harmonic in (1, 2, 3):
        phase = rng.uniform(0, 2 * np.pi)
        harmonics += (tonal_amp * np.sin(2 * np.pi * bpf * harmonic * t + phase)
                      ).astype(np.float32)

    return (pink + harmonics).astype(np.float32)


def add_fan(sig: np.ndarray, fs: int = FS, rng=None) -> np.ndarray:
    """Add fan noise to `sig`."""
    fan = make_fan_noise(len(sig), fs=fs, rng=rng)
    return (sig + fan).astype(np.float32)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def normalize_for_save(x: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """
    Scale signal so its peak is at most `target_peak`, preventing the silent
    clipping that np.clip would otherwise inflict in to_int16().
    Only scales DOWN; never amplifies a quiet signal.
    """
    peak = np.max(np.abs(x)) + 1e-12
    if peak > target_peak:
        x = x * (target_peak / peak)
    return x.astype(np.float32)


def to_int16(x: np.ndarray) -> np.ndarray:
    """Clip to [-1, 1] and convert to 16-bit PCM."""
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)


def save_wav(path: str, sig: np.ndarray, fs: int = FS) -> None:
    sig = normalize_for_save(sig)
    wavfile.write(path, fs, to_int16(sig))
    dur = len(sig) / fs
    rms = float(np.sqrt(np.mean(sig ** 2)))
    print(f'  wrote  {path:<50}  ({dur:.2f}s, RMS={rms:.3f})')


# ---------------------------------------------------------------------------
# File factory
# ---------------------------------------------------------------------------
def make_file(clean: np.ndarray,
              snr_db=None,
              hum: bool = False,
              fan: bool = False,
              seed: int | None = None,
              outdir: str = '.') -> None:
    """Generate one noisy variant of the clean speech."""
    rng = np.random.default_rng(seed)
    sig = clean.copy()

    if snr_db is not None:
        sig = add_awgn(sig, snr_db, rng=rng)
        tag = f'snr{int(snr_db)}'
    else:
        tag = 'clean'

    if fan:
        sig = add_fan(sig, rng=rng)
        tag = tag + '_fan'

    if hum:
        sig = add_hum(sig)
        tag = tag + '_hum'

    fname = f'test_voice_{tag}.wav'
    save_wav(os.path.join(outdir, fname), sig)


def make_noise_only_file(duration_s: float = 5.0,
                         outdir: str = '.',
                         seed: int | None = None) -> None:
    """
    Pure fan noise clip for calibration. Independent seed → realistic
    mismatch between calibration noise and test-file noise.
    """
    rng = np.random.default_rng(seed)
    n   = int(FS * duration_s)
    fan = make_fan_noise(n, rng=rng)
    save_wav(os.path.join(outdir, 'noise_fan_only.wav'), fan)


# ---------------------------------------------------------------------------
# Default demo battery
# ---------------------------------------------------------------------------
def build_default_set(outdir: str = '.') -> None:
    os.makedirs(outdir, exist_ok=True)
    print(f'Generating default test set in "{outdir}/"  ({FS} Hz wideband)')
    print()

    clean = load_clean_voice()

    # Clean reference (no noise)
    print('\n--- Clean reference ---')
    save_wav(os.path.join(outdir, 'test_voice_clean.wav'), clean)

    # White-noise SNR sweep
    print('\n--- White-noise SNR sweep ---')
    for snr in (30, 20, 15, 10, 0):
        make_file(clean, snr_db=snr, outdir=outdir)

    # Hum-only contamination (notch-filter demo)
    print('\n--- Mains hum (notch demo) ---')
    make_file(clean, snr_db=15, hum=True, outdir=outdir)

    # Fan noise (spectral-subtraction demo) — independent seeds per file
    print('\n--- Broadband fan noise (spectral-subtraction demo) ---')
    make_file(clean, snr_db=20, fan=True, seed=101, outdir=outdir)
    make_file(clean, snr_db=15, fan=True, seed=102, outdir=outdir)
    make_file(clean, snr_db=10, fan=True, seed=103, outdir=outdir)

    # Full "worst case": fan + hum + noise
    print('\n--- Fan + hum + noise (full demo) ---')
    make_file(clean, snr_db=15, fan=True, hum=True, seed=201, outdir=outdir)
    make_file(clean, snr_db=10, fan=True, hum=True, seed=202, outdir=outdir)

    # Calibration clip — INDEPENDENT seed from above, simulating real-life
    # mismatch between "noise sample I captured" and "noise in the recording"
    print('\n--- Calibration clip (fan noise only) ---')
    make_noise_only_file(duration_s=5.0, outdir=outdir, seed=999)

    print('\nDone.  Hero demo sequence:')
    print('  1.  python equalizer.py --file test_voice_snr15_fan_hum.wav')
    print('  2.  python equalizer.py --file noise_fan_only.wav')
    print('      → click [Calibrate Noise] to capture the noise profile')
    print('  3.  Back in step-1 window, toggle [Spectral Sub] + [Notch 60Hz]')
    print('  4.  For metrics:  python analyze.py')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) == 1:
        build_default_set(outdir='.')
    else:
        snr     = float(sys.argv[1]) if len(sys.argv) >= 2 else None
        use_hum = '--hum' in sys.argv
        use_fan = '--fan' in sys.argv
        clean   = load_clean_voice()
        make_file(clean, snr_db=snr, hum=use_hum, fan=use_fan, outdir='.')