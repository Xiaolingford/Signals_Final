"""
generate_test_files.py
======================
Generates DTMF test WAV files for the audio equalizer / denoiser demo.

File formats produced:
  test_<digits>_clean.wav              — no noise
  test_<digits>_snr<N>.wav            — white-noise contaminated
  test_<digits>_snr<N>_hum.wav        — white noise + 60 Hz mains hum
  test_<digits>_snr<N>_fan.wav        — white noise + broadband fan noise
  test_<digits>_snr<N>_fan_hum.wav    — the full "worst case" file
  noise_fan_only.wav                  — pure fan noise (for calibration demo)

Broadband fan noise model
--------------------------
Real PC / HVAC fan noise is *pink noise* (power ∝ 1/f), not white.
We generate it by shaping white noise with a 1/sqrt(f) spectral envelope,
then adding narrow tonal components at the blade-pass frequency and its first
two harmonics (typical desktop fan: 120–200 Hz with harmonics).
This gives a realistic "computer fan" sound that cannot be removed with a
simple notch or low-pass filter.

Usage:
    python generate_test_files.py              # default demo set
    python generate_test_files.py 911 20       # one file: digits=911, snr=20 dB
    python generate_test_files.py 911 20 --fan --hum
"""

import os
import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter

FS        = 8000     # Hz
TONE_MS   = 100      # ms per DTMF digit
GAP_MS    = 50       # ms silence between digits
TONE_AMP  = 0.40     # per-tone amplitude (sum of two ≈ 0.80 peak)

# 60 Hz hum
HUM_FREQ  = 60.0     # Hz
HUM_AMP   = 0.20

# Fan noise parameters
FAN_BPF   = 150.0    # blade-pass fundamental (Hz) — typical desktop fan
FAN_TONAL = 0.08     # amplitude of each tonal component
FAN_PINK  = 0.15     # overall pink-noise amplitude scale

# Standard DTMF frequency table
DTMF_ROW = {
    '1': 697, '2': 697, '3': 697, 'A': 697,
    '4': 770, '5': 770, '6': 770, 'B': 770,
    '7': 852, '8': 852, '9': 852, 'C': 852,
    '*': 941, '0': 941, '#': 941, 'D': 941,
}
DTMF_COL = {
    '1': 1209, '4': 1209, '7': 1209, '*': 1209,
    '2': 1336, '5': 1336, '8': 1336, '0': 1336,
    '3': 1477, '6': 1477, '9': 1477, '#': 1477,
    'A': 1633, 'B': 1633, 'C': 1633, 'D': 1633,
}


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------
def dtmf_signal(digits: str, fs: int = FS) -> np.ndarray:
    """Clean DTMF waveform for the given digit string."""
    tone_n = int(fs * TONE_MS / 1000)
    gap_n  = int(fs * GAP_MS  / 1000)
    t      = np.arange(tone_n) / fs
    silence = np.zeros(gap_n, dtype=np.float32)
    out = []
    for d in digits:
        if d not in DTMF_ROW:
            raise ValueError(f'Unknown DTMF digit: {d!r}')
        f1, f2 = DTMF_ROW[d], DTMF_COL[d]
        tone = (TONE_AMP * np.sin(2 * np.pi * f1 * t)
              + TONE_AMP * np.sin(2 * np.pi * f2 * t))
        out.append(tone.astype(np.float32))
        out.append(silence)
    return np.concatenate(out)


def add_awgn(sig: np.ndarray, snr_db: float, rng=None) -> np.ndarray:
    """Add additive white Gaussian noise at the requested SNR (dB)."""
    rng = rng or np.random.default_rng(0xC0FFEE)
    sig_power   = np.mean(sig ** 2) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=sig.shape).astype(np.float32)
    return sig + noise


def add_hum(sig: np.ndarray, fs: int = FS,
            freq: float = HUM_FREQ, amp: float = HUM_AMP) -> np.ndarray:
    """Inject a single mains-frequency sinusoid."""
    t = np.arange(len(sig)) / fs
    return sig + (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_pink_noise(n_samples: int, fs: int = FS, rng=None) -> np.ndarray:
    """
    Generate normalised pink (1/f) noise via spectral shaping.

    Algorithm:
      1. White noise in frequency domain.
      2. Multiply by 1/sqrt(f) envelope  (gives pink PSD ∝ 1/f).
      3. Force Hermitian symmetry → real IFFT.
      4. Normalise to unit RMS.
    """
    rng = rng or np.random.default_rng(0xDEADBEEF)
    n   = n_samples
    # rfft bins
    white_rfft = (rng.standard_normal(n // 2 + 1)
                + 1j * rng.standard_normal(n // 2 + 1)).astype(np.complex64)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    freqs[0] = 1.0          # avoid divide-by-zero at DC
    pink_rfft = white_rfft / np.sqrt(freqs)
    pink_rfft[0] = 0.0      # zero DC component
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
    The result is normalised so that RMS ≈ pink_amp + 3×tonal_amp.
    """
    rng  = rng or np.random.default_rng(0xFAFAFAFA)
    t    = np.arange(n_samples) / fs
    pink = make_pink_noise(n_samples, fs=fs, rng=rng) * pink_amp

    # Random phase offsets for each harmonic (more realistic)
    rng2 = np.random.default_rng(0xBADCAFE)
    harmonics = np.zeros(n_samples, dtype=np.float32)
    for k, harmonic in enumerate([1, 2, 3], start=1):
        phase = rng2.uniform(0, 2 * np.pi)
        harmonics += (tonal_amp * np.sin(2 * np.pi * bpf * harmonic * t + phase)
                      ).astype(np.float32)

    fan = pink + harmonics
    return fan.astype(np.float32)


def add_fan(sig: np.ndarray, fs: int = FS,
            bpf: float = FAN_BPF,
            tonal_amp: float = FAN_TONAL,
            pink_amp: float = FAN_PINK) -> np.ndarray:
    """Add fan noise to `sig`."""
    fan = make_fan_noise(len(sig), fs=fs, bpf=bpf,
                         tonal_amp=tonal_amp, pink_amp=pink_amp)
    return (sig + fan).astype(np.float32)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def to_int16(x: np.ndarray) -> np.ndarray:
    """Clip to [-1, 1] and convert to 16-bit PCM."""
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)


def save_wav(path: str, sig: np.ndarray, fs: int = FS) -> None:
    wavfile.write(path, fs, to_int16(sig))
    dur = len(sig) / fs
    rms = float(np.sqrt(np.mean(sig ** 2)))
    print(f'  wrote  {path:<50}  ({dur:.2f}s, RMS={rms:.3f})')


# ---------------------------------------------------------------------------
# File factory
# ---------------------------------------------------------------------------
def make_file(digits: str,
              snr_db=None,
              hum: bool = False,
              fan: bool = False,
              outdir: str = '.') -> None:
    """Generate one .wav according to the naming convention."""
    sig = dtmf_signal(digits)

    # Build noise tag
    if snr_db is not None:
        sig = add_awgn(sig, snr_db)
        tag = f'snr{int(snr_db)}'
    else:
        tag = 'clean'

    if fan:
        sig = add_fan(sig)
        tag = tag + '_fan'

    if hum:
        sig = add_hum(sig)
        tag = tag + '_hum'

    fname = f'test_{digits}_{tag}.wav'
    save_wav(os.path.join(outdir, fname), sig)


def make_noise_only_file(duration_s: float = 4.0,
                         outdir: str = '.') -> None:
    """
    Pure fan noise clip — used in the live demo to show the calibration step.
    Run equalizer.py --file noise_fan_only.wav, click [Calibrate Noise],
    then switch back to the speech+fan file.
    """
    n   = int(FS * duration_s)
    fan = make_fan_noise(n)
    fname = 'noise_fan_only.wav'
    save_wav(os.path.join(outdir, fname), fan)


# ---------------------------------------------------------------------------
# Default demo battery
# ---------------------------------------------------------------------------
def build_default_set(outdir: str = '.') -> None:
    os.makedirs(outdir, exist_ok=True)
    print(f'Generating default test set in "{outdir}/"')
    print()

    # Clean references (no noise)
    print('--- Clean references ---')
    for digits in ('911', '5551234', '0123456789'):
        make_file(digits, snr_db=None, outdir=outdir)

    # White-noise SNR sweep on 5551234
    print('\n--- White-noise SNR sweep ---')
    for snr in (30, 20, 15, 10, 0):
        make_file('5551234', snr_db=snr, outdir=outdir)

    # Hum-only contamination (original notch-filter demo)
    print('\n--- Mains-hum contamination (notch demo) ---')
    make_file('5551234', snr_db=15, hum=True, outdir=outdir)
    make_file('911',     snr_db=10, hum=True, outdir=outdir)

    # Fan-noise contamination (spectral-subtraction demo)
    print('\n--- Broadband fan noise (spectral-subtraction demo) ---')
    make_file('5551234', snr_db=20, fan=True,             outdir=outdir)
    make_file('5551234', snr_db=15, fan=True,             outdir=outdir)
    make_file('911',     snr_db=15, fan=True,             outdir=outdir)

    # Full "worst case": fan + hum + noise
    print('\n--- Fan + hum + noise (full demo) ---')
    make_file('5551234', snr_db=15, fan=True, hum=True, outdir=outdir)
    make_file('911',     snr_db=10, fan=True, hum=True, outdir=outdir)

    # Pure noise calibration clip
    print('\n--- Calibration clip (fan noise only) ---')
    make_noise_only_file(duration_s=5.0, outdir=outdir)

    print('\nDone.  Hero demo sequence:')
    print('  1.  python equalizer.py --file test_5551234_snr15_fan_hum.wav')
    print('  2.  In another terminal / tab:')
    print('      python equalizer.py --file noise_fan_only.wav')
    print('      → click [Calibrate Noise] to capture the noise profile')
    print('  3.  Back in step-1 window, toggle [Spectral Sub] + [Notch 60Hz]')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) == 1:
        build_default_set(outdir='.')
    else:
        digits  = sys.argv[1]
        snr     = float(sys.argv[2]) if len(sys.argv) >= 3 else None
        use_hum = '--hum' in sys.argv
        use_fan = '--fan' in sys.argv
        make_file(digits, snr_db=snr, hum=use_hum, fan=use_fan, outdir='.')