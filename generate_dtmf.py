"""
generate_dtmf.py
================
Generates DTMF-encoded test WAV files for the Signal Generation & Testing task.

Output format:
  - Mono, 8000 Hz sample rate, 16-bit PCM
  - Filename: test_<digits>_snr<level>.wav  or  test_<digits>_clean.wav
  - 100 ms tone per digit, 50 ms silence between digits

DTMF frequency table (standard ITU-T):
         1209 Hz  1336 Hz  1477 Hz  1633 Hz
697 Hz     1        2        3        A
770 Hz     4        5        6        B
852 Hz     7        8        9        C
941 Hz     *        0        #        D

Usage:
    python generate_dtmf.py
    python generate_dtmf.py --digits 5551234 --snr 20
    python generate_dtmf.py --digits 911 --clean
    python generate_dtmf.py --suite          # builds full test suite
"""

import argparse
import os
import struct
import wave

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 8000          # Hz
TONE_MS       = 100           # ms per digit
SILENCE_MS    = 50            # ms silence between digits
AMPLITUDE     = 0.4           # 0–1 full scale (equal for both DTMF freqs)
BIT_DEPTH     = 16            # PCM bit depth
OUTPUT_DIR    = "test_wavs"   # folder for generated files

# DTMF frequency pairs {digit: (row_freq, col_freq)}
DTMF_FREQS: dict[str, tuple[int, int]] = {
    "1": (697, 1209), "2": (697, 1336), "3": (697, 1477),
    "4": (770, 1209), "5": (770, 1336), "6": (770, 1477),
    "7": (852, 1209), "8": (852, 1336), "9": (852, 1477),
    "*": (941, 1209), "0": (941, 1336), "#": (941, 1477),
    "A": (697, 1633), "B": (770, 1633), "C": (852, 1633), "D": (941, 1633),
}


# ── Core signal generation ────────────────────────────────────────────────────

def generate_tone(freq1: int, freq2: int, duration_ms: int,
                  sample_rate: int = SAMPLE_RATE,
                  amplitude: float = AMPLITUDE) -> np.ndarray:
    """
    Generate a single DTMF tone (sum of two sinusoids).

    Parameters
    ----------
    freq1, freq2   : row and column frequencies (Hz)
    duration_ms    : tone duration in milliseconds
    sample_rate    : samples per second
    amplitude      : per-component amplitude (0–1); final signal is 2× this

    Returns
    -------
    samples : float64 ndarray, peak ≤ 1.0
    """
    n_samples = int(sample_rate * duration_ms / 1000)
    t = np.arange(n_samples) / sample_rate
    signal = amplitude * np.sin(2 * np.pi * freq1 * t) \
           + amplitude * np.sin(2 * np.pi * freq2 * t)
    return signal


def generate_silence(duration_ms: int,
                     sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Return a zero-valued silence block."""
    return np.zeros(int(sample_rate * duration_ms / 1000))


def build_dtmf_signal(digit_string: str,
                      tone_ms:    int   = TONE_MS,
                      silence_ms: int   = SILENCE_MS,
                      sample_rate: int  = SAMPLE_RATE,
                      amplitude:  float = AMPLITUDE) -> np.ndarray:
    """
    Concatenate DTMF tones for every character in *digit_string*.

    Unknown characters are silently skipped with a warning.

    Returns
    -------
    signal : float64 ndarray (mono, values in [-1, 1])
    """
    segments: list[np.ndarray] = []
    for i, ch in enumerate(digit_string.upper()):
        if ch not in DTMF_FREQS:
            print(f"  [WARN] '{ch}' is not a valid DTMF character — skipped.")
            continue
        f1, f2 = DTMF_FREQS[ch]
        segments.append(generate_tone(f1, f2, tone_ms, sample_rate, amplitude))
        if i < len(digit_string) - 1:                   # no trailing silence
            segments.append(generate_silence(silence_ms, sample_rate))

    if not segments:
        raise ValueError("No valid DTMF digits found in the input string.")

    return np.concatenate(segments)


# ── WAV I/O ──────────────────────────────────────────────────────────────────

def float_to_pcm16(signal: np.ndarray) -> bytes:
    """
    Convert float64 [-1, 1] samples to signed 16-bit little-endian PCM bytes.
    Clips to prevent overflow.
    """
    clipped = np.clip(signal, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    return pcm.tobytes()


def save_wav(filename: str, signal: np.ndarray,
             sample_rate: int = SAMPLE_RATE) -> None:
    """Write *signal* (float64) to a 16-bit mono WAV file."""
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with wave.open(filename, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 2 bytes = 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(float_to_pcm16(signal))
    print(f"  Saved → {filename}")


def load_wav(filename: str) -> tuple[np.ndarray, int]:
    """
    Load a 16-bit mono WAV file.

    Returns
    -------
    signal      : float64 ndarray normalised to [-1, 1]
    sample_rate : int
    """
    with wave.open(filename, "r") as wf:
        assert wf.getnchannels() == 1,    "Only mono WAV supported."
        assert wf.getsampwidth() == 2,    "Only 16-bit WAV supported."
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32767
    return samples, sample_rate


# ── Noise injection ───────────────────────────────────────────────────────────

def compute_snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """Measure SNR in dB given separate signal and noise arrays (same length)."""
    power_signal = np.mean(signal ** 2)
    power_noise  = np.mean(noise  ** 2)
    if power_noise == 0:
        return float("inf")
    return 10 * np.log10(power_signal / power_noise)


def add_white_noise(signal: np.ndarray, snr_db: float,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Add AWGN (additive white Gaussian noise) to *signal* at a given SNR.

    Parameters
    ----------
    signal : clean float64 signal
    snr_db : desired output SNR in dB
    rng    : optional numpy random generator for reproducibility

    Returns
    -------
    noisy_signal : float64 ndarray, same length as *signal*
    """
    rng = rng or np.random.default_rng()
    power_signal = np.mean(signal ** 2)
    if power_signal == 0:
        raise ValueError("Signal has zero power — cannot set SNR.")
    snr_linear   = 10 ** (snr_db / 10)
    power_noise  = power_signal / snr_linear
    noise = rng.normal(0, np.sqrt(power_noise), size=len(signal))
    return signal + noise


def add_60hz_hum(signal: np.ndarray, hum_amplitude: float = 0.15,
                 sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Inject a 60 Hz sinusoidal hum (power-line interference).

    Parameters
    ----------
    hum_amplitude : amplitude of the hum component (0–1 FS)
    """
    t = np.arange(len(signal)) / sample_rate
    hum = hum_amplitude * np.sin(2 * np.pi * 60 * t)
    return signal + hum


def add_bandlimited_noise(signal: np.ndarray, snr_db: float,
                          low_hz: float = 4000, high_hz: float = 4000,
                          sample_rate: int = SAMPLE_RATE,
                          rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Add high-frequency hiss (noise above *low_hz*).
    Uses a simple spectral-shaping approach via numpy FFT.
    """
    rng = rng or np.random.default_rng()
    n = len(signal)
    white = rng.normal(0, 1, n)
    # Zero out below low_hz in frequency domain
    freqs = np.fft.rfftfreq(n, 1 / sample_rate)
    spectrum = np.fft.rfft(white)
    spectrum[freqs < low_hz] = 0
    hiss_raw = np.fft.irfft(spectrum, n=n)
    # Scale to desired SNR
    power_signal = np.mean(signal ** 2)
    power_hiss   = np.mean(hiss_raw ** 2)
    if power_hiss == 0:
        return signal
    snr_linear = 10 ** (snr_db / 10)
    scale = np.sqrt(power_signal / (power_hiss * snr_linear))
    return signal + hiss_raw * scale


# ── Filename helpers ──────────────────────────────────────────────────────────

def make_filename(digits: str, snr: int | None = None,
                  noise_type: str = "white",
                  output_dir: str = OUTPUT_DIR) -> str:
    """
    Build output filename following the project convention:
        test_<digits>_snr<level>.wav   — noisy
        test_<digits>_clean.wav        — clean
    """
    digits_clean = digits.replace("*", "star").replace("#", "hash")
    if snr is None:
        tag = "clean"
    else:
        tag = f"snr{snr:+d}".replace("+", "")   # e.g. snr20, snr-5
        if noise_type != "white":
            tag += f"_{noise_type}"
    return os.path.join(output_dir, f"test_{digits_clean}_{tag}.wav")


# ── High-level generators ─────────────────────────────────────────────────────

def generate_clean(digits: str, output_dir: str = OUTPUT_DIR) -> str:
    """Generate and save a clean (no noise) DTMF WAV. Returns filename."""
    signal   = build_dtmf_signal(digits)
    filename = make_filename(digits, snr=None, output_dir=output_dir)
    save_wav(filename, signal)
    return filename


def generate_noisy(digits: str, snr_db: int,
                   noise_type: str = "white",
                   add_hum: bool  = False,
                   output_dir: str = OUTPUT_DIR,
                   seed: int = 42) -> str:
    """
    Generate and save a noisy DTMF WAV at the specified SNR.

    Parameters
    ----------
    digits     : digit string (e.g. "5551234")
    snr_db     : target SNR in dB
    noise_type : "white" | "hiss"
    add_hum    : if True, also inject 60 Hz hum
    seed       : random seed for reproducibility

    Returns filename.
    """
    rng    = np.random.default_rng(seed)
    signal = build_dtmf_signal(digits)

    if noise_type == "hiss":
        noisy = add_bandlimited_noise(signal, snr_db, rng=rng)
    else:
        noisy = add_white_noise(signal, snr_db, rng=rng)

    if add_hum:
        noisy = add_60hz_hum(noisy)

    filename = make_filename(digits, snr=snr_db,
                             noise_type=noise_type, output_dir=output_dir)
    save_wav(filename, noisy)

    # Report actual measured SNR
    pure_noise = noisy - signal
    actual_snr = compute_snr_db(signal, pure_noise)
    print(f"    Actual SNR = {actual_snr:.1f} dB  (target {snr_db} dB)")
    return filename


# ── Test-suite builder ────────────────────────────────────────────────────────

def build_test_suite(output_dir: str = OUTPUT_DIR) -> None:
    """
    Generate the full standard test suite:

    Digit sequences
    ---------------
    • Short  : "911", "123", "0"
    • Medium : "5551234", "4085550100"
    • Long   : "18005551234"

    SNR levels (dB)
    ---------------
    30, 25, 20, 15, 10, 5, 0, -5

    Noise types
    -----------
    white noise, high-frequency hiss, white noise + 60 Hz hum
    """
    digit_sequences = [
        "911",
        "123",
        "0",
        "5551234",
        "4085550100",
        "18005551234",
    ]
    snr_levels  = [30, 25, 20, 15, 10, 5, 0, -5]
    noise_types = ["white", "hiss"]

    os.makedirs(output_dir, exist_ok=True)
    total = 0

    print("=" * 60)
    print(" Building test suite …")
    print("=" * 60)

    for digits in digit_sequences:
        print(f"\n── {digits} ──")

        # 1. Clean reference
        generate_clean(digits, output_dir)
        total += 1

        # 2. White noise at every SNR level
        for snr in snr_levels:
            generate_noisy(digits, snr, noise_type="white",
                           add_hum=False, output_dir=output_dir)
            total += 1

        # 3. High-frequency hiss at every SNR level
        for snr in snr_levels:
            generate_noisy(digits, snr, noise_type="hiss",
                           add_hum=False, output_dir=output_dir)
            total += 1

        # 4. White noise + 60 Hz hum at every SNR level
        for snr in snr_levels:
            generate_noisy(digits, snr, noise_type="white",
                           add_hum=True, output_dir=output_dir)
            total += 1

    print("\n" + "=" * 60)
    print(f" Done. {total} files written to '{output_dir}/'")
    print("=" * 60)


# ── Metrics helper ────────────────────────────────────────────────────────────

def measure_file_snr(clean_file: str, noisy_file: str) -> dict:
    """
    Measure SNR (dB) between a clean and a noisy WAV.
    Returns a dict with keys: snr_db, rms_signal, rms_noise.
    """
    clean, sr1 = load_wav(clean_file)
    noisy, sr2 = load_wav(noisy_file)
    assert sr1 == sr2, "Sample rates differ."
    min_len   = min(len(clean), len(noisy))
    signal    = clean[:min_len]
    noise     = noisy[:min_len] - signal
    snr       = compute_snr_db(signal, noise)
    return {
        "snr_db":     snr,
        "rms_signal": float(np.sqrt(np.mean(signal ** 2))),
        "rms_noise":  float(np.sqrt(np.mean(noise  ** 2))),
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate DTMF test WAV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--digits",  default="5551234",
                   help="Digit string to encode (default: 5551234)")
    p.add_argument("--snr",     type=int, default=None,
                   help="Target SNR in dB; omit for clean output")
    p.add_argument("--clean",   action="store_true",
                   help="Generate clean (no noise) file regardless of --snr")
    p.add_argument("--noise",   choices=["white", "hiss"], default="white",
                   help="Noise type (default: white)")
    p.add_argument("--hum",     action="store_true",
                   help="Also inject 60 Hz mains hum")
    p.add_argument("--suite",   action="store_true",
                   help="Build the full standard test suite and exit")
    p.add_argument("--out",     default=OUTPUT_DIR,
                   help=f"Output directory (default: {OUTPUT_DIR})")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.suite:
        build_test_suite(args.out)
        return

    if args.clean or args.snr is None:
        generate_clean(args.digits, args.out)
    else:
        generate_noisy(args.digits, args.snr,
                       noise_type=args.noise,
                       add_hum=args.hum,
                       output_dir=args.out)


if __name__ == "__main__":
    main()