"""
noise_inject.py
===============
Standalone noise-injection library.

Functions are designed to be imported by generate_dtmf.py and by the
live equalizer demo (real_time_eq.py).  Can also be run directly for
a quick sanity check.

Noise catalogue
---------------
  white       — AWGN (broadband, flat spectrum)
  hiss        — high-frequency hiss (>4 kHz shelf)
  hum60       — 60 Hz power-line sinusoidal hum
  hum50       — 50 Hz power-line sinusoidal hum  (European variant)
  chatter     — band-pass coloured noise (300–3400 Hz, simulates crowd)
  mixed       — white + 60 Hz hum + chatter (worst-case demo scenario)

Usage (standalone):
    python noise_inject.py --input test_wavs/test_911_clean.wav \
                           --snr 10 --type mixed --hum
"""

import argparse
import os
import wave

import numpy as np

# ── WAV helpers (duplicated here so module is self-contained) ─────────────────

SAMPLE_RATE = 8000


def _load_wav(filename: str) -> tuple[np.ndarray, int]:
    with wave.open(filename, "r") as wf:
        sr  = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32767, sr


def _save_wav(filename: str, signal: np.ndarray,
              sample_rate: int = SAMPLE_RATE) -> None:
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    pcm = (np.clip(signal, -1, 1) * 32767).astype(np.int16).tobytes()
    with wave.open(filename, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    print(f"  Saved → {filename}")


# ── SNR utilities ─────────────────────────────────────────────────────────────

def rms(x: np.ndarray) -> float:
    """Root-mean-square of array."""
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    SNR = 10·log10(P_signal / P_noise).

    Both arrays must be the same length.  Signal and noise must be
    provided as SEPARATE arrays (i.e. noise is NOT the mixed signal).
    """
    p_signal = np.mean(signal.astype(np.float64) ** 2)
    p_noise  = np.mean(noise.astype(np.float64)  ** 2)
    if p_noise == 0:
        return float("inf")
    return 10 * np.log10(p_signal / p_noise)


def scale_noise_to_snr(signal: np.ndarray, noise: np.ndarray,
                        target_snr_db: float) -> np.ndarray:
    """
    Scale *noise* so that SNR(signal, scaled_noise) == target_snr_db.

    Parameters
    ----------
    signal        : clean signal array
    noise         : noise array of arbitrary amplitude (same length or longer)
    target_snr_db : desired SNR in dB

    Returns
    -------
    scaled_noise  : ndarray, same length as *signal*
    """
    noise = noise[:len(signal)]
    p_signal = np.mean(signal.astype(np.float64) ** 2)
    p_noise  = np.mean(noise.astype(np.float64)  ** 2)
    if p_noise == 0 or p_signal == 0:
        return np.zeros(len(signal))
    target_linear = 10 ** (target_snr_db / 10)
    scale = np.sqrt(p_signal / (p_noise * target_linear))
    return noise[:len(signal)] * scale


# ── Individual noise generators ───────────────────────────────────────────────

def white_noise(n_samples: int, sample_rate: int = SAMPLE_RATE,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """Flat-spectrum Gaussian white noise, unit variance."""
    rng = rng or np.random.default_rng()
    return rng.standard_normal(n_samples)


def hiss_noise(n_samples: int, sample_rate: int = SAMPLE_RATE,
               rng: np.random.Generator | None = None,
               cutoff_hz: float = 4000) -> np.ndarray:
    """
    High-frequency hiss: white noise with energy only above *cutoff_hz*.
    Models tape hiss, microphone self-noise, or cooling-fan noise.
    """
    rng = rng or np.random.default_rng()
    raw      = rng.standard_normal(n_samples)
    freqs    = np.fft.rfftfreq(n_samples, 1 / sample_rate)
    spectrum = np.fft.rfft(raw)
    spectrum[freqs < cutoff_hz] = 0
    result = np.fft.irfft(spectrum, n=n_samples)
    p = np.mean(result ** 2)
    return result / np.sqrt(p) if p > 0 else result   # normalise to unit RMS


def hum_noise(n_samples: int, frequency: float = 60.0,
              sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Sinusoidal mains hum at *frequency* Hz (60 Hz USA, 50 Hz Europe).
    Returns unit-amplitude sinusoid — caller controls level via SNR scaling.
    """
    t = np.arange(n_samples) / sample_rate
    return np.sin(2 * np.pi * frequency * t)


def chatter_noise(n_samples: int, sample_rate: int = SAMPLE_RATE,
                  rng: np.random.Generator | None = None,
                  low_hz: float = 300, high_hz: float = 3400) -> np.ndarray:
    """
    Band-limited coloured noise in the telephone band (300–3400 Hz).
    Simulates background crowd chatter or music bleed.
    """
    rng = rng or np.random.default_rng()
    raw      = rng.standard_normal(n_samples)
    freqs    = np.fft.rfftfreq(n_samples, 1 / sample_rate)
    spectrum = np.fft.rfft(raw)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    spectrum[~mask] = 0
    result = np.fft.irfft(spectrum, n=n_samples)
    p = np.mean(result ** 2)
    return result / np.sqrt(p) if p > 0 else result


def mixed_noise(n_samples: int, sample_rate: int = SAMPLE_RATE,
                rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    white   = white_noise(n_samples, sample_rate, rng)
    hum     = hum_noise(n_samples, 60, sample_rate)
    chatter = chatter_noise(n_samples, sample_rate, rng=rng)
    # Normalise each component to unit RMS then sum
    def norm(x):
        p = np.mean(x ** 2)
        return x / np.sqrt(p) if p > 0 else x
    composite = norm(white) + norm(hum) + norm(chatter)
    p = np.mean(composite ** 2)
    return composite / np.sqrt(p) if p > 0 else composite


# ── Noise-type registry ───────────────────────────────────────────────────────

_NOISE_BUILDERS = {
    "white":   white_noise,
    "hiss":    hiss_noise,
    "hum60":   lambda n, sr, rng: hum_noise(n, 60.0, sr),
    "hum50":   lambda n, sr, rng: hum_noise(n, 50.0, sr),
    "chatter": chatter_noise,
    "mixed":   mixed_noise,
}

NOISE_TYPES = list(_NOISE_BUILDERS.keys())


def make_noise(noise_type: str, n_samples: int,
               sample_rate: int = SAMPLE_RATE,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Factory function: return a unit-RMS noise array of *n_samples*.

    Parameters
    ----------
    noise_type  : one of "white", "hiss", "hum60", "hum50", "chatter", "mixed"
    n_samples   : length of the output array
    sample_rate : Hz (required for frequency-domain generators)
    rng         : optional numpy random Generator
    """
    if noise_type not in _NOISE_BUILDERS:
        raise ValueError(
            f"Unknown noise type '{noise_type}'. "
            f"Choose from: {NOISE_TYPES}"
        )
    return _NOISE_BUILDERS[noise_type](n_samples, sample_rate, rng)


# ── Main injection function ───────────────────────────────────────────────────

def inject_noise(signal: np.ndarray, snr_target_db: float,
                 noise_type: str = "white",
                 sample_rate: int = SAMPLE_RATE,
                 add_60hz_hum: bool = False,
                 hum_snr_db: float = 10.0,
                 seed: int | None = None) -> tuple[np.ndarray, dict]:
    """
    Add noise of *noise_type* to *signal* at *snr_target_db*.

    Parameters
    ----------
    signal          : clean float64 signal array
    snr_target_db   : desired SNR in dB
    noise_type      : "white" | "hiss" | "hum60" | "hum50" | "chatter" | "mixed"
    sample_rate     : Hz
    add_60hz_hum    : add an extra 60 Hz hum component on top
    hum_snr_db      : SNR of the hum component (default 10 dB → audible)
    seed            : random seed for reproducibility

    Returns
    -------
    noisy   : float64 ndarray same length as *signal*
    metrics : dict with measured SNR, RMS values, and noise type
    """
    rng   = np.random.default_rng(seed)
    n     = len(signal)
    noise = make_noise(noise_type, n, sample_rate, rng)
    noise = scale_noise_to_snr(signal, noise, snr_target_db)

    noisy = signal + noise

    if add_60hz_hum:
        hum = hum_noise(n, 60.0, sample_rate)
        hum = scale_noise_to_snr(signal, hum, hum_snr_db)
        noisy = noisy + hum
        noise = noise + hum           # update noise estimate for metrics

    # Clamp to [-1, 1]
    noisy = np.clip(noisy, -1.0, 1.0)

    # Measure actual SNR (min-length match)
    min_len = min(len(signal), len(noisy))
    measured = snr_db(signal[:min_len], noise[:min_len])

    metrics = {
        "noise_type":         noise_type,
        "target_snr_db":      snr_target_db,
        "measured_snr_db":    round(measured, 2),
        "rms_signal":         round(rms(signal), 6),
        "rms_noise":          round(rms(noise[:min_len]),  6),
        "rms_noisy":          round(rms(noisy),  6),
        "add_60hz_hum":       add_60hz_hum,
        "n_samples":          n,
        "sample_rate_hz":     sample_rate,
    }
    return noisy, metrics


# ── File-level interface ──────────────────────────────────────────────────────

def inject_from_file(input_wav: str, output_wav: str,
                     snr_db_target: float,
                     noise_type: str = "white",
                     add_60hz_hum: bool = False,
                     seed: int = 42) -> dict:
    """
    Load *input_wav*, inject noise, save to *output_wav*.

    Returns the metrics dict from inject_noise().
    """
    signal, sr = _load_wav(input_wav)
    noisy, metrics = inject_noise(
        signal, snr_db_target,
        noise_type=noise_type,
        sample_rate=sr,
        add_60hz_hum=add_60hz_hum,
        seed=seed,
    )
    _save_wav(output_wav, noisy, sr)
    print(f"  Measured SNR = {metrics['measured_snr_db']:.1f} dB  "
          f"(target {snr_db_target} dB, type={noise_type})")
    return metrics


# ── Batch sweep ───────────────────────────────────────────────────────────────

def snr_sweep(input_wav: str, snr_levels: list[int],
              noise_types: list[str] | None = None,
              output_dir: str = "test_wavs",
              add_60hz_hum: bool = False) -> list[dict]:
    """
    Inject every combination of (snr_level × noise_type) for *input_wav*.

    Returns a list of metrics dicts (one per generated file).
    """
    noise_types = noise_types or ["white", "hiss"]
    results     = []
    base        = os.path.splitext(os.path.basename(input_wav))[0]

    os.makedirs(output_dir, exist_ok=True)
    for nt in noise_types:
        for snr in snr_levels:
            hum_suffix = "_hum" if add_60hz_hum else ""
            out = os.path.join(output_dir, f"{base}_{nt}_snr{snr:+d}{hum_suffix}.wav")
            m = inject_from_file(input_wav, out, snr,
                                 noise_type=nt, add_60hz_hum=add_60hz_hum)
            m["output_file"] = out
            results.append(m)
    return results


# ── Metrics report ────────────────────────────────────────────────────────────

def print_metrics_table(metrics_list: list[dict]) -> None:
    """Pretty-print a table of SNR measurements (for the report)."""
    header = f"{'File':<45} {'Type':<8} {'Target':>8} {'Actual':>8}"
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for m in metrics_list:
        fname = os.path.basename(m.get("output_file", ""))
        print(
            f"{fname:<45} {m['noise_type']:<8} "
            f"{m['target_snr_db']:>7.1f}  {m['measured_snr_db']:>7.1f}"
        )
    print("─" * len(header))


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inject noise into a WAV file.")
    p.add_argument("--input",  required=True, help="Clean input WAV")
    p.add_argument("--output", default=None,  help="Output path (auto if omitted)")
    p.add_argument("--snr",    type=float, required=True, help="Target SNR in dB")
    p.add_argument("--type",   choices=NOISE_TYPES, default="white",
                   help=f"Noise type (default: white). Choices: {NOISE_TYPES}")
    p.add_argument("--hum",    action="store_true",
                   help="Also inject 60 Hz mains hum")
    p.add_argument("--sweep",  action="store_true",
                   help="Sweep all SNR levels and noise types for the input file")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.sweep:
        snr_levels  = [30, 25, 20, 15, 10, 5, 0, -5]
        noise_types = NOISE_TYPES
        results = snr_sweep(args.input, snr_levels, noise_types,
                            add_60hz_hum=args.hum)
        print_metrics_table(results)
        return

    out = args.output
    if out is None:
        base = os.path.splitext(args.input)[0]
        out  = f"{base}_{args.type}_snr{int(args.snr):+d}.wav"

    inject_from_file(args.input, out, args.snr,
                     noise_type=args.type, add_60hz_hum=args.hum)


if __name__ == "__main__":
    main()