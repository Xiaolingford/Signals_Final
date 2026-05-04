"""
DTMF Decoder Core
=================

Detects the 16 DTMF digits (0-9, *, #, A-D) using the Goertzel algorithm,
which is a single-bin DFT — much cheaper than a full FFT when you only
care about 8 specific frequencies.

The 8 DTMF frequencies form a 4x4 grid:

                1209 Hz   1336 Hz   1477 Hz   1633 Hz
        697 Hz    1         2         3         A
        770 Hz    4         5         6         B
        852 Hz    7         8         9         C
        941 Hz    *         0         #         D

A digit is the SUM of one row tone + one column tone. We look at the
strongest peak in each band, then validate it before emitting a digit.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Frequency table (ITU-T Q.23 / Q.24)
# ---------------------------------------------------------------------------
LOW_FREQS  = [697, 770, 852, 941]    # row frequencies
HIGH_FREQS = [1209, 1336, 1477, 1633]  # column frequencies

DIGIT_TABLE = {
    (697, 1209): '1', (697, 1336): '2', (697, 1477): '3', (697, 1633): 'A',
    (770, 1209): '4', (770, 1336): '5', (770, 1477): '6', (770, 1633): 'B',
    (852, 1209): '7', (852, 1336): '8', (852, 1477): '9', (852, 1633): 'C',
    (941, 1209): '*', (941, 1336): '0', (941, 1477): '#', (941, 1633): 'D',
}


# ---------------------------------------------------------------------------
# Goertzel algorithm
# ---------------------------------------------------------------------------
def goertzel_magnitude(samples, target_freq, sample_rate):
    """
    Return the magnitude-squared of `target_freq` inside `samples`.

    The Goertzel recurrence:
        s[n] = x[n] + 2*cos(omega)*s[n-1] - s[n-2]
    After feeding the whole block, the magnitude squared is:
        |X(k)|^2 = s_prev^2 + s_prev2^2 - coeff * s_prev * s_prev2
    """
    N = len(samples)
    # Use the continuous-frequency form (no bin rounding) so DTMF tones
    # don't have to land exactly on a DFT bin — important for short blocks
    # where bin spacing (fs/N) is wider than the tone-to-bin error.
    omega = 2.0 * np.pi * target_freq / sample_rate
    coeff = 2.0 * np.cos(omega)

    s_prev = 0.0
    s_prev2 = 0.0
    for x in samples:
        s = float(x) + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s

    return s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2


def goertzel_bank(samples, frequencies, sample_rate):
    """Run Goertzel for a list of frequencies, return numpy array of powers."""
    return np.array(
        [goertzel_magnitude(samples, f, sample_rate) for f in frequencies],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Peak detection + digit lookup
# ---------------------------------------------------------------------------
def detect_digit(samples,
                 sample_rate,
                 min_power=1e3,
                 dominance_ratio=4.0,
                 max_twist_db=8.0):
    """
    Decode a single block of audio.

    Returns
    -------
    (digit, low_powers, high_powers)
        digit          : the detected character, or None
        low_powers     : numpy array of 4 row-band powers
        high_powers    : numpy array of 4 column-band powers

    Validation gates (all must pass):

    1. Power gate    — peaks in BOTH bands above `min_power` (rejects silence)
    2. Dominance     — winning peak must be `dominance_ratio` x stronger than
                       the next strongest tone in its own band (rejects noise
                       and music, which spreads energy across multiple bins)
    3. Twist         — |10*log10(low/high)| <= max_twist_db
                       (real DTMF generators keep the two tones within ~8 dB)
    """
    low_powers  = goertzel_bank(samples, LOW_FREQS,  sample_rate)
    high_powers = goertzel_bank(samples, HIGH_FREQS, sample_rate)

    low_idx  = int(np.argmax(low_powers))
    high_idx = int(np.argmax(high_powers))
    low_peak  = low_powers[low_idx]
    high_peak = high_powers[high_idx]

    # Gate 1: Power threshold
    if low_peak < min_power or high_peak < min_power:
        return None, low_powers, high_powers

    # Gate 2: Dominance — peak vs. next-strongest in same band
    other_low  = np.delete(low_powers,  low_idx).max()
    other_high = np.delete(high_powers, high_idx).max()
    if other_low > 0 and low_peak < other_low * dominance_ratio:
        return None, low_powers, high_powers
    if other_high > 0 and high_peak < other_high * dominance_ratio:
        return None, low_powers, high_powers

    # Gate 3: Twist (low/high balance)
    twist_db = 10.0 * np.log10(low_peak / high_peak)
    if abs(twist_db) > max_twist_db:
        return None, low_powers, high_powers

    digit = DIGIT_TABLE.get((LOW_FREQS[low_idx], HIGH_FREQS[high_idx]))
    return digit, low_powers, high_powers

