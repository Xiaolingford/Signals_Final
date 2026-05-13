"""
equalizer.py
============
Real-Time Voice Denoiser — Signals & Systems demo.
macOS-compatible: uses SEPARATE input and output streams,
supports device selection, and resamples internally if needed.

Modes:
    python equalizer.py                              # mic -> filtered -> speakers
    python equalizer.py --file <wav>                 # play WAV through filters
    python equalizer.py --list-devices               # show device indices
    python equalizer.py --device-out 2               # pick output device
    python equalizer.py --device-in 1 --device-out 2
    python equalizer.py --fs 44100                   # force native sample rate

Filters (toggle live):
    [Notch 60 Hz]       IIR notch cascade (60, 120, 180 Hz) — kills mains hum
    [Low-pass]          FIR Hamming-windowed sinc — kills hiss above speech
    [Spectral Sub]      STFT spectral subtraction with Wiener-style smoothing
                        — kills broadband fan / HVAC noise

Calibration:
    Click [Calibrate Noise] while only background noise is present (~2 s).

S&S concepts demonstrated:
    IIR notch cascade, FIR low-pass via windowing, STFT spectral subtraction,
    Wiener gain, FFT / spectrogram, overlap-add reconstruction.
"""

import argparse
import os
import queue
import sys
import threading
import time
import numpy as np
import sounddevice as sd
import scipy.signal as signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation

# Resolve paths relative to THIS script (works regardless of CWD)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Config  (overridable via CLI)
# ---------------------------------------------------------------------------
FS_DEFAULT  = 16000     # wideband speech (was 8 kHz telephony)
BLOCK       = 1024      # larger block = more stable on macOS Core Audio
SPEC_SECONDS = 4
SPEC_NFFT   = 1024

# 60 Hz hum + 2nd and 3rd harmonics (real mains interference has these)
NOTCH_FREQS = [60.0, 120.0, 180.0]
NOTCH_Q     = 30.0

# Low-pass: speech intelligibility lives < 7 kHz; cut above that for hiss
LP_CUTOFF   = 7000.0
LP_TAPS     = 101

# Spectral subtraction
SS_ALPHA    = 2.0       # over-subtraction factor
SS_BETA     = 0.02      # spectral floor (prevents negative magnitudes)
SS_SMOOTH   = 0.7       # temporal smoothing of the Wiener gain (0..1)
SS_CAL_SECS = 2.0


# ---------------------------------------------------------------------------
# SpectralSubtractor  (Wiener-style, sqrt-Hanning overlap-add STFT)
# ---------------------------------------------------------------------------
class SpectralSubtractor:
    """
    STFT spectral subtraction with TEMPORAL SMOOTHING of the gain.
    hop must equal nfft // 2 (50% overlap) for sqrt-Hanning COLA to hold.

    For each frame:
        SNR_post = |X(k)|^2 / |N(k)|^2
        gain     = SNR_post / (1 + SNR_post)         (Wiener)
        gain     = smooth_prev*gain_prev + (1-smooth)*gain
        gain     = max(gain, beta)                    (spectral floor)
        Y(k)     = gain * X(k)

    The temporal smoothing kills "musical noise" — the tinkly random tones
    that plain magnitude subtraction leaves behind — which is much more
    obvious on speech than on tonal signals.
    """

    def __init__(self, nfft, hop, alpha=SS_ALPHA, beta=SS_BETA, smooth=SS_SMOOTH):
        assert hop == nfft // 2
        self.nfft   = nfft
        self.hop    = hop
        self.alpha  = alpha
        self.beta   = beta
        self.smooth = smooth
        self.nbins  = nfft // 2 + 1
        self.win    = np.sqrt(np.hanning(nfft)).astype(np.float32)

        self.noise_mag  = np.zeros(self.nbins, dtype=np.float32)
        self.calibrated = False
        self.enabled    = False

        self._in_buf    = np.zeros(nfft, dtype=np.float32)
        self._out_buf   = np.zeros(nfft, dtype=np.float32)
        self._gain_prev = np.ones(self.nbins, dtype=np.float32)

    def calibrate(self, noise_frames):
        """Average magnitude spectrum over ~2 s of noise-only audio."""
        mags = []
        buf  = np.zeros(self.nfft, dtype=np.float32)
        for blk in noise_frames:
            buf[:-self.hop] = buf[self.hop:]
            buf[-self.hop:] = blk[:self.hop]
            mags.append(np.abs(np.fft.rfft(buf * self.win)))
        if mags:
            self.noise_mag  = np.mean(mags, axis=0).astype(np.float32)
            self.calibrated = True
            # Reset smoothed gain
            self._gain_prev = np.ones(self.nbins, dtype=np.float32)

    def process(self, block):
        # Shift in `hop` new samples
        self._in_buf[:-self.hop] = self._in_buf[self.hop:]
        self._in_buf[-self.hop:] = block.astype(np.float32)

        if not self.enabled or not self.calibrated:
            return self._in_buf[:self.hop].copy()

        frame = self._in_buf * self.win
        X     = np.fft.rfft(frame)
        mag   = np.abs(X)
        phase = np.angle(X)

        # Wiener-style gain (better than hard subtraction for speech)
        noise_pow = self.alpha * (self.noise_mag ** 2) + 1e-12
        sig_pow   = mag ** 2
        snr_post  = sig_pow / noise_pow
        gain_raw  = snr_post / (1.0 + snr_post)

        # Temporal smoothing — kills musical noise
        gain      = self.smooth * self._gain_prev + (1 - self.smooth) * gain_raw
        self._gain_prev = gain

        # Spectral floor so we never zero a bin completely
        gain = np.maximum(gain, self.beta).astype(np.float32)

        mag_clean = mag * gain
        y_frame   = np.fft.irfft(mag_clean * np.exp(1j * phase)).real.astype(np.float32)
        y_frame  *= self.win

        # Overlap-add
        self._out_buf            += y_frame
        out                       = self._out_buf[:self.hop].copy()
        self._out_buf[:-self.hop] = self._out_buf[self.hop:]
        self._out_buf[-self.hop:] = 0.0
        return out


# ---------------------------------------------------------------------------
# Shared mutable state
# ---------------------------------------------------------------------------
state = {
    'notch'    : False,
    'lowpass'  : False,
    'spec_sub' : False,
    'cal_mode' : False,
    'recording': False,
}

# Notch cascade: list of (b, a, zi) — built in _init_filters()
_notch_stages: list = []
_lp_b = _lp_a = None
_lp_zi = None
ss: SpectralSubtractor | None = None

audio_q    = queue.Queue(maxsize=128)
cal_q      = queue.Queue(maxsize=2048)
_mic_out_q = queue.Queue(maxsize=32)
ring: np.ndarray = np.zeros(1, dtype=np.float32)  # resized in run()

# Recording buffers — list.append is atomic in CPython, safe across threads
_rec_raw_blocks:  list[np.ndarray] = []
_rec_proc_blocks: list[np.ndarray] = []


def save_recording(fs: int) -> tuple[str | None, str | None, float]:
    """
    Concatenate accumulated recording blocks, normalise, save TWO WAV files
    (raw input + processed output) with a timestamp, then clear buffers.
    Returns (raw_path, processed_path, duration_seconds) or (None, None, 0).
    """
    if not _rec_raw_blocks:
        return None, None, 0.0

    raw  = np.concatenate(_rec_raw_blocks).astype(np.float32)
    proc = np.concatenate(_rec_proc_blocks).astype(np.float32)
    _rec_raw_blocks.clear()
    _rec_proc_blocks.clear()

    def to_int16(x: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(x))) + 1e-12
        if peak > 0.95:
            x = x * (0.95 / peak)
        return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)

    ts        = time.strftime('%Y%m%d_%H%M%S')
    raw_path  = os.path.join(SCRIPT_DIR, f'recording_{ts}_raw.wav')
    proc_path = os.path.join(SCRIPT_DIR, f'recording_{ts}_proc.wav')
    wavfile.write(raw_path,  fs, to_int16(raw))
    wavfile.write(proc_path, fs, to_int16(proc))
    return raw_path, proc_path, len(raw) / fs


def _init_filters(fs: int):
    global _notch_stages, _lp_b, _lp_a, _lp_zi, ss

    # Cascade a notch at each harmonic that fits below Nyquist
    _notch_stages = []
    for f0 in NOTCH_FREQS:
        if f0 < fs / 2 - 1.0:
            b, a = signal.iirnotch(f0, NOTCH_Q, fs=fs)
            zi   = signal.lfilter_zi(b, a) * 0.0
            _notch_stages.append([b, a, zi])

    lp_f = min(LP_CUTOFF, fs / 2 - 100.0)
    _lp_b  = signal.firwin(LP_TAPS, lp_f, fs=fs, window='hamming')
    _lp_a  = np.array([1.0])
    _lp_zi = signal.lfilter_zi(_lp_b, _lp_a) * 0.0

    # Choose SS NFFT so that hop (= nfft//2) <= BLOCK
    nfft = SPEC_NFFT
    while nfft // 2 > BLOCK:
        nfft //= 2
    ss = SpectralSubtractor(nfft=nfft, hop=nfft // 2,
                            alpha=SS_ALPHA, beta=SS_BETA, smooth=SS_SMOOTH)


# ---------------------------------------------------------------------------
# DSP pipeline (called from audio thread — keep it fast)
# ---------------------------------------------------------------------------
def process(x: np.ndarray) -> np.ndarray:
    global _lp_zi
    y = x.astype(np.float32, copy=True)

    if state['spec_sub'] and ss is not None:
        hop = ss.hop
        # Caller guarantees len(y) is a multiple of hop (BLOCK is set that way)
        assert len(y) % hop == 0, f'block length {len(y)} not divisible by hop {hop}'
        out = np.empty_like(y)
        for i in range(0, len(y), hop):
            out[i:i + hop] = ss.process(y[i:i + hop])
        y = out

    if state['notch'] and _notch_stages:
        # Cascade: pass through each (b, a) stage in turn, preserving state
        for stage in _notch_stages:
            b, a, zi = stage
            y, stage[2] = signal.lfilter(b, a, y, zi=zi)

    if state['lowpass'] and _lp_b is not None:
        y, _lp_zi = signal.lfilter(_lp_b, _lp_a, y, zi=_lp_zi)

    if state['cal_mode']:
        try:
            cal_q.put_nowait(x.copy())
        except queue.Full:
            pass

    return y


# ---------------------------------------------------------------------------
# Audio callbacks
# ---------------------------------------------------------------------------

# --- Microphone: separate IN and OUT streams (macOS fix) ---
def _mic_in_cb(indata, frames, time_info, status):
    if status:
        print(f"[mic-in]  {status}", file=sys.stderr)
    raw = indata[:, 0].copy()       # indata buffer gets reused — must copy
    y   = process(raw)
    if state['recording']:
        _rec_raw_blocks.append(raw)
        _rec_proc_blocks.append(y.copy())
    try:
        _mic_out_q.put_nowait(y)
    except queue.Full:
        pass
    try:
        audio_q.put_nowait(y.copy())
    except queue.Full:
        pass

def _mic_out_cb(outdata, frames, time_info, status):
    if status:
        print(f"[mic-out] {status}", file=sys.stderr)
    try:
        y = _mic_out_q.get_nowait()
        if len(y) < frames:
            y = np.pad(y, (0, frames - len(y)))
        outdata[:, 0] = y[:frames]
    except queue.Empty:
        outdata[:, 0] = 0.0

# --- File playback ---
def make_file_cb(data: np.ndarray):
    idx = [0]
    def cb(outdata, frames, time_info, status):
        if status:
            print(f"[file] {status}", file=sys.stderr)
        i   = idx[0]
        end = i + frames
        blk = data[i:end]
        if len(blk) < frames:
            blk    = np.concatenate([blk, data[:frames - len(blk)]])
            idx[0] = frames - len(data[i:end])
        else:
            idx[0] = end % len(data)
        raw_in = blk.copy()
        y = process(blk)
        outdata[:, 0] = y
        if state['recording']:
            _rec_raw_blocks.append(raw_in)
            _rec_proc_blocks.append(y.copy())
        try:
            audio_q.put_nowait(y.copy())
        except queue.Full:
            pass
    return cb


# ---------------------------------------------------------------------------
# Calibration thread
# ---------------------------------------------------------------------------
_cal_thread = None

def _run_cal(n_blocks, done_cb):
    blocks = []
    while len(blocks) < n_blocks:
        try:
            blocks.append(cal_q.get(timeout=0.5))
        except queue.Empty:
            if not state['cal_mode']:
                break
    state['cal_mode'] = False
    if blocks and ss is not None:
        ss.calibrate(blocks)
    done_cb()

def start_calibration(n_blocks, done_cb):
    global _cal_thread
    while not cal_q.empty():
        cal_q.get_nowait()
    state['cal_mode'] = True
    _cal_thread = threading.Thread(target=_run_cal,
                                   args=(n_blocks, done_cb), daemon=True)
    _cal_thread.start()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def rough_snr_db(x, fs):
    """Crude speech-band SNR estimate (200–3000 Hz speech vs > 3500 Hz noise)."""
    X   = np.abs(np.fft.rfft(x)) ** 2
    f   = np.fft.rfftfreq(len(x), 1 / fs)
    sig = X[(f >= 200) & (f <= 3000)].sum() + 1e-12
    noi = X[f > 3500].sum() + 1e-12
    return 10 * np.log10(sig / noi)

def power_at(x, freq, fs, bw=5.0):
    X   = np.abs(np.fft.rfft(x)) ** 2
    f   = np.fft.rfftfreq(len(x), 1 / fs)
    sel = (f >= freq - bw) & (f <= freq + bw)
    return 10 * np.log10(X[sel].sum() + 1e-12)


# ---------------------------------------------------------------------------
# Pitch / note detection
# ---------------------------------------------------------------------------
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def detect_pitch_autocorr(x: np.ndarray, fs: int,
                          fmin: float = 70.0, fmax: float = 800.0,
                          rms_thresh: float = 0.005,
                          peak_ratio_thresh: float = 0.3) -> float | None:
    """
    Estimate fundamental frequency via autocorrelation.

    The autocorrelation r[k] of a periodic signal has a peak at the lag k
    equal to its period; freq = fs / k.  We search the lag range
    corresponding to [fmin, fmax] (typical voice fundamentals).

    Returns Hz, or None if the signal is too quiet OR the autocorrelation
    peak isn't sharp enough (i.e. no clear pitch — could be noise/silence).
    """
    x = x.astype(np.float32)
    x = x - np.mean(x)

    # Silence / too-quiet check
    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms < rms_thresh:
        return None

    # Full autocorrelation, then drop negative lags (symmetric anyway)
    corr = np.correlate(x, x, mode='full')
    corr = corr[len(corr) // 2:]

    # Lag bounds for the target pitch range
    lag_min = max(1, int(fs / fmax))
    lag_max = min(len(corr) - 1, int(fs / fmin))
    if lag_max <= lag_min:
        return None

    # Find max-correlation lag in the valid pitch range
    region    = corr[lag_min:lag_max]
    peak_idx  = int(np.argmax(region)) + lag_min

    # Confidence: peak should be a meaningful fraction of zero-lag energy.
    # Random noise has a flat autocorrelation past lag 0 (no clear peak).
    if corr[peak_idx] / (corr[0] + 1e-12) < peak_ratio_thresh:
        return None

    # Parabolic interpolation around the peak — sub-sample lag accuracy,
    # which matters because Hz = fs / lag is sensitive at small lags.
    if 1 <= peak_idx < len(corr) - 1:
        a, b, c = float(corr[peak_idx - 1]), float(corr[peak_idx]), float(corr[peak_idx + 1])
        denom = (a - 2 * b + c)
        if abs(denom) > 1e-12:
            offset    = 0.5 * (a - c) / denom
            peak_idx  = peak_idx + offset

    if peak_idx <= 0:
        return None
    return fs / peak_idx


def freq_to_note(freq: float | None) -> tuple[str, float]:
    """
    Convert a frequency in Hz to (note_name, cents_deviation).
    Cents: how far off from a perfectly-tuned note (-50..+50; 0 = on pitch).
    A4 = 440 Hz reference, midi 69.
    """
    if freq is None or freq <= 0:
        return '—', 0.0
    midi       = 12.0 * np.log2(freq / 440.0) + 69.0
    midi_round = int(round(midi))
    cents      = (midi - midi_round) * 100.0
    name       = NOTE_NAMES[midi_round % 12]
    octave     = midi_round // 12 - 1
    return f'{name}{octave}', cents


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_ui(fs: int):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 8))
    fig.canvas.manager.set_window_title('Real-Time Voice Denoiser — S&S Demo')

    gs = fig.add_gridspec(3, 6, height_ratios=[4, 1.4, 0.65],
                          width_ratios=[1, 1, 1, 1, 1, 2.2],
                          hspace=0.45, wspace=0.3)

    ax_spec = fig.add_subplot(gs[0, :])
    im = ax_spec.imshow(
        np.full((SPEC_NFFT // 2 + 1, 50), -100.0),
        aspect='auto', origin='lower',
        extent=[0, SPEC_SECONDS, 0, fs / 2],
        vmin=-80, vmax=-10, cmap='magma', interpolation='nearest',
    )
    ax_spec.set_xlabel('Time (s)')
    ax_spec.set_ylabel('Frequency (Hz)')
    ax_spec.set_title('Live Spectrogram  —  toggle filters below')
    for f0 in NOTCH_FREQS:
        if f0 < fs / 2:
            ax_spec.axhline(f0, color='cyan', alpha=0.3, lw=0.6, ls='--')
    if LP_CUTOFF < fs / 2:
        ax_spec.axhline(LP_CUTOFF, color='lime', alpha=0.4, lw=0.8, ls='--')
    fig.colorbar(im, ax=ax_spec, label='dB')

    # Big live note read-out in the top-right of the spectrogram.
    # Detection happens in the animation update(); this text just displays it.
    note_text = ax_spec.text(
        0.985, 0.94, '',
        ha='right', va='top',
        fontsize=22, color='#88ff88', weight='bold', family='monospace',
        transform=ax_spec.transAxes,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#000000',
                  alpha=0.55, edgecolor='#88ff88', linewidth=1.2),
    )

    ax_wave = fig.add_subplot(gs[1, :])
    (line,) = ax_wave.plot(np.zeros(fs), color='cyan', lw=0.6)
    ax_wave.set_ylim(-1, 1)
    ax_wave.set_xlim(0, fs)
    ax_wave.set_title('Waveform  (last 1 s)')
    ax_wave.set_xticks([])

    ax_b1 = fig.add_subplot(gs[2, 0])
    ax_b2 = fig.add_subplot(gs[2, 1])
    ax_b3 = fig.add_subplot(gs[2, 2])
    ax_b4 = fig.add_subplot(gs[2, 3])
    ax_b5 = fig.add_subplot(gs[2, 4])
    ax_m  = fig.add_subplot(gs[2, 5]); ax_m.axis('off')

    mt = ax_m.text(0.0, 0.5, '', ha='left', va='center',
                   fontsize=10, family='monospace', color='white',
                   transform=ax_m.transAxes)

    b_notch = Button(ax_b1, 'Hum Notches: OFF',  color='#2a2a2a', hovercolor='#444')
    b_lp    = Button(ax_b2, 'Low-pass: OFF',      color='#2a2a2a', hovercolor='#444')
    b_ss    = Button(ax_b3, 'Spectral Sub: OFF',  color='#2a2a2a', hovercolor='#444')
    b_cal   = Button(ax_b4, 'Calibrate Noise',    color='#1a3a5c', hovercolor='#2456a4')
    b_rec   = Button(ax_b5, 'Record: OFF',        color='#2a2a2a', hovercolor='#444')
    for b in (b_notch, b_lp, b_ss, b_cal, b_rec):
        b.label.set_fontsize(10)

    cal_st = ax_b4.text(0.5, -0.45, '', ha='center', va='top',
                        fontsize=8, color='#88ccff',
                        transform=ax_b4.transAxes)
    rec_st = ax_b5.text(0.5, -0.45, '', ha='center', va='top',
                        fontsize=8, color='#ff9999',
                        transform=ax_b5.transAxes)

    def refresh():
        b_notch.label.set_text(f"Hum Notches: {'ON' if state['notch'] else 'OFF'}")
        b_lp.label.set_text(   f"Low-pass: {'ON' if state['lowpass'] else 'OFF'}")
        b_ss.label.set_text(   f"Spectral Sub: {'ON' if state['spec_sub'] else 'OFF'}")
        # Also sync SS enabled flag so the class actually does work
        if ss is not None:
            ss.enabled = state['spec_sub']
        b_notch.color = '#1e6b1e' if state['notch']    else '#2a2a2a'
        b_lp.color    = '#1e6b1e' if state['lowpass']  else '#2a2a2a'
        b_ss.color    = '#4a2080' if state['spec_sub'] else '#2a2a2a'
        fig.canvas.draw_idle()

    def on_cal(_e):
        if state['cal_mode']:
            return
        n_blocks = int(SS_CAL_SECS * fs / BLOCK)
        cal_st.set_text(f'Recording {SS_CAL_SECS:.0f}s…')
        b_cal.color = '#7a4400'
        fig.canvas.draw_idle()
        def done():
            cal_st.set_text('Calibrated ✓' if (ss and ss.calibrated) else 'Failed')
            b_cal.color = '#1a3a5c'
            fig.canvas.draw_idle()
        start_calibration(n_blocks, done)

    def on_rec(_e):
        if state['recording']:
            # Stop and save
            state['recording'] = False
            raw_path, proc_path, dur = save_recording(fs)
            b_rec.label.set_text('Record: OFF')
            b_rec.color = '#2a2a2a'
            if raw_path:
                rec_st.set_text(f'Saved {dur:.1f}s ✓')
                print(f'[record] saved {dur:.1f} s:')
                print(f'  {raw_path}')
                print(f'  {proc_path}')
            else:
                rec_st.set_text('Nothing recorded')
        else:
            # Start fresh recording
            _rec_raw_blocks.clear()
            _rec_proc_blocks.clear()
            state['recording'] = True
            b_rec.label.set_text('Record: REC ●')
            b_rec.color = '#7a1a1a'
            rec_st.set_text('Recording…')
            print('[record] started — click Record again to stop & save')
        fig.canvas.draw_idle()

    b_notch.on_clicked(lambda e: (state.update(notch=not state['notch']),    refresh()))
    b_lp.on_clicked(   lambda e: (state.update(lowpass=not state['lowpass']),refresh()))
    b_ss.on_clicked(   lambda e: (state.update(spec_sub=not state['spec_sub']),refresh()))
    b_cal.on_clicked(on_cal)
    b_rec.on_clicked(on_rec)

    fig._buttons = (b_notch, b_lp, b_ss, b_cal, b_rec)
    return fig, im, line, mt, note_text

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(source='mic', wav_path=None, device_in=None, device_out=None,
        fs_override=None):
    global ring

    fs = fs_override or FS_DEFAULT

    # Auto-detect native rate from chosen output device
    if device_out is not None and fs_override is None:
        info = sd.query_devices(device_out)
        native = int(info.get('default_samplerate', fs))
        if native != fs:
            print(f"[audio] Device native rate {native} Hz → using that (was {fs} Hz).")
            fs = native
    elif fs_override is None:
        _, default_out_idx = sd.default.device
        if default_out_idx is not None and default_out_idx >= 0:
            info = sd.query_devices(default_out_idx)
            native = int(info.get('default_samplerate', fs))
            if native != fs:
                print(f"[audio] Default output native rate {native} Hz → using that.")
                fs = native

    _init_filters(fs)
    ring = np.zeros(fs * SPEC_SECONDS, dtype=np.float32)

    fig, im, line, mt, note_text = build_ui(fs)
    streams = []

    if source == 'file':
        fs_in, data = wavfile.read(wav_path)
        data = data.astype(np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if data.max() > 1.5:
            data /= 32767.0
        if fs_in != fs:
            print(f'[audio] Resampling file {fs_in} → {fs} Hz')
            data = signal.resample_poly(data, fs, fs_in).astype(np.float32)
        streams.append(sd.OutputStream(
            samplerate=fs, blocksize=BLOCK, channels=1,
            device=device_out, callback=make_file_cb(data),
        ))

    else:
        # Two separate streams — the macOS-safe approach
        streams.append(sd.InputStream(
            samplerate=fs, blocksize=BLOCK, channels=1,
            device=device_in, callback=_mic_in_cb,
        ))
        streams.append(sd.OutputStream(
            samplerate=fs, blocksize=BLOCK, channels=1,
            device=device_out, callback=_mic_out_cb,
        ))

    def update(_frame):
        global ring
        while not audio_q.empty():
            blk  = audio_q.get_nowait()
            n    = len(blk)
            ring = np.roll(ring, -n)
            ring[-n:] = blk

        f_s, _t, Sxx = signal.spectrogram(
            ring, fs=fs, nperseg=SPEC_NFFT,
            noverlap=SPEC_NFFT // 2, scaling='spectrum',
        )
        im.set_data(10 * np.log10(Sxx + 1e-12))
        im.set_extent([0, SPEC_SECONDS, 0, fs / 2])

        line.set_ydata(ring[-fs:])
        line.set_xdata(np.arange(fs))

        recent = ring[-fs:]

        # Live pitch detection on the most recent ~200 ms of audio.
        # Short window = responsive; autocorrelation handles 70–800 Hz well.
        pitch_window = ring[-fs // 5:] if len(ring) >= fs // 5 else ring
        pitch        = detect_pitch_autocorr(pitch_window, fs)
        note, cents  = freq_to_note(pitch)

        if pitch is not None:
            # Color-code: green if within ±15¢ of pitch, yellow if off
            colour = '#88ff88' if abs(cents) < 15 else '#ffdd66'
            note_text.set_color(colour)
            note_text.get_bbox_patch().set_edgecolor(colour)
            note_text.set_text(f'{note}  {cents:+.0f}¢')
            pitch_str = f'{pitch:6.1f} Hz'
            note_str  = f'{note} {cents:+.0f}¢'
        else:
            note_text.set_text('')
            pitch_str = '   —'
            note_str  = '       —'

        cal = '✓' if (ss and ss.calibrated) else '–'
        mt.set_text(
            f"Note:           {note_str}\n"
            f"Pitch:          {pitch_str}\n"
            f"SNR (band):     {rough_snr_db(recent, fs):5.1f} dB\n"
            f"60 Hz power:    {power_at(recent, 60.0, fs):5.1f} dB\n"
            f"Calibrated:     {cal}\n"
            f"FS: {fs} Hz"
        )
        return im, line, mt, note_text

    ani = FuncAnimation(fig, update, interval=80, blit=False, cache_frame_data=False)
    fig._ani = ani

    for s in streams:
        s.start()
    print(f"[audio] Started — {fs} Hz, block={BLOCK}. Close window to stop.")
    try:
        plt.show()
    finally:
        for s in streams:
            s.stop(); s.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Real-time voice denoiser — macOS-compatible')
    p.add_argument('--file',         default=None,        help='WAV file (omit for mic)')
    p.add_argument('--device-in',    type=int, default=None, help='Input device index')
    p.add_argument('--device-out',   type=int, default=None, help='Output device index')
    p.add_argument('--fs',           type=int, default=None, help='Force sample rate')
    p.add_argument('--list-devices', action='store_true',   help='Print devices and exit')
    p.add_argument('--alpha',        type=float, default=SS_ALPHA)
    p.add_argument('--beta',         type=float, default=SS_BETA)
    p.add_argument('--smooth',       type=float, default=SS_SMOOTH,
                   help='Wiener-gain temporal smoothing (0..1, higher = smoother)')
    args = p.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        sys.exit(0)

    SS_ALPHA  = args.alpha
    SS_BETA   = args.beta
    SS_SMOOTH = args.smooth

    if args.file is None:
        # Quick mic permission check on macOS
        try:
            t = sd.InputStream(channels=1, samplerate=44100, blocksize=1024)
            t.start(); t.stop(); t.close()
        except Exception as e:
            print(f"\nERROR opening microphone: {e}")
            print("macOS fix:  System Settings → Privacy & Security → Microphone → Terminal → ON")
            sys.exit(1)

    run(source='file' if args.file else 'mic',
        wav_path=args.file,
        device_in=args.device_in,
        device_out=args.device_out,
        fs_override=args.fs)