"""
equalizer.py
============
Real-Time Audio Equalizer / Noise Filter — Signals & Systems demo.
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
    [Notch 60 Hz]       IIR notch — kills mains hum
    [Low-pass]          FIR Hamming-windowed sinc — kills hiss
    [Spectral Sub]      Spectral subtraction — kills broadband fan noise

Calibration:
    Click [Calibrate Noise] while only fan noise is present (~2 s).

S&S concepts:
    IIR notch, FIR low-pass via windowing, STFT spectral subtraction,
    FFT / spectrogram, overlap-add reconstruction.
"""

import argparse
import queue
import sys
import threading
import numpy as np
import sounddevice as sd
import scipy.signal as signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation

# ---------------------------------------------------------------------------
# Config  (overridable via CLI)
# ---------------------------------------------------------------------------
FS_DEFAULT  = 8000
BLOCK       = 512      # larger block = more stable on macOS Core Audio
SPEC_SECONDS = 4
SPEC_NFFT   = 512

NOTCH_FREQ  = 60.0
NOTCH_Q     = 30.0
LP_CUTOFF   = 3400.0
LP_TAPS     = 101

SS_ALPHA    = 2.0
SS_BETA     = 0.02
SS_CAL_SECS = 2.0


# ---------------------------------------------------------------------------
# SpectralSubtractor  (sqrt-Hanning overlap-add STFT)
# ---------------------------------------------------------------------------
class SpectralSubtractor:
    """
    STFT spectral subtraction with perfect-reconstruction overlap-add.
    hop must equal nfft // 2 (50% overlap) for sqrt-Hanning COLA to hold.

    Subtraction:  |Y(k)| = max( |X(k)| - alpha*|N(k)|,  beta*|N(k)| )
    """

    def __init__(self, nfft, hop, alpha=SS_ALPHA, beta=SS_BETA):
        assert hop == nfft // 2
        self.nfft  = nfft
        self.hop   = hop
        self.alpha = alpha
        self.beta  = beta
        self.nbins = nfft // 2 + 1
        self.win   = np.sqrt(np.hanning(nfft)).astype(np.float32)

        self.noise_mag  = np.zeros(self.nbins, dtype=np.float32)
        self.calibrated = False
        self.enabled    = False

        self._in_buf  = np.zeros(nfft, dtype=np.float32)
        self._out_buf = np.zeros(nfft, dtype=np.float32)

    def calibrate(self, noise_frames):
        mags = []
        buf  = np.zeros(self.nfft, dtype=np.float32)
        for blk in noise_frames:
            buf[:-self.hop] = buf[self.hop:]
            buf[-self.hop:] = blk[:self.hop]
            mags.append(np.abs(np.fft.rfft(buf * self.win)))
        if mags:
            self.noise_mag  = np.mean(mags, axis=0).astype(np.float32)
            self.calibrated = True

    def process(self, block):
        self._in_buf[:-self.hop] = self._in_buf[self.hop:]
        self._in_buf[-self.hop:] = block.astype(np.float32)

        if not self.enabled or not self.calibrated:
            return self._in_buf[:self.hop].copy()

        frame     = self._in_buf * self.win
        X         = np.fft.rfft(frame)
        mag       = np.abs(X)
        phase     = np.angle(X)
        mag_clean = np.maximum(mag - self.alpha * self.noise_mag,
                               self.beta  * self.noise_mag)
        y_frame   = np.fft.irfft(mag_clean * np.exp(1j * phase)).real.astype(np.float32)
        y_frame  *= self.win

        self._out_buf       += y_frame
        out                  = self._out_buf[:self.hop].copy()
        self._out_buf[:-self.hop] = self._out_buf[self.hop:]
        self._out_buf[-self.hop:] = 0.0
        return out


# ---------------------------------------------------------------------------
# Shared mutable state
# ---------------------------------------------------------------------------
state = {
    'notch'   : False,
    'lowpass' : False,
    'spec_sub': False,
    'cal_mode': False,
}

# These are filled in by _init_filters() once the real FS is known
_notch_b = _notch_a = _lp_b = _lp_a = None
_notch_zi = _lp_zi = None
ss: SpectralSubtractor | None = None

audio_q    = queue.Queue(maxsize=128)
cal_q      = queue.Queue(maxsize=2048)
_mic_out_q = queue.Queue(maxsize=32)
ring: np.ndarray = np.zeros(1, dtype=np.float32)  # resized in run()


def _init_filters(fs: int):
    global _notch_b, _notch_a, _lp_b, _lp_a, _notch_zi, _lp_zi, ss

    notch_f = min(NOTCH_FREQ, fs / 2 - 1.0)
    lp_f    = min(LP_CUTOFF,  fs / 2 - 100.0)

    _notch_b, _notch_a = signal.iirnotch(notch_f, NOTCH_Q, fs=fs)
    _lp_b               = signal.firwin(LP_TAPS, lp_f, fs=fs, window='hamming')
    _lp_a               = np.array([1.0])
    _notch_zi           = signal.lfilter_zi(_notch_b, _notch_a) * 0.0
    _lp_zi              = signal.lfilter_zi(_lp_b,    _lp_a)    * 0.0

    # Choose SS NFFT so that hop (= nfft//2) <= BLOCK
    nfft = SPEC_NFFT
    while nfft // 2 > BLOCK:
        nfft //= 2
    ss = SpectralSubtractor(nfft=nfft, hop=nfft // 2,
                            alpha=SS_ALPHA, beta=SS_BETA)


# ---------------------------------------------------------------------------
# DSP pipeline (called from audio thread — keep it fast)
# ---------------------------------------------------------------------------
def process(x: np.ndarray) -> np.ndarray:
    global _notch_zi, _lp_zi
    y = x.astype(np.float32, copy=True)

    if state['spec_sub'] and ss is not None:
        hop = ss.hop
        out = np.empty_like(y)
        for i in range(0, len(y), hop):
            chunk = y[i:i + hop]
            pad   = hop - len(chunk)
            if pad:
                chunk = np.pad(chunk, (0, pad))
            out[i:i + hop] = ss.process(chunk)[:len(y[i:i + hop])]
        y = out

    if state['notch'] and _notch_b is not None:
        y, _notch_zi = signal.lfilter(_notch_b, _notch_a, y, zi=_notch_zi)

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
    y = process(indata[:, 0])
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
        y = process(blk)
        outdata[:, 0] = y
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
    X   = np.abs(np.fft.rfft(x)) ** 2
    f   = np.fft.rfftfreq(len(x), 1 / fs)
    sig = X[(f >= 200) & (f <= 2000)].sum() + 1e-12
    noi = X[f > 2500].sum() + 1e-12
    return 10 * np.log10(sig / noi)

def power_at(x, freq, fs, bw=5.0):
    X   = np.abs(np.fft.rfft(x)) ** 2
    f   = np.fft.rfftfreq(len(x), 1 / fs)
    sel = (f >= freq - bw) & (f <= freq + bw)
    return 10 * np.log10(X[sel].sum() + 1e-12)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_ui(fs: int):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 8))
    fig.canvas.manager.set_window_title('Real-Time Denoiser — S&S Demo')

    gs = fig.add_gridspec(3, 5, height_ratios=[4, 1.4, 0.65],
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
    ax_spec.axhline(min(60, fs/2-1),       color='cyan', alpha=0.4, lw=0.8, ls='--')
    ax_spec.axhline(min(LP_CUTOFF, fs/2-1),color='lime', alpha=0.4, lw=0.8, ls='--')
    fig.colorbar(im, ax=ax_spec, label='dB')

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
    ax_m  = fig.add_subplot(gs[2, 4]); ax_m.axis('off')

    mt = ax_m.text(0.0, 0.5, '', ha='left', va='center',
                   fontsize=10, family='monospace', color='white',
                   transform=ax_m.transAxes)

    b_notch = Button(ax_b1, 'Notch 60Hz: OFF',   color='#2a2a2a', hovercolor='#444')
    b_lp    = Button(ax_b2, 'Low-pass: OFF',      color='#2a2a2a', hovercolor='#444')
    b_ss    = Button(ax_b3, 'Spectral Sub: OFF',  color='#2a2a2a', hovercolor='#444')
    b_cal   = Button(ax_b4, 'Calibrate Noise',    color='#1a3a5c', hovercolor='#2456a4')
    for b in (b_notch, b_lp, b_ss, b_cal):
        b.label.set_fontsize(10)

    cal_st = ax_b4.text(0.5, -0.45, '', ha='center', va='top',
                        fontsize=8, color='#88ccff',
                        transform=ax_b4.transAxes)

    def refresh():
        b_notch.label.set_text(f"Notch 60Hz: {'ON' if state['notch'] else 'OFF'}")
        b_lp.label.set_text(   f"Low-pass: {'ON' if state['lowpass'] else 'OFF'}")
        b_ss.label.set_text(   f"Spectral Sub: {'ON' if state['spec_sub'] else 'OFF'}")
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

    b_notch.on_clicked(lambda e: (state.update(notch=not state['notch']),    refresh()))
    b_lp.on_clicked(   lambda e: (state.update(lowpass=not state['lowpass']),refresh()))
    b_ss.on_clicked(   lambda e: (state.update(spec_sub=not state['spec_sub']),refresh()))
    b_cal.on_clicked(on_cal)

    fig._buttons = (b_notch, b_lp, b_ss, b_cal)
    return fig, im, line, mt


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
        # Check default output device
        _, default_out_idx = sd.default.device
        if default_out_idx is not None and default_out_idx >= 0:
            info = sd.query_devices(default_out_idx)
            native = int(info.get('default_samplerate', fs))
            if native != fs:
                print(f"[audio] Default output native rate {native} Hz → using that.")
                fs = native

    _init_filters(fs)
    ring = np.zeros(fs * SPEC_SECONDS, dtype=np.float32)

    fig, im, line, mt = build_ui(fs)
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
        cal    = '✓' if (ss and ss.calibrated) else '–'
        mt.set_text(
            f"SNR (band):   {rough_snr_db(recent, fs):5.1f} dB\n"
            f"60 Hz pwr:    {power_at(recent, min(60, fs//2-1), fs):5.1f} dB\n"
            f"Fan (~300 Hz):{power_at(recent, min(300, fs//2-1), fs, bw=200):5.1f} dB\n"
            f"Calibrated:   {cal}\n"
            f"FS: {fs} Hz"
        )
        return im, line, mt

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
    p = argparse.ArgumentParser(description='Real-time denoiser — macOS-compatible')
    p.add_argument('--file',         default=None,        help='WAV file (omit for mic)')
    p.add_argument('--device-in',    type=int, default=None, help='Input device index')
    p.add_argument('--device-out',   type=int, default=None, help='Output device index')
    p.add_argument('--fs',           type=int, default=None, help='Force sample rate')
    p.add_argument('--list-devices', action='store_true',   help='Print devices and exit')
    p.add_argument('--alpha',        type=float, default=SS_ALPHA)
    p.add_argument('--beta',         type=float, default=SS_BETA)
    args = p.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        sys.exit(0)

    SS_ALPHA = args.alpha
    SS_BETA  = args.beta

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