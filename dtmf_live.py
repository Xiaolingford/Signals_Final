"""
DTMF Live Demo
==============

Listens to the default microphone, runs the Goertzel decoder on every
40 ms block, and prints a live verbose display: 8 energy bars (one per
DTMF frequency), the currently-detected digit, and the running history
of all digits decoded so far.

Run:
    python dtmf_live.py                 # default mic, 8 kHz
    python dtmf_live.py --list          # list audio devices, then exit
    python dtmf_live.py --device 3      # pick a specific input device
    python dtmf_live.py --rate 16000    # use 16 kHz capture instead

Tip for the demo: play DTMF tones from your phone's keypad
(or use a tone generator app / website) into the laptop mic.
"""

import argparse
import queue
import sys
import time

import numpy as np
import sounddevice as sd

from dtmf_decoder import (
    LOW_FREQS,
    HIGH_FREQS,
    detect_digit,
    goertzel_bank,
)

# ---------------------------------------------------------------------------
# Terminal styling (ANSI escape codes)
# ---------------------------------------------------------------------------
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"

CLEAR_SCREEN = "\033[2J"
MOVE_HOME    = "\033[H"
CLEAR_DOWN   = "\033[J"
HIDE_CURSOR  = "\033[?25l"
SHOW_CURSOR  = "\033[?25h"

BAR_WIDTH = 36

# ---------------------------------------------------------------------------
# Audio constants — 40 ms block @ 8 kHz is the DTMF sweet spot
# (DTMF tones are spec'd to be >= 40 ms long)
# ---------------------------------------------------------------------------
DEFAULT_RATE = 8000
BLOCK_MS = 40


# ---------------------------------------------------------------------------
# Bar rendering
# ---------------------------------------------------------------------------
def make_bar(value, max_value, width=BAR_WIDTH, color=GREEN):
    """Render a horizontal energy bar."""
    if max_value <= 0:
        ratio = 0.0
    else:
        ratio = min(1.0, value / max_value)
    filled = int(round(ratio * width))
    return f"{color}{'█' * filled}{DIM}{'·' * (width - filled)}{RESET}"


def db_value(power):
    """Convert raw Goertzel magnitude-squared to a dB-ish value for display."""
    if power <= 0:
        return -99.0
    return 10.0 * np.log10(power + 1e-12)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def render(low_powers, high_powers,
           current_digit, history, display_max,
           block_count, sample_rate, status_msg):
    """Build the full verbose display string."""
    out = [MOVE_HOME, CLEAR_DOWN]

    out.append(f"{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════╗{RESET}\n")
    out.append(f"{BOLD}{CYAN}║          DTMF LIVE DECODER  ·  Goertzel + Live Mic           ║{RESET}\n")
    out.append(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════════════╝{RESET}\n")
    out.append("\n")

    # --- LOW band -----------------------------------------------------------
    low_peak_idx = int(np.argmax(low_powers))
    out.append(f"  {BOLD}LOW BAND  (rows){RESET}\n")
    for i, (f, p) in enumerate(zip(LOW_FREQS, low_powers)):
        marker = f"{BOLD}{GREEN}◀{RESET}" if i == low_peak_idx else " "
        bar = make_bar(p, display_max, color=YELLOW)
        out.append(f"   {f:>5} Hz  {bar}  {db_value(p):>6.1f} dB {marker}\n")

    out.append("\n")

    # --- HIGH band ----------------------------------------------------------
    high_peak_idx = int(np.argmax(high_powers))
    out.append(f"  {BOLD}HIGH BAND (cols){RESET}\n")
    for i, (f, p) in enumerate(zip(HIGH_FREQS, high_powers)):
        marker = f"{BOLD}{GREEN}◀{RESET}" if i == high_peak_idx else " "
        bar = make_bar(p, display_max, color=MAGENTA)
        out.append(f"   {f:>5} Hz  {bar}  {db_value(p):>6.1f} dB {marker}\n")

    out.append("\n")

    # --- Current digit ------------------------------------------------------
    if current_digit is not None:
        out.append(f"  {BOLD}{GREEN}▶ DETECTED:  {current_digit}{RESET}\n")
    else:
        out.append(f"  {DIM}  (listening...){RESET}                                       \n")

    out.append("\n")

    # --- History ------------------------------------------------------------
    out.append(f"  {BOLD}HISTORY:{RESET}  {CYAN}{history or '—'}{RESET}\n")
    out.append("\n")

    # --- Status footer ------------------------------------------------------
    out.append(f"  {DIM}{status_msg}{RESET}\n")
    out.append(f"  {DIM}blocks: {block_count}   sample rate: {sample_rate} Hz   "
               f"block: {BLOCK_MS} ms     Ctrl+C to quit{RESET}\n")

    return "".join(out)


# ---------------------------------------------------------------------------
# Detection state machine — emits a digit once per "press"
# ---------------------------------------------------------------------------
class DigitGate:
    """
    Edge-triggered digit emitter.

    A digit is emitted when:
      - it's been detected for STABLE_BLOCKS consecutive blocks
      - AND we haven't just emitted it (gate must reset via SILENCE_BLOCKS
        of None / different-digit detections first).

    This stops one phone-key press (which is ~80-200 ms long) from
    flooding the history with repeats.
    """
    STABLE_BLOCKS  = 2  # need 2 same-digit blocks (~80 ms) to emit
    SILENCE_BLOCKS = 2  # need 2 quiet blocks before another emit can fire

    def __init__(self):
        self.candidate = None
        self.candidate_run = 0
        self.last_emitted = None
        self.silence_run = 0
        self.armed = True   # ready to emit

    def update(self, detected_digit):
        """Feed one block's detection. Returns the digit just emitted, or None."""
        if detected_digit is None:
            self.candidate = None
            self.candidate_run = 0
            self.silence_run += 1
            if self.silence_run >= self.SILENCE_BLOCKS:
                self.armed = True
            return None

        # We have a digit.
        if detected_digit == self.candidate:
            self.candidate_run += 1
        else:
            self.candidate = detected_digit
            self.candidate_run = 1

        self.silence_run = 0

        if (self.armed
                and self.candidate_run >= self.STABLE_BLOCKS
                and self.candidate is not None):
            self.armed = False
            self.last_emitted = self.candidate
            return self.candidate

        return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DTMF live mic decoder")
    parser.add_argument("--list", action="store_true",
                        help="List audio devices and exit")
    parser.add_argument("--device", type=int, default=None,
                        help="Input device index (see --list)")
    parser.add_argument("--rate", type=int, default=DEFAULT_RATE,
                        help=f"Sample rate in Hz (default {DEFAULT_RATE})")
    parser.add_argument("--min-power", type=float, default=None,
                        help="Override minimum-power gate (auto if omitted)")
    args = parser.parse_args()

    if args.list:
        print(sd.query_devices())
        return

    sample_rate = args.rate
    block_size = int(sample_rate * BLOCK_MS / 1000)

    # Auto-scale min_power with block size: peak power scales ~ N for sine
    # input, so 1e3 was tuned for N=320. For other N, scale linearly.
    if args.min_power is None:
        min_power = 1e3 * (block_size / 320.0) ** 2 * 0.25
    else:
        min_power = args.min_power

    audio_q = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            # Drop overflow/underflow notices into the queue as a string
            audio_q.put(("status", str(status)))
        # indata is shape (frames, channels) — take channel 0
        audio_q.put(("data", indata[:, 0].copy()))

    gate = DigitGate()
    history = ""
    current_digit = None
    block_count = 0
    display_max = 1.0   # auto-ranges so bars fill nicely
    last_status = "starting capture..."
    last_redraw = 0.0
    REDRAW_HZ = 30      # cap redraws so terminal isn't overwhelmed

    print(HIDE_CURSOR + CLEAR_SCREEN, end="", flush=True)

    try:
        with sd.InputStream(samplerate=sample_rate,
                            blocksize=block_size,
                            channels=1,
                            dtype="float32",
                            device=args.device,
                            callback=audio_callback):
            last_status = (f"capturing from "
                           f"{sd.query_devices(args.device)['name'] if args.device is not None else 'default mic'}")
            while True:
                tag, payload = audio_q.get()
                if tag == "status":
                    last_status = f"audio status: {payload}"
                    continue

                samples = payload
                block_count += 1

                detected, low_p, high_p = detect_digit(
                    samples, sample_rate, min_power=min_power)

                # Auto-range the bar scale based on running peaks
                cur_peak = max(low_p.max(), high_p.max(), 1.0)
                # Smoothly track the max
                display_max = max(display_max * 0.995, cur_peak * 1.1)

                emitted = gate.update(detected)
                if emitted is not None:
                    history += emitted

                # Show whatever the decoder *currently* sees, not just emitted
                current_digit = detected

                now = time.monotonic()
                if now - last_redraw >= 1.0 / REDRAW_HZ:
                    sys.stdout.write(render(
                        low_p, high_p, current_digit, history,
                        display_max, block_count, sample_rate, last_status))
                    sys.stdout.flush()
                    last_redraw = now

    except KeyboardInterrupt:
        pass
    except Exception as e:
        sys.stdout.write(SHOW_CURSOR + "\n")
        print(f"{RED}Error:{RESET} {e}", file=sys.stderr)
        if "PortAudio" in str(e) or "sounddevice" in str(e).lower():
            print("Hint: install PortAudio (`brew install portaudio` on macOS, "
                  "`sudo apt install portaudio19-dev` on Linux), then "
                  "`pip install sounddevice`.", file=sys.stderr)
        sys.exit(1)
    finally:
        sys.stdout.write(SHOW_CURSOR + "\n")
        sys.stdout.flush()
        if history:
            print(f"{BOLD}Final decoded sequence:{RESET} {CYAN}{history}{RESET}")


if __name__ == "__main__":
    main()