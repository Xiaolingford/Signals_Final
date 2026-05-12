"""
build_test_suite.py  (v2 — realistic stress test)

Two key fixes over v1:
  1. Blind windowing  — the decoder gets the full signal and must find tones
                        by sliding a 40 ms window, just like dtmf_live.py does.
                        No oracle timing.
  2. Tonal interference — new noise type "tonal" injects sine waves at
                          frequencies adjacent to the DTMF grid. This is what
                          actually challenges the Goertzel dominance gate.

Expected result: accuracy degrades realistically below ~5 dB for tonal noise
and below ~0 dB for white/hiss (because Goertzel is genuinely robust for
broadband noise — that's a correct and reportable finding).
"""
import argparse, csv, os, time, wave
import numpy as np

try:
    from dtmf_decoder import detect_digit, LOW_FREQS, HIGH_FREQS
    DECODER_SOURCE = "dtmf_decoder.py (shared)"
except ImportError:
    print("[WARN] dtmf_decoder.py not found — using built-in fallback.\n")
    LOW_FREQS  = [697, 770, 852, 941]
    HIGH_FREQS = [1209, 1336, 1477, 1633]
    def _g(samples, freq, sr):
        omega=2*np.pi*freq/sr; coeff=2*np.cos(omega); s1=s2=0.0
        for x in samples:
            s=float(x)+coeff*s1-s2; s2,s1=s1,s
        return s2*s2+s1*s1-coeff*s1*s2
    _T={(697,1209):'1',(697,1336):'2',(697,1477):'3',(697,1633):'A',
        (770,1209):'4',(770,1336):'5',(770,1477):'6',(770,1633):'B',
        (852,1209):'7',(852,1336):'8',(852,1477):'9',(852,1633):'C',
        (941,1209):'*',(941,1336):'0',(941,1477):'#',(941,1633):'D'}
    def detect_digit(samples, sr, min_power=1e3, dominance_ratio=4.0, max_twist_db=8.0):
        lp=np.array([_g(samples,f,sr) for f in LOW_FREQS])
        hp=np.array([_g(samples,f,sr) for f in HIGH_FREQS])
        li,hi=int(np.argmax(lp)),int(np.argmax(hp))
        if lp[li]<min_power or hp[hi]<min_power: return None,lp,hp
        ol=np.delete(lp,li).max(); oh=np.delete(hp,hi).max()
        if ol>0 and lp[li]<ol*dominance_ratio: return None,lp,hp
        if oh>0 and hp[hi]<oh*dominance_ratio: return None,lp,hp
        if abs(10*np.log10(lp[li]/hp[hi]))>max_twist_db: return None,lp,hp
        return _T.get((LOW_FREQS[li],HIGH_FREQS[hi])),lp,hp
    DECODER_SOURCE = "built-in fallback"

from generate_dtmf import build_dtmf_signal, OUTPUT_DIR, SAMPLE_RATE, TONE_MS, SILENCE_MS, save_wav
from noise_inject import snr_db, make_noise, scale_noise_to_snr, hum_noise

DIGIT_SEQUENCES = ["911","123","0","5551234","4085550100","18005551234"]
SNR_LEVELS      = [30, 25, 20, 15, 10, 5, 0, -5]
RESULTS_CSV     = "results.csv"

# ── Tonal interference noise ──────────────────────────────────────────────────
def tonal_noise(n_samples, sample_rate=SAMPLE_RATE, rng=None):
    """
    Sine tones at frequencies BETWEEN the DTMF grid rows/columns.
    These sit near (but not on) the DTMF frequencies, making the
    Goertzel dominance check fail — the real stress test for the decoder.

    Offsets chosen so they fall exactly between rows and between columns:
      Low band:   733, 811, 896         (midpoints between 697/770, 770/852, 852/941)
      High band:  1272, 1406            (midpoints between 1209/1336, 1336/1477)
    """
    rng = rng or np.random.default_rng()
    t = np.arange(n_samples) / sample_rate
    interferers = [733, 811, 896, 1272, 1406]
    # Random phase per component so they don't constructively add every time
    phases = rng.uniform(0, 2*np.pi, len(interferers))
    signal = sum(np.sin(2*np.pi*f*t + p) for f, p in zip(interferers, phases))
    p = np.mean(signal**2)
    return signal / np.sqrt(p) if p > 0 else signal


NOISE_CATALOGUE = [
    {"type": "white",  "hum": False},
    {"type": "hiss",   "hum": False},
    {"type": "mixed",  "hum": False},
    {"type": "white",  "hum": True},
    {"type": "tonal",  "hum": False},   # NEW: the real stress test
]


# ── WAV I/O ───────────────────────────────────────────────────────────────────
def load_wav_float(path):
    with wave.open(path,"r") as wf:
        sr=wf.getframerate(); raw=wf.readframes(wf.getnframes())
    return np.frombuffer(raw,dtype=np.int16).astype(np.float64)/32767, sr


# ── BLIND sliding-window decoder (no oracle timing) ──────────────────────────
def decode_blind(signal, sr, n_digits,
                 block_ms=40, tone_ms=TONE_MS, silence_ms=SILENCE_MS):
    """
    Slide a 40 ms window (same as dtmf_live.py) across the signal,
    collect detections, then de-bounce into a digit sequence.

    This is the honest test: the decoder does NOT know where tones start.
    It has to find them the same way the live demo does.

    Returns the decoded string (may be shorter than n_digits if some missed).
    """
    block_n   = int(sr * block_ms   / 1000)
    tone_n    = int(sr * tone_ms    / 1000)
    silence_n = int(sr * silence_ms / 1000)
    stride    = tone_n + silence_n

    # Slide at half-block steps for better onset coverage
    step      = block_n // 2
    detections = []   # list of (sample_index, digit)

    for start in range(0, len(signal) - block_n, step):
        block = signal[start:start + block_n]
        digit, _, _ = detect_digit(block, sr)
        if digit is not None:
            detections.append((start, digit))

    if not detections:
        return ""

    # De-bounce: group consecutive detections of the same digit that are
    # separated by < 1.5× the expected stride. Emit one digit per group.
    decoded   = ""
    prev_digit = None
    prev_idx   = -9999
    run        = 0
    STABLE     = 2   # need ≥2 consecutive detections to commit

    for idx, digit in detections:
        gap = idx - prev_idx
        if digit == prev_digit and gap <= int(step * 1.5):
            run += 1
            if run == STABLE:
                decoded += digit
        else:
            prev_digit = digit
            run = 1
        prev_idx = idx

    return decoded


# ── Accuracy helpers ──────────────────────────────────────────────────────────
def digit_accuracy(ref, dec):
    if not ref: return 0.0
    correct = sum(r==d for r,d in zip(ref, dec.ljust(len(ref))))
    return correct / len(ref)

def seq_accuracy(ref, dec):
    return ref == dec


# ── Suite builder ─────────────────────────────────────────────────────────────
def build_suite(output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    records=[]; total=0; t0=time.time()
    print("="*68)
    print(f"  Building DTMF test suite v2  [decoder: {DECODER_SOURCE}]")
    print(f"  Noise types: white, hiss, mixed, white+hum, tonal (interferer)")
    print(f"  Windowing  : BLIND sliding window (no oracle timing)")
    print("="*68)

    for digits in DIGIT_SEQUENCES:
        print(f"\n── {digits} ──")
        signal = build_dtmf_signal(digits)

        # Clean reference
        clean_path = os.path.join(output_dir, f"test_{digits}_clean.wav")
        save_wav(clean_path, signal)
        records.append({"digits":digits,"noise_type":"clean","hum":False,
                        "target_snr_db":None,"measured_snr_db":None,"path":clean_path})
        total += 1

        for nc in NOISE_CATALOGUE:
            nt, hum = nc["type"], nc["hum"]
            for snr in SNR_LEVELS:
                seed = 42 + hash(digits+nt+str(hum)+str(snr)) % 9999
                rng  = np.random.default_rng(seed)
                n    = len(signal)

                # Build noise
                if nt == "tonal":
                    raw_noise = tonal_noise(n, SAMPLE_RATE, rng)
                else:
                    raw_noise = make_noise(nt, n, SAMPLE_RATE, rng)

                noise = scale_noise_to_snr(signal, raw_noise, snr)
                noisy = signal + noise

                if hum:
                    h = hum_noise(n, 60.0, SAMPLE_RATE)
                    h = scale_noise_to_snr(signal, h, snr + 10)
                    noisy += h; noise += h

                noisy = np.clip(noisy, -1, 1)
                actual = snr_db(signal[:n], noise[:n])

                hum_tag  = "_hum" if hum else ""
                fname    = os.path.join(output_dir,
                           f"test_{digits}_{nt}{hum_tag}_snr{snr}.wav")
                save_wav(fname, noisy)
                records.append({"digits":digits,"noise_type":nt,"hum":hum,
                                "target_snr_db":snr,
                                "measured_snr_db":round(actual,2),
                                "path":fname})
                total += 1

    print(f"\n{'='*68}")
    print(f"  Generated {total} files in {time.time()-t0:.1f}s → '{output_dir}/'")
    print("="*68+"\n")
    return records


# ── Analysis pass (blind windowing) ──────────────────────────────────────────
def analyze_suite(records):
    print("Decoding with blind sliding window (no oracle timing) …\n")
    for rec in records:
        if rec["noise_type"] == "clean":
            # Even clean: use blind decoder to show true baseline
            sig, sr = load_wav_float(rec["path"])
            decoded  = decode_blind(sig, sr, len(rec["digits"]))
            rec["decoded"]   = decoded
            rec["digit_acc"] = digit_accuracy(rec["digits"], decoded)
            rec["seq_acc"]   = seq_accuracy(rec["digits"], decoded)
            continue
        sig, sr = load_wav_float(rec["path"])
        decoded  = decode_blind(sig, sr, len(rec["digits"]))
        rec["decoded"]   = decoded
        rec["digit_acc"] = digit_accuracy(rec["digits"], decoded)
        rec["seq_acc"]   = seq_accuracy(rec["digits"], decoded)
    return records


# ── Criterion 3 table ─────────────────────────────────────────────────────────
def print_summary(records):
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in records:
        if r.get("target_snr_db") is None: continue
        buckets[(r["target_snr_db"], r["noise_type"])].append(r)

    print("\n"+"─"*76)
    print(f"  CRITERION 3 — Decoder accuracy vs SNR")
    print(f"  Method: blind sliding-window  |  {DECODER_SOURCE}")
    print("─"*76)
    print(f"  {'SNR':>6}  {'Noise':>8}  {'N':>4}  {'Digit Acc':>10}  {'Seq Acc':>9}  Bar")
    print("─"*76)

    for (snr, nt), recs in sorted(buckets.items(), key=lambda x:(-x[0][0], x[0][1])):
        n = len(recs)
        d = 100*np.mean([r["digit_acc"] for r in recs])
        s = 100*np.mean([r["seq_acc"]   for r in recs])
        bar = "█" * int(d/5)
        print(f"  {snr:>6}  {nt:>8}  {n:>4}  {d:>9.1f}%  {s:>8.1f}%  {bar}")

    print("─"*76)
    noise_types = sorted({r["noise_type"] for r in records if r.get("target_snr_db") is not None})
    print("\n  Min SNR for ≥90% digit accuracy:")
    for nt in noise_types:
        by_snr = {}
        for r in records:
            if r.get("noise_type") != nt or r.get("target_snr_db") is None: continue
            by_snr.setdefault(r["target_snr_db"],[]).append(r["digit_acc"])
        passing = [s for s,accs in by_snr.items() if np.mean(accs) >= 0.90]
        print(f"    {nt:<10}  →  {f'{min(passing)} dB' if passing else 'not achieved'}")
    print()


def save_csv(records, path=RESULTS_CSV):
    if not records: return
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0].keys()))
        w.writeheader(); w.writerows(records)
    print(f"  Results saved → {path}")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out", default=OUTPUT_DIR)
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--csv", default=RESULTS_CSV)
    args=p.parse_args()
    records=build_suite(args.out)
    if not args.build_only:
        records=analyze_suite(records)
        print_summary(records)
        save_csv(records,args.csv)

if __name__=="__main__":
    main()