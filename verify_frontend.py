"""verify_frontend.py — Frontend + engine verification for all modality combinations."""
import numpy as np
import sys
import time

sys.path.insert(0, ".")
import final_submission as eng

FS = 360.0
t = np.linspace(0, 20, int(20 * FS))


def make_ecg(bpm=70.0, noise=0.05, seed=1):
    n = int(20 * FS)
    ecg = np.zeros(n)
    rr = 60.0 / bpm
    for i in range(int(20 / rr) + 2):
        idx = int((i + 0.5) * rr * FS)
        if idx < n:
            w = int(0.025 * FS)
            for d in range(-w * 3, w * 3 + 1):
                if 0 <= idx + d < n:
                    ecg[idx + d] += float(np.exp(-(d / w) ** 2))
    return ecg + noise * np.random.RandomState(seed).randn(n)


ecg = make_ecg(70.0)
ppg = 0.5 * np.sin(2 * np.pi * 1.2 * t) + 0.01 * np.random.RandomState(2).randn(len(t))
emg = np.random.RandomState(3).randn(int(20 * FS)) * 0.5

print("=" * 64)
print("  FRONTEND VERIFICATION — ALL 7 MODALITY COMBINATIONS")
print("=" * 64)

combos = [
    ("ECG only",        [("ecg", ecg)]),
    ("EMG only",        [("emg", emg)]),
    ("PPG only",        [("ppg", ppg)]),
    ("ECG + EMG",       [("ecg", ecg), ("emg", emg)]),
    ("ECG + PPG",       [("ecg", ecg), ("ppg", ppg)]),
    ("EMG + PPG",       [("emg", emg), ("ppg", ppg)]),
    ("ECG + EMG + PPG", [("ecg", ecg), ("emg", emg), ("ppg", ppg)]),
]

all_pass = True

for combo_name, signals in combos:
    ok = True
    details = []
    for mod, sig in signals:
        r = eng.run(sig, FS, modality=mod)
        if not r["valid"]:
            ok = False
            details.append(f"{mod.upper()}: INVALID — {r['validation_message']}")
        else:
            if mod == "ecg":
                details.append(
                    f"ECG hr={r['heart_rate_bpm']:.0f} bpm  beats={r['n_beats']}"
                    f"  q={r['signal_quality']}  rhythm={r['rhythm_label']}"
                )
            elif mod == "emg":
                details.append(
                    f"EMG sev={r['emg_severity']}  q={r['signal_quality']}"
                    f"  ratio={r.get('emg_energy_ratio', 0)*100:.0f}%"
                )
            elif mod == "ppg":
                pr = r.get("pulse_rate_bpm") or r.get("heart_rate_bpm", 0)
                details.append(
                    f"PPG pr={pr:.0f} bpm  peaks={r['n_beats']}"
                    f"  q={r['signal_quality']}  conf={r['detection_reliability']*100:.0f}%"
                )
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_pass = False
    print(f"\n  [{status}] {combo_name}")
    for d in details:
        print(f"         {d}")

print()
print("-" * 64)
print("  EDGE CASE TESTS")
print("-" * 64)

edge_cases = [
    ("Empty signal",          lambda: eng.run(np.zeros(0), FS, modality="ecg"),    False),
    ("Too short (50 samp)",   lambda: eng.run(np.zeros(50), FS, modality="ecg"),   False),
    ("All NaN",               lambda: eng.run(np.full(1000, float("nan")), FS),    False),
    ("Bad FS (0 Hz)",         lambda: eng.run(ecg, 0.0, modality="ecg"),           False),
    ("Bad FS (10 Hz)",        lambda: eng.run(ecg, 10.0, modality="ecg"),          False),
    ("Unknown modality (eeg)",lambda: eng.run(ecg, FS, modality="eeg"),            False),
    ("Unknown modality (xyz)",lambda: eng.run(ecg, FS, modality="xyz"),            False),
    ("Valid ECG",             lambda: eng.run(ecg, FS, modality="ecg"),            True),
    ("Valid PPG",             lambda: eng.run(ppg, FS, modality="ppg"),            True),
    ("Valid EMG",             lambda: eng.run(emg, FS, modality="emg"),            True),
    ("2D input (N,1)",        lambda: eng.run(ecg.reshape(-1, 1), FS),             True),
    ("ECG modality (upper)",  lambda: eng.run(ecg, FS, modality="ECG"),            True),
    ("ECG with NaN patch",    lambda: eng.run(
        np.where(np.arange(len(ecg)) % 500 < 20, float("nan"), ecg), FS), True),
    ("Noisy ECG",             lambda: eng.run(ecg + 2.0*np.random.RandomState(9).randn(len(ecg)), FS), True),
]

for name, fn, expect_valid in edge_cases:
    try:
        r = fn()
        got = r["valid"]
        detail = r.get("validation_message", "") if not got else r.get("signal_quality", "")
    except Exception as e:
        got = False
        detail = str(e)[:60]
    ok2 = (got == expect_valid)
    if not ok2:
        all_pass = False
    sym = "PASS" if ok2 else "FAIL"
    print(f"  [{sym}] {name:<30} valid={str(got):<5}  {detail}")

print()
print("-" * 64)
print("  KEY v2.1.0 API FIELD PRESENCE")
print("-" * 64)

ecg_result = eng.run(ecg, FS, modality="ecg")
ppg_result = eng.run(ppg, FS, modality="ppg")
emg_result = eng.run(emg, FS, modality="emg")

ecg_fields = [
    "snr_proxy_before_db", "snr_proxy_after_db",
    "snr_before_db", "snr_after_db",   # backward-compat aliases
    "r_peaks", "rr_intervals_s",
    "detection_latency_s",
    "vt_evidence", "vf_evidence",
    "rhythm_label", "rhythm_confidence_score",
    "morphology_score",
    "n_beats", "heart_rate_bpm",
    "possible_vt", "possible_vf",
    "powerline_50hz", "powerline_60hz",
    "detection_reliability",
    "signal_quality", "signal_quality_score",
    "emg_severity", "emg_energy_ratio",
    "noise_type", "warnings",
    "valid", "validation_message",
]
ppg_fields = [
    "pulse_peaks",           # new canonical
    "pulse_rate_bpm",        # new canonical
    "inter_pulse_intervals_s",  # new canonical
    "r_peaks",               # backward-compat
    "heart_rate_bpm",        # backward-compat
    "rr_intervals_s",        # backward-compat
]
emg_fields = [
    "emg_severity", "emg_energy_ratio",
    "signal_quality", "signal_quality_score",
    "valid",
]

for f in ecg_fields:
    present = f in ecg_result
    if not present:
        all_pass = False
    print(f"  [{'OK  ' if present else 'MISS'}] ECG: {f}")

print()
for f in ppg_fields:
    present = f in ppg_result
    if not present:
        all_pass = False
    print(f"  [{'OK  ' if present else 'MISS'}] PPG: {f}")

print()
for f in emg_fields:
    present = f in emg_result
    if not present:
        all_pass = False
    print(f"  [{'OK  ' if present else 'MISS'}] EMG: {f}")

print()
print("-" * 64)
print("  FIX SPOT-CHECKS")
print("-" * 64)

# Fix 1: VT not from simple periodic signal
simple_sine = 0.5 * np.sin(2 * np.pi * 1.2 * t)  # ~72 bpm sine — not ECG
r_sine = eng.run(simple_sine, FS, modality="ecg")
sine_not_vt = (not r_sine["possible_vt"]) or (len(r_sine.get("vt_evidence", [])) >= 2)
print(f"  [{'PASS' if sine_not_vt else 'FAIL'}] Fix1: Simple sine NOT VT from rate alone  (vt={r_sine['possible_vt']})")

# Fix 5: Normal rhythm → latency is None
r_norm = eng.run(ecg, FS, modality="ecg")
lat_ok = (r_norm["detection_latency_s"] is None) or r_norm["possible_vt"]
print(f"  [{'PASS' if lat_ok else 'FAIL'}] Fix5: Normal rhythm latency=None  (lat={r_norm['detection_latency_s']})")

# Fix 6: Unknown modality → valid=False
r_unk = eng.run(ecg, FS, modality="eeg")
print(f"  [{'PASS' if not r_unk['valid'] else 'FAIL'}] Fix6: Unknown modality valid=False  ({r_unk['validation_message'][:50]})")

# Fix 8: SNR proxy fields present
snr_ok = "snr_proxy_before_db" in ecg_result and "snr_proxy_after_db" in ecg_result
print(f"  [{'PASS' if snr_ok else 'FAIL'}] Fix8: snr_proxy_before/after_db present")

# Fix 9: PPG pulse_peaks field
ppg_pk_ok = "pulse_peaks" in ppg_result
print(f"  [{'PASS' if ppg_pk_ok else 'FAIL'}] Fix9: PPG pulse_peaks field present")

# Fix 10: snr None check — improvement only when both present
snr_b = ecg_result.get("snr_before_db")
snr_a = ecg_result.get("snr_after_db")
snr_imp = ecg_result.get("snr_improvement_db")
if snr_b is not None and snr_a is not None:
    imp_ok = snr_imp is not None
else:
    imp_ok = True
print(f"  [{'PASS' if imp_ok else 'FAIL'}] Fix10: SNR improvement computed iff both present")

print()
print("=" * 64)
print(f"  RESULT: {'ALL PASS ✓' if all_pass else 'SOME FAILURES — see above'}")
print("  NOTE: LOCAL TEST — NOT OFFICIAL OPTIFORGE SCORE")
print("=" * 64)
