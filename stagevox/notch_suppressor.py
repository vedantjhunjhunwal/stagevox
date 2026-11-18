"""Classic DSP baseline: automatic feedback suppressor (howl detection + dynamic notches).

This is what most live consoles ship: detect narrow spectral peaks that are both
much louder than the rest of the spectrum (peak-to-median ratio) and frequency-stable
for a sustained time, then drop a narrow notch on each.  Frequency stability is
measured with parabolic peak interpolation, so sung notes with vibrato or gliding
speech pitch are not mistaken for feedback.  Notches are IIR biquads: zero
algorithmic latency, but they only remove tonal ringing, not broadband bleed.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .fx import biquad


class NotchFeedbackSuppressor:
    def __init__(self, sr=16000, nfft=2048, hop=256, papr_db=25.0, persist_s=0.4,
                 max_std_hz=0.5, max_notches=8, q=40.0, min_hz=150.0):
        self.sr, self.nfft, self.hop = sr, nfft, hop
        self.papr_db, self.max_std_hz, self.max_notches, self.q = papr_db, max_std_hz, max_notches, q
        self.persist = max(2, int(persist_s * sr / hop))
        self.min_bin = int(min_hz * nfft / sr)
        self.win = np.hanning(nfft)
        self.notch_hz = []

    def _peaks(self, mag):
        med = np.median(mag)
        pk, _ = signal.find_peaks(mag)
        pk = [p for p in pk if p >= self.min_bin and p < len(mag) - 1 and mag[p] - med > self.papr_db]
        out = []
        for p in sorted(pk, key=lambda p: -mag[p])[:6]:
            a, b, c = mag[p - 1], mag[p], mag[p + 1]
            d = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
            out.append((p, (p + d) * self.sr / self.nfft))
        return out

    def __call__(self, x):
        H = self.hop
        y = np.zeros(len(x))
        hist = np.zeros(self.nfft)
        notches = []                      # [hz, b, a, zi]
        tracks = {}                       # bin -> list of interpolated frequencies
        for start in range(0, len(x), H):
            blk = np.asarray(x[start:start + H], np.float64)
            out = blk
            for nt in notches:
                out, nt[3] = signal.lfilter(nt[1], nt[2], out, zi=nt[3])
            y[start:start + len(blk)] = out
            # causal detector on the newest input
            hist = np.concatenate([hist[len(blk):], blk])
            mag = 20 * np.log10(np.abs(np.fft.rfft(hist * self.win)) + 1e-9)
            new = {}
            for p, f in self._peaks(mag):
                prev = next((tracks[q] for q in (p, p - 1, p + 1) if q in tracks), [])
                new[p] = (prev + [f])[-self.persist:]
            tracks = new
            for p, fs in tracks.items():
                if len(fs) < self.persist or np.std(fs) > self.max_std_hz or len(notches) >= self.max_notches:
                    continue
                hz = float(np.mean(fs))
                if all(abs(hz - nt[0]) > 3 * self.sr / self.nfft for nt in notches):
                    b, a = biquad("notch", hz, self.sr, q=self.q)
                    notches.append([hz, b, a, np.zeros(2)])
        self.notch_hz = [nt[0] for nt in notches]
        return y
