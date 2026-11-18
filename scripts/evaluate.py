"""Evaluate all trained configs + DSP baseline + oracle mask; measure streaming compute.

usage: python scripts/evaluate.py --data data --ckpt checkpoints --out results
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stagevox import SR  # noqa: E402
from stagevox.metrics import si_sdr  # noqa: E402
from stagevox.model import Separator  # noqa: E402
from stagevox.notch_suppressor import NotchFeedbackSuppressor  # noqa: E402

WINS = [512, 256, 128]


def load_model(path):
    c = torch.load(path, map_location="cpu")
    m = Separator(c["win"], c["hidden"]); m.load_state_dict(c["state"]); return m.eval()


@torch.no_grad()
def run_model(m, mix):
    return torch.cat([m(torch.from_numpy(mix[i:i + 16])) for i in range(0, len(mix), 16)]).numpy()


@torch.no_grad()
def oracle_mask(win, mix, tgt):
    """Ideal (clipped) magnitude mask with mixture phase: ceiling for a W-sample window."""
    m = Separator(win)
    out = []
    for x, y in zip(mix, tgt):
        X, Y = [torch.stft(torch.from_numpy(a), win, win // 2, window=m.window, center=True,
                           return_complex=True) for a in (x, y)]
        M = (Y.abs() / (X.abs() + 1e-8)).clamp(max=1.0)
        out.append(torch.istft(X * M, win, win // 2, window=m.window, center=True, length=len(x)).numpy())
    return np.stack(out)


def scores(est, mix, tgt, kinds):
    d = np.array([si_sdr(e, t) - si_sdr(m, t) for e, m, t in zip(est, mix, tgt)])
    r = {k: float(d[kinds == k].mean()) for k in np.unique(kinds)}
    r["all"] = float(d.mean()); r["median_all"] = float(np.median(d))
    return r


@torch.no_grad()
def stream_timing(m, seconds=10.0):
    torch.set_num_threads(1)
    st = m.stream(); x = torch.randn(int(seconds * SR)) * 0.05
    for c in x[: 50 * m.hop].split(m.hop):                       # warm-up
        st.process(c)
    st = m.stream(); ts = []
    for c in x[: (len(x) // m.hop) * m.hop].split(m.hop):
        t = time.perf_counter(); st.process(c); ts.append(time.perf_counter() - t)
    ts = np.array(ts) * 1000; hop_ms = 1000 * m.hop / SR
    return {"hop_ms": hop_ms, "compute_ms_p50": float(np.median(ts)), "compute_ms_p99": float(np.percentile(ts, 99)),
            "rtf": float(ts.mean() / hop_ms), "deadline_miss_pct": float(100 * np.mean(ts > hop_ms))}


def macs_per_second(m):
    H, Fb, L = m.net.gru.hidden_size, m.n_bins, m.net.gru.num_layers
    per_frame = Fb * H + 3 * H * H + L * 3 * (2 * H * H) + H * Fb * m.order * 2 + 4 * Fb * m.order
    return per_frame * SR / m.hop


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    sets = {k: np.load(os.path.join(a.data, f"test_{k}.npz")) for k in ("synth", "real")}
    res = {"n_scenes": {k: int(len(v["mix"])) for k, v in sets.items()}, "models": {}, "baselines": {}}
    for k, d in sets.items():
        mix, tgt, kinds = d["mix"], d["target"], d["kind"]
        res["baselines"].setdefault("input_si_sdr_db", {})[k] = float(np.mean([si_sdr(m, t) for m, t in zip(mix, tgt)]))
        sup = [NotchFeedbackSuppressor(SR)(m) for m in mix]
        res["baselines"].setdefault("notch_suppressor", {})[k] = scores(np.stack(sup), mix, tgt, kinds)
        for w in WINS:
            res["baselines"].setdefault(f"oracle_mask_w{w}", {})[k] = scores(oracle_mask(w, mix, tgt), mix, tgt, kinds)
    for w in WINS:
        p = os.path.join(a.ckpt, f"sep_w{w}.pt")
        if not os.path.exists(p):
            continue
        m = load_model(p)
        torch.set_num_threads(2)
        r = {"latency_ms": m.latency_ms, "params": sum(q.numel() for q in m.parameters()),
             "mmacs_per_s": macs_per_second(m) / 1e6}
        for k, d in sets.items():
            r[k] = scores(run_model(m, d["mix"]), d["mix"], d["target"], d["kind"])
        r["stream"] = stream_timing(m)
        res["models"][f"w{w}"] = r
        print(w, json.dumps(r), flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    print(json.dumps(res["baselines"], indent=1))
