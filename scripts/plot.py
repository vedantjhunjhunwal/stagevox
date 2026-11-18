"""Make the latency/quality trade-off plot and a before/after spectrogram from results."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stagevox import SR  # noqa: E402
from stagevox.metrics import si_sdr  # noqa: E402
from stagevox.notch_suppressor import NotchFeedbackSuppressor  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import load_model  # noqa: E402

res = json.load(open("results/results.json"))
wins = [512, 256, 128]
lat = [1000 * w / SR for w in wins]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k, style in (("synth", "-o"), ("real", "-s")):
    ax[0].plot(lat, [res["models"][f"w{w}"][k]["all"] for w in wins], style, label=f"StageVox ({k} test)")
    ax[0].plot(lat, [res["baselines"][f"oracle_mask_w{w}"][k]["all"] for w in wins], style.replace("-", ":"),
               alpha=0.6, label=f"oracle mask ({k})")
ax[0].axhline(res["baselines"]["notch_suppressor"]["synth"]["all"], color="gray", ls="--", label="notch suppressor (synth)")
ax[0].axhline(res["baselines"]["notch_suppressor"]["real"]["all"], color="gray", ls="-.", label="notch suppressor (real)")
ax[0].set_xscale("log", base=2); ax[0].set_xticks(lat); ax[0].set_xticklabels([f"{x:g}" for x in lat])
ax[0].set_xlabel("algorithmic latency (ms)"); ax[0].set_ylabel("SI-SDR improvement (dB)")
ax[0].set_title("Separation quality vs latency"); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
rtf = [res["models"][f"w{w}"]["stream"]["rtf"] for w in wins]
ax[1].bar([f"{x:g} ms" for x in lat], rtf, color="#1F4E79")
ax[1].axhline(1.0, color="red", ls="--", label="real-time limit")
ax[1].set_ylabel("real-time factor (1 CPU thread, streaming)"); ax[1].set_title("Compute cost vs latency")
ax[1].legend(); ax[1].grid(alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig("results/tradeoff.png", dpi=110)

# before/after on a held-out REAL scene with bleed + feedback
d = np.load("data/test_real.npz")
idx = [i for i, k in enumerate(d["kind"]) if k == "bleed+feedback"]
m = load_model("checkpoints/sep_w128.pt")
best = None
for i in idx:
    with torch.no_grad():
        y = m(torch.from_numpy(d["mix"][i:i + 1]))[0].numpy()
    gain = si_sdr(y, d["target"][i]) - si_sdr(d["mix"][i], d["target"][i])
    if best is None or abs(gain - res["models"]["w128"]["real"]["bleed+feedback"]) < best[0]:
        best = (abs(gain - res["models"]["w128"]["real"]["bleed+feedback"]), i, y, gain)
_, i, y, gain = best                                  # a typical (near-mean) example, not the best one
notch = NotchFeedbackSuppressor(SR)(d["mix"][i])
fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
for a, x, title in ((ax[0], d["mix"][i], f"stage mic input (SI-SDR {si_sdr(d['mix'][i], d['target'][i]):.1f} dB)"),
                    (ax[1], notch, f"notch feedback suppressor ({si_sdr(notch, d['target'][i]):.1f} dB)"),
                    (ax[2], y, f"StageVox, 8 ms latency ({si_sdr(y, d['target'][i]):.1f} dB)"),
                    (ax[3], d["target"][i], "clean direct vocal (target)")):
    a.specgram(x + 1e-7, NFFT=512, Fs=SR, noverlap=384, cmap="magma", vmin=-120)
    a.set_title(title, fontsize=9); a.set_ylabel("Hz")
ax[-1].set_xlabel("s"); plt.tight_layout(); plt.savefig("results/example_real.png", dpi=90)
print("example scene", i, "gain", round(gain, 2))
