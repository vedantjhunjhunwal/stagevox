"""Train one Separator config.  usage: python scripts/train.py --win 128 --steps 2000"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stagevox.metrics import si_sdr  # noqa: E402
from stagevox.model import Separator, si_sdr_loss  # noqa: E402


def batches(mix, tgt, rng, bs, seg):
    N = mix.shape[1]
    while True:
        xb, yb = [], []
        while len(xb) < bs:
            i = rng.integers(len(mix)); s = rng.integers(0, N - seg)
            y = tgt[i, s:s + seg]
            if np.sqrt(np.mean(y ** 2)) < 1e-3:          # skip crops where the singer rests
                continue
            g = 10 ** (rng.uniform(-6, 6) / 20)
            xb.append(mix[i, s:s + seg] * g); yb.append(y * g)
        yield torch.from_numpy(np.stack(xb)), torch.from_numpy(np.stack(yb))


@torch.no_grad()
def evaluate(model, mix, tgt):
    model.eval()
    est = torch.cat([model(torch.from_numpy(mix[i:i + 16])) for i in range(0, len(mix), 16)]).numpy()
    model.train()
    return float(np.mean([si_sdr(e, t) - si_sdr(m, t) for e, m, t in zip(est, mix, tgt)]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", type=int, required=True)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seg", type=float, default=1.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); torch.set_num_threads(int(os.environ.get("THREADS", 2)))
    rng = np.random.default_rng(a.seed)
    tr = np.load(os.path.join(a.data, "train.npz")); va = np.load(os.path.join(a.data, "val.npz"))
    mix, tgt = tr["mix"].astype(np.float32), tr["target"].astype(np.float32)
    vmix, vtgt = va["mix"].astype(np.float32), va["target"].astype(np.float32)
    model = Separator(win=a.win, hidden=a.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s / a.steps)))
    gen = batches(mix, tgt, rng, a.bs, int(a.seg * model.sr))
    os.makedirs(a.out, exist_ok=True)
    ckpt = os.path.join(a.out, f"sep_w{a.win}.pt")
    best, log, t0, run = -1e9, [], time.time(), 0.0
    for step in range(1, a.steps + 1):
        x, y = next(gen)
        loss = si_sdr_loss(model(x), y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 250 == 0 or step == a.steps:
            v = evaluate(model, vmix, vtgt)
            log.append({"step": step, "train_loss": run, "val_si_sdri": v, "min": (time.time() - t0) / 60})
            print(json.dumps(log[-1]), flush=True)
            if v > best:
                best = v
                torch.save({"win": a.win, "hidden": a.hidden, "state": model.state_dict()}, ckpt)
    json.dump({"win": a.win, "best_val_si_sdri": best, "log": log},
              open(os.path.join(a.out, f"train_w{a.win}.json"), "w"), indent=1)
