"""Deployment exports for the streaming separator.

* `StreamStep`  - one full hop of streaming inference (framing, window, FFT, network,
  deep filter, iFFT, overlap-add) with all state passed explicitly, scripted to
  TorchScript so it runs from C++ (LibTorch) without Python.
* `NetStep`     - the network only (log-magnitude features -> deep-filter taps), with
  real/imag outputs, exported to ONNX for runtimes such as ONNX Runtime.

usage: python -m stagevox.export checkpoints/sep_w128.pt exports/
"""
from __future__ import annotations

import os
import sys
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Separator


class _Net(nn.Module):
    """The MaskNet layers re-wired for single-frame, explicit-state inference
    (the training module's optional-state forward is not scriptable)."""

    def __init__(self, sep: Separator):
        super().__init__()
        n = sep.net
        self.inp, self.norm, self.conv, self.gru, self.out = n.inp, n.norm, n.conv, n.gru, n.out
        self.n_bins, self.order, self.kernel = n.n_bins, n.order, n.kernel

    def forward(self, feats, buf, h):
        x = F.relu(self.norm(self.inp(feats))).transpose(1, 2)    # (1,H,1)
        xc = torch.cat([buf, x], dim=2)
        new_buf = xc[:, :, 1:]
        x = F.relu(self.conv(xc)).transpose(1, 2)                  # (1,1,H)
        x, new_h = self.gru(x, h)
        c = torch.tanh(self.out(x)).reshape(self.n_bins, self.order, 2)
        return c[:, :, 0], c[:, :, 1], new_buf, new_h


class NetStep(_Net):
    pass


class StreamStep(nn.Module):
    """Pure-tensor streaming step: identical maths to model.StreamingSeparator."""

    def __init__(self, sep: Separator):
        super().__init__()
        self.net = _Net(sep)
        self.win, self.hop, self.order, self.n_bins = sep.win, sep.hop, sep.order, sep.n_bins
        self.register_buffer("window", sep.window.clone())

    @torch.jit.export
    def initial_state(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        H = self.net.gru.hidden_size
        return (torch.zeros(self.win), torch.zeros(self.win),
                torch.zeros(self.order, self.n_bins), torch.zeros(self.order, self.n_bins),
                torch.zeros(1, H, self.net.kernel - 1), torch.zeros(self.net.gru.num_layers, 1, H))

    def forward(self, chunk, inbuf, ola, hist_re, hist_im, buf, h):
        inbuf = torch.cat([inbuf[self.hop:], chunk])
        X = torch.fft.rfft(inbuf * self.window)
        feats = torch.log(X.abs() + 1e-5).reshape(1, 1, self.n_bins)
        cr, ci, buf, h = self.net(feats, buf, h)
        hist_re = torch.cat([X.real.unsqueeze(0), hist_re[:-1]])
        hist_im = torch.cat([X.imag.unsqueeze(0), hist_im[:-1]])
        # complex multiply-accumulate over the K taps: (a+ib)(c+id)
        yr = (hist_re.t() * cr - hist_im.t() * ci).sum(-1)
        yi = (hist_re.t() * ci + hist_im.t() * cr).sum(-1)
        y = torch.fft.irfft(torch.complex(yr, yi), n=self.win) * self.window
        ola = ola + y
        out = ola[: self.hop].clone()
        ola = torch.cat([ola[self.hop:], torch.zeros(self.hop)])
        return out, inbuf, ola, hist_re, hist_im, buf, h


def load_separator(path):
    c = torch.load(path, map_location="cpu")
    m = Separator(c["win"], c["hidden"]); m.load_state_dict(c["state"]); return m.eval()


def export(ckpt, outdir):
    os.makedirs(outdir, exist_ok=True)
    sep = load_separator(ckpt)
    step = torch.jit.script(StreamStep(sep).eval())
    ts_path = os.path.join(outdir, f"stream_step_w{sep.win}.pt")
    step.save(ts_path)
    net = NetStep(sep).eval()
    Hd = sep.net.gru.hidden_size
    args = (torch.zeros(1, 1, sep.n_bins), torch.zeros(1, Hd, sep.net.kernel - 1),
            torch.zeros(sep.net.gru.num_layers, 1, Hd))
    onnx_path = os.path.join(outdir, f"net_step_w{sep.win}.onnx")
    torch.onnx.export(net, args, onnx_path, dynamo=False, opset_version=17,
                      input_names=["feats", "conv_buf", "gru_h"],
                      output_names=["coef_re", "coef_im", "new_conv_buf", "new_gru_h"])
    return ts_path, onnx_path


if __name__ == "__main__":
    print(export(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "exports"))
