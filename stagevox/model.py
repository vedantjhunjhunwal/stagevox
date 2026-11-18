"""Causal STFT deep-filtering separator with an exact frame-by-frame streaming implementation.

Latency budget
--------------
Analysis/synthesis use a sqrt-Hann window of W samples with hop W/2 (perfect
reconstruction).  A sample can only be emitted once every frame that overlaps it has
been processed, so the algorithmic latency is W samples (e.g. W=128 @ 16 kHz = 8 ms).
The network itself is strictly causal across frames (left-padded conv + uni-GRU),
so no extra look-ahead is added on top of the window.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sqrt_hann(W, device=None):
    return torch.sqrt(torch.hann_window(W, periodic=True, device=device))


class MaskNet(nn.Module):
    """Predicts, per frame and bin, K complex filter taps applied to the current and
    K-1 previous STFT frames (DeepFilterNet-style "deep filtering").  K=1 reduces to a
    complex ratio mask; K>1 lets the model undo short convolutive paths such as the
    monitor-to-mic feedback loop, which a magnitude mask cannot."""

    def __init__(self, n_bins: int, hidden: int = 128, layers: int = 2, kernel: int = 3, order: int = 3):
        super().__init__()
        self.kernel, self.order, self.n_bins = kernel, order, n_bins
        self.inp = nn.Linear(n_bins, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.conv = nn.Conv1d(hidden, hidden, kernel)            # causal by left padding
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True)
        self.out = nn.Linear(hidden, n_bins * order * 2)

    def forward(self, feats, state=None):
        """feats: (B, T, F) log-magnitudes. state: (conv_buf, h) for streaming."""
        x = F.relu(self.norm(self.inp(feats)))                  # (B,T,H)
        x = x.transpose(1, 2)                                    # (B,H,T)
        if state is None:
            buf = x.new_zeros(x.shape[0], x.shape[1], self.kernel - 1); h = None
        else:
            buf, h = state
        xc = torch.cat([buf, x], dim=2)
        new_buf = xc[:, :, -(self.kernel - 1):]
        x = F.relu(self.conv(xc)).transpose(1, 2)                # (B,T,H)
        x, h = self.gru(x, h)
        c = torch.tanh(self.out(x)).unflatten(-1, (self.n_bins, self.order, 2))
        return torch.complex(c[..., 0], c[..., 1]), (new_buf, h)       # (B,T,F,K)


class Separator(nn.Module):
    def __init__(self, win: int = 256, hidden: int = 128, layers: int = 2, sr: int = 16000, order: int = 3):
        super().__init__()
        assert win % 2 == 0
        self.win, self.hop, self.sr = win, win // 2, sr
        self.n_bins = win // 2 + 1
        self.order = order
        self.net = MaskNet(self.n_bins, hidden, layers, order=order)
        self.register_buffer("window", sqrt_hann(win), persistent=False)

    @property
    def latency_ms(self):
        return 1000.0 * self.win / self.sr

    # -------------------------------------------------------------- offline
    def forward(self, x):
        """x: (B, N) -> (B, N). Frames are aligned so an identity mask reconstructs x."""
        B, N = x.shape
        W, H = self.win, self.hop
        T = math.ceil(N / H) + 1
        total = (T - 1) * H + W
        xp = F.pad(x, (H, total - N - H))                       # left pad = hop
        frames = xp.unfold(1, W, H) * self.window                # (B,T,W)
        X = torch.fft.rfft(frames, dim=-1)                       # (B,T,F)
        coef, _ = self.net(torch.log(X.abs() + 1e-5))
        Xh = F.pad(X.transpose(1, 2), (self.order - 1, 0)).unfold(2, self.order, 1)  # (B,F,T,K)
        Y = (Xh.transpose(1, 2).flip(-1) * coef).sum(-1)                  # tap 0 = current frame
        y_frames = torch.fft.irfft(Y, n=W, dim=-1) * self.window
        y = F.fold(y_frames.transpose(1, 2), output_size=(1, total), kernel_size=(1, W),
                   stride=(1, H)).reshape(B, total)
        return y[:, H:H + N]

    # ------------------------------------------------------------ streaming
    def stream(self):
        return StreamingSeparator(self)


class StreamingSeparator:
    """Feed `hop` samples at a time; get `hop` samples back, delayed by `win` samples."""

    def __init__(self, model: Separator):
        self.m = model.eval()
        W, H = model.win, model.hop
        self.inbuf = torch.zeros(W)            # last W input samples
        self.ola = torch.zeros(W)              # overlap-add accumulator
        self.state = None
        self.hist = torch.zeros(model.order, model.n_bins, dtype=torch.complex64)  # newest first

    @torch.no_grad()
    def process(self, chunk: torch.Tensor) -> torch.Tensor:
        m = self.m; W, H = m.win, m.hop
        assert chunk.numel() == H
        self.inbuf = torch.cat([self.inbuf[H:], chunk])
        X = torch.fft.rfft(self.inbuf * m.window)
        coef, self.state = m.net(torch.log(X.abs() + 1e-5)[None, None], self.state)
        self.hist = torch.cat([X[None], self.hist[:-1]])
        y = torch.fft.irfft((self.hist.T * coef[0, 0]).sum(-1), n=W) * m.window
        self.ola = self.ola + y
        out = self.ola[:H].clone()
        self.ola = torch.cat([self.ola[H:], torch.zeros(H)])
        return out


def si_sdr_loss(est, ref, eps=1e-8):
    est = est - est.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    a = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    t = a * ref
    return -(10 * torch.log10(t.pow(2).sum(-1) / ((est - t).pow(2).sum(-1) + eps) + eps)).mean()
