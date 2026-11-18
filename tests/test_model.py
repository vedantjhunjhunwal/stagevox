import pytest
import torch

from stagevox.model import Separator


@pytest.mark.parametrize("win", [64, 128, 256, 512])
def test_streaming_matches_offline(win):
    torch.manual_seed(0)
    m = Separator(win).eval()
    x = torch.randn(1, 64 * win)
    with torch.no_grad():
        y = m(x)[0]
        st = m.stream()
        s = torch.cat([st.process(c) for c in x[0].split(m.hop)])
    # streaming output is the offline output delayed by one hop (plus one hop of buffering)
    assert torch.allclose(s[m.hop:], y[:-m.hop], atol=1e-5)


@pytest.mark.parametrize("win", [128, 512])
def test_no_lookahead_beyond_window(win):
    """Changing input at sample k must not change any output before k - win + 1."""
    torch.manual_seed(1)
    m = Separator(win).eval()
    x = torch.randn(1, 40 * win)
    k = 20 * win + 7
    x2 = x.clone(); x2[0, k:] += torch.randn(x.shape[1] - k)
    with torch.no_grad():
        y1, y2 = m(x)[0], m(x2)[0]
    assert torch.equal(y1[: k - win + 1], y2[: k - win + 1])


def test_identity_mask_reconstructs_input():
    m = Separator(256).eval()
    with torch.no_grad():
        m.net.out.weight.zero_(); b = torch.zeros(m.n_bins, m.order, 2)
        b[:, 0, 0] = 40.0                                                 # tap0 = tanh(40) ~ 1, rest 0
        m.net.out.bias.copy_(b.flatten())
        x = torch.randn(2, 16000)
        assert torch.allclose(m(x), x, atol=1e-4)


def test_latency_property():
    assert Separator(128, sr=16000).latency_ms == 8.0
