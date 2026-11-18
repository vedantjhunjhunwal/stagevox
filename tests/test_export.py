import numpy as np
import pytest
import torch

from stagevox.export import NetStep, StreamStep
from stagevox.model import Separator


def test_torchscript_stream_step_matches_python_streaming():
    torch.manual_seed(0)
    sep = Separator(128).eval()
    step = torch.jit.script(StreamStep(sep).eval())
    x = torch.randn(64 * 100) * 0.1
    st = sep.stream()
    ref = torch.cat([st.process(c) for c in x.split(sep.hop)])
    state = step.initial_state(); outs = []
    with torch.no_grad():
        for c in x.split(sep.hop):
            o, *state = step(c, *state); outs.append(o)
    assert torch.allclose(torch.cat(outs), ref, atol=1e-6)


def test_onnx_net_step_matches_pytorch(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    torch.manual_seed(1)
    sep = Separator(128).eval()
    net = NetStep(sep).eval()
    H = sep.net.gru.hidden_size
    args = (torch.zeros(1, 1, sep.n_bins), torch.zeros(1, H, 2), torch.zeros(2, 1, H))
    path = str(tmp_path / "net.onnx")
    torch.onnx.export(net, args, path, dynamo=False, opset_version=17,
                      input_names=["feats", "conv_buf", "gru_h"],
                      output_names=["coef_re", "coef_im", "new_conv_buf", "new_gru_h"])
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    buf, h = np.zeros((1, H, 2), np.float32), np.zeros((2, 1, H), np.float32)
    tb, th = torch.zeros(1, H, 2), torch.zeros(2, 1, H)
    for _ in range(50):                                   # recurrent state carried across steps
        f = rng.standard_normal((1, 1, sep.n_bins)).astype(np.float32)
        r = s.run(None, {"feats": f, "conv_buf": buf, "gru_h": h})
        with torch.no_grad():
            p = net(torch.from_numpy(f), tb, th)
        assert np.allclose(r[0], p[0].numpy(), atol=1e-5) and np.allclose(r[1], p[1].numpy(), atol=1e-5)
        buf, h, tb, th = r[2], r[3], p[2], p[3]
