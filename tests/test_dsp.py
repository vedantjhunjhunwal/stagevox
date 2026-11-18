import numpy as np
import pytest
from scipy import signal

from stagevox import SR, fx, scenes, synth
from stagevox.metrics import si_sdr
from stagevox.notch_suppressor import NotchFeedbackSuppressor


def tone(f, dur=1.0, a=0.5):
    return a * np.sin(2 * np.pi * f * np.arange(int(dur * SR)) / SR)


@pytest.mark.parametrize("kind", ["lowpass", "highpass", "bandpass", "notch", "peak", "lowshelf", "highshelf"])
def test_biquads_are_stable(kind):
    b, a = fx.biquad(kind, 1000, SR, q=2.0, gain_db=6.0)
    assert np.all(np.abs(np.roots(a)) < 1.0)


def test_notch_removes_tone():
    y = fx.filt(tone(1000, 2.0), "notch", 1000, SR, q=30)
    assert 20 * np.log10(fx.rms(y[SR:]) / fx.rms(tone(1000, 1.0))) < -30


def test_peak_eq_gain_matches_spec():
    b, a = fx.biquad("peak", 2000, SR, q=1.0, gain_db=6.0)
    _, h = signal.freqz(b, a, worN=[2000], fs=SR)
    assert abs(20 * np.log10(abs(h[0])) - 6.0) < 0.05


def test_compressor_reduces_loud_parts_only():
    x = np.concatenate([tone(440, 1, 0.05), tone(440, 1, 0.9)])
    y = fx.compressor(x, SR, threshold_db=-20, ratio=4, attack_ms=1, release_ms=50)
    assert abs(fx.rms(y[:SR // 2]) - fx.rms(x[:SR // 2])) / fx.rms(x[:SR // 2]) < 0.02
    over = 20 * np.log10(0.9) + 20                                      # 19.1 dB over threshold
    gr = 20 * np.log10(fx.rms(x[-SR // 2:]) / fx.rms(y[-SR // 2:]))
    assert abs(gr - over * (1 - 1 / 4)) < 1.0


def test_feedback_loop_is_stable_and_rings():
    rng = np.random.default_rng(0)
    h, d = scenes.feedback_path(SR, rng)
    assert np.all(h[:d] == 0)
    imp = np.zeros(4 * SR); imp[0] = 1.0
    r = scenes.acoustic_feedback(imp, h, 0.95)
    assert np.all(np.isfinite(r))
    assert fx.rms(r[-SR // 2:]) < fx.rms(r[: SR // 2])                    # decays, does not explode
    assert fx.rms(r[SR // 10: SR // 2]) > 3 * fx.rms(scenes.acoustic_feedback(imp, h, 0.5)[SR // 10: SR // 2])


def test_scene_target_is_clean_and_deterministic():
    def mk(seed):
        rng = np.random.default_rng(seed)
        v = synth.singing_voice(2.0, SR, rng)
        return scenes.make_scene(v, list(synth.band(2.0, SR, rng).values()), SR, rng, kind="bleed+feedback")
    a, b = mk(3), mk(3)
    np.testing.assert_array_equal(a.mix, b.mix)
    assert si_sdr(a.mix, a.target) < 25


def test_notch_suppressor_finds_howl_and_spares_vibrato():
    rng = np.random.default_rng(1)
    t = np.arange(3 * SR) / SR
    vib = 0.3 * np.sin(2 * np.pi * (220 * t + 3 * np.sin(2 * np.pi * 5.5 * t)))   # sung note with vibrato
    howl = 0.3 * np.sin(2 * np.pi * 2500 * t) * np.minimum(1, t / 0.5)
    sup = NotchFeedbackSuppressor(SR)
    y = sup(vib + howl + 0.003 * rng.standard_normal(len(t)))
    assert any(abs(f - 2500) < 40 for f in sup.notch_hz)
    assert not any(abs(f - 220) < 40 for f in sup.notch_hz)
    assert si_sdr(y[2 * SR:], vib[2 * SR:]) > si_sdr((vib + howl)[2 * SR:], vib[2 * SR:]) + 20


def test_saturating_loop_howls_above_unity_and_stays_bounded():
    rng = np.random.default_rng(2)
    h, d = scenes.feedback_path(SR, rng)
    x = 0.01 * rng.standard_normal(3 * SR)
    quiet = scenes.acoustic_feedback(x, h, 0.9, sat_level=0.3)
    howl = scenes.acoustic_feedback(x, h, 1.2, sat_level=0.3)
    assert np.all(np.isfinite(howl)) and np.abs(howl).max() < 2.0          # bounded by the amp limit
    assert fx.rms(howl[-SR:]) > 10 * fx.rms(quiet[-SR:])                     # self-sustaining tone
    spec = np.abs(np.fft.rfft(howl[-SR:]))
    assert spec.max() > 20 * np.median(spec)                                 # narrowband


def test_linear_and_saturating_loops_agree_when_quiet():
    rng = np.random.default_rng(4)
    h, _ = scenes.feedback_path(SR, rng)
    x = 1e-3 * rng.standard_normal(SR)
    a = scenes.acoustic_feedback(x, h, 0.8)
    b = scenes.acoustic_feedback(x, h, 0.8, sat_level=10.0)
    assert np.allclose(a, b, atol=1e-6)
