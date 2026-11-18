"""Generate train / val / test scene sets.

Train + val: vocal = synthetic singer or real VCTK speech (training speakers), band =
synthetic stems (+ some real instrumental music from the TRAIN pool).  test_synth:
synthetic singer + synthetic band only.  test_real: real LibriSpeech speech as the vocal and
real music from a disjoint HELD-OUT pool as the bleed source; never seen in training.

usage: python scripts/make_dataset.py --real-audio <librosa-data/audio> --speech <MS-SNSD/clean_train> --out data
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stagevox import SR, scenes, synth  # noqa: E402

DUR = 4.0
TRAIN_MUSIC = ["147793__setuniman__sweet-waltz-0i-22mi.ogg",
               "442789__lena-orsa__happy-music-pistachio-ice-cream-ragtime.ogg",
               "Kevin_MacLeod_-_P_I_Tchaikovsky_Dance_of_the_Sugar_Plum_Fairy.ogg"]
TEST_MUSIC = ["Kevin_MacLeod_-_Vibe_Ace.ogg", "admiralbob77_-_Choice_-_Drum-bass.ogg",
              "Hungarian_Dance_number_5_-_Allegro_in_F_sharp_minor_(string_orchestra).ogg"]
SPEECH = ["198-209-0000.ogg", "3436-172162-0000.ogg", "5703-47212-0000.ogg"]

_MUSIC = {}
_SPEECH = []


def load(path):
    x, sr = sf.read(path, dtype="float64")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        g = np.gcd(sr, SR); x = resample_poly(x, SR // g, sr // g)
    return x


def crop(rng, x, n):
    if len(x) <= n:
        return np.pad(x, (0, n - len(x)))
    s = rng.integers(0, len(x) - n); return x[s:s + n]


def synth_scene(args):
    seed, kind, use_real = args
    rng = np.random.default_rng(seed)
    n = int(DUR * SR)
    bpm = rng.uniform(80, 140)
    if _SPEECH and rng.random() < 0.5:                       # real speech half the time
        v = speech_track(rng, n)
    else:
        v = synth.singing_voice(DUR, SR, rng, bpm)
    stems = list(synth.band(DUR, SR, rng, bpm).values())
    if use_real and _MUSIC.get("train") and rng.random() < 0.3:
        stems.append(crop(rng, _MUSIC["train"][rng.integers(len(_MUSIC["train"]))], n))
    sc = scenes.make_scene(v, stems, SR, rng, kind=kind)
    return sc.mix, sc.target, sc.meta["kind"]


def speech_track(rng, n):
    """Concatenate random training-speaker utterances (with short pauses) to fill n samples."""
    out = np.zeros(n); i = int(rng.uniform(0, 0.5) * SR)
    while i < n:
        u = _SPEECH[rng.integers(len(_SPEECH))]
        u = u[: n - i]; out[i:i + len(u)] = u / (np.max(np.abs(u)) + 1e-9)
        i += len(u) + int(rng.uniform(0.1, 0.6) * SR)
    return out


def init(real_dir, speech_dir=None):
    if real_dir:
        _MUSIC["train"] = [load(os.path.join(real_dir, f)) for f in TRAIN_MUSIC]
    if speech_dir:
        import glob
        _SPEECH[:] = [load(f) for f in sorted(glob.glob(os.path.join(speech_dir, "*.wav")))]


def build(seeds, kinds, use_real, real_dir, workers, speech_dir=None):
    with Pool(workers, initializer=init, initargs=(real_dir, speech_dir)) as p:
        out = p.map(synth_scene, list(zip(seeds, kinds, [use_real] * len(seeds))), chunksize=8)
    mix = np.stack([o[0] for o in out]); tgt = np.stack([o[1] for o in out])
    return mix, tgt, np.array([o[2] for o in out])


def real_test(real_dir, seed=12345):
    rng = np.random.default_rng(seed)
    n = int(DUR * SR)
    speech = [load(os.path.join(real_dir, f)) for f in SPEECH]
    music = [load(os.path.join(real_dir, f)) for f in TEST_MUSIC]
    mixes, tgts, kinds = [], [], []
    for sp in speech:
        for s in range(0, len(sp) - n + 1, n // 2):          # 50%-overlapping 4 s windows
            seg = sp[s:s + n]
            for kind in scenes.SCENE_TYPES:
                m = music[rng.integers(len(music))]
                sc = scenes.make_scene(seg, [crop(rng, m, n)], SR, rng, kind=kind)
                mixes.append(sc.mix); tgts.append(sc.target); kinds.append(kind)
    return np.stack(mixes), np.stack(tgts), np.array(kinds)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-audio", required=True)
    ap.add_argument("--speech", required=True, help="dir of clean training-speaker wavs (VCTK via MS-SNSD)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-train", type=int, default=1600)
    ap.add_argument("--n-val", type=int, default=96)
    ap.add_argument("--n-test", type=int, default=60, help="per scene type")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W = a.workers
    m, t, k = build(range(a.n_train), [None] * a.n_train, True, a.real_audio, W, a.speech)
    np.savez(os.path.join(a.out, "train.npz"), mix=m, target=t, kind=k)
    m, t, k = build(range(10**6, 10**6 + a.n_val), [None] * a.n_val, True, a.real_audio, W, a.speech)
    np.savez(os.path.join(a.out, "val.npz"), mix=m, target=t, kind=k)
    kinds = [kk for kk in scenes.SCENE_TYPES for _ in range(a.n_test)]
    m, t, k = build(range(2 * 10**6, 2 * 10**6 + len(kinds)), kinds, False, None, W)
    np.savez(os.path.join(a.out, "test_synth.npz"), mix=m, target=t, kind=k)
    m, t, k = real_test(a.real_audio)
    np.savez(os.path.join(a.out, "test_real.npz"), mix=m, target=t, kind=k)
    print("done", {f: np.load(os.path.join(a.out, f))["mix"].shape for f in os.listdir(a.out)})
