"""Train the ink-detection U-Net on the official Vesuvius Challenge fragments.

Trains on some fragments and validates on a whole held-out fragment (never a held-out crop of a
training fragment, which would flatter the score), keeping the checkpoint with the best F0.5.

Usage:
    python -m src.m4_ink_detection.train_fragments --data data/real_scans/fragments \
        --train frag1 frag2 --val frag3 --out data/processed/models/ink_unet.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.m4_ink_detection.fragments import DEFAULT_N_SLICES, DEFAULT_Z_START, Fragment, intensity_stats, load_fragment
from src.m4_ink_detection.infer import best_fbeta, normalize_stack, predict_stack


class CropSampler:
    """Random training crops, half centred on ink so the (rare) positive class is seen often."""

    def __init__(self, fragments: list[Fragment], crop: int, ink_fraction: float = 0.5, seed: int = 0):
        self.fragments, self.crop, self.ink_fraction = fragments, crop, ink_fraction
        self.rng = np.random.default_rng(seed)
        self.ink_pts = [np.argwhere(f.labels[::4, ::4]) * 4 for f in fragments]
        self.paper_pts = [np.argwhere(f.mask[::8, ::8]) * 8 for f in fragments]
        areas = np.array([len(p) for p in self.paper_pts], dtype=np.float64)
        self.frag_p = areas / areas.sum()

    def sample(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        i = int(self.rng.choice(len(self.fragments), p=self.frag_p))
        frag = self.fragments[i]
        pts = self.ink_pts[i] if (len(self.ink_pts[i]) and self.rng.random() < self.ink_fraction) else self.paper_pts[i]
        cy, cx = pts[self.rng.integers(len(pts))] + self.rng.integers(-self.crop // 3, self.crop // 3 + 1, size=2)
        h, w = frag.shape
        y = int(np.clip(cy - self.crop // 2, 0, h - self.crop))
        x = int(np.clip(cx - self.crop // 2, 0, w - self.crop))
        sl = (slice(y, y + self.crop), slice(x, x + self.crop))
        return np.asarray(frag.stack[:, sl[0], sl[1]]), frag.labels[sl], frag.mask[sl]


def _augment(x: np.ndarray, y: np.ndarray, m: np.ndarray, rng: np.random.Generator):
    k = int(rng.integers(4))
    x, y, m = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k), np.rot90(m, k)
    if rng.random() < 0.5:
        x, y, m = x[:, :, ::-1], y[:, ::-1], m[:, ::-1]
    x = x * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)  # contrast / brightness jitter (after normalisation)
    return np.ascontiguousarray(x), np.ascontiguousarray(y), np.ascontiguousarray(m)


def make_batch(sampler: CropSampler, batch: int, mean: float, std: float):
    xs, ys, ms = [], [], []
    for _ in range(batch):
        raw, lab, mask = sampler.sample()
        x, y, m = _augment(normalize_stack(raw, mean, std), lab, mask, sampler.rng)
        xs.append(x), ys.append(y), ms.append(m)
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32), np.stack(ms).astype(np.float32)


def masked_loss(logits, target, mask):
    import torch
    import torch.nn.functional as F

    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * mask).sum() / mask.sum().clamp_min(1)
    p = torch.sigmoid(logits) * mask
    inter = (p * target).sum()
    dice = 1 - (2 * inter + 1) / (p.sum() + (target * mask).sum() + 1)
    return bce + dice


def evaluate(model, frag: Fragment, mean: float, std: float, tile: int, stride: int) -> dict:
    prob = predict_stack(model, frag.stack, mean, std, mask=frag.mask, tile=tile, stride=stride)
    return {**best_fbeta(prob, frag.labels, frag.mask), "_prob": prob}


def classical_baseline(frag: Fragment, mean: float, std: float) -> dict:
    """Mean-over-slices brightness as an 'ink score', scored the same way - what a plain density heuristic achieves."""
    score = np.zeros(frag.shape, dtype=np.float32)
    for z in range(frag.stack.shape[0]):
        score += normalize_stack(np.asarray(frag.stack[z]), mean, std)
    score /= frag.stack.shape[0]
    lo, hi = np.percentile(score[frag.mask], [1, 99])
    return best_fbeta(np.clip((score - lo) / (hi - lo + 1e-6), 0, 1), frag.labels, frag.mask)


def train(
    data: Path,
    train_names: list[str],
    val_name: str | None,
    out: Path,
    steps: int = 6000,
    batch: int = 12,
    crop: int = 256,
    lr: float = 1e-3,
    base: int = 32,
    eval_every: int = 1000,
    z_start: int = DEFAULT_Z_START,
    n_slices: int = DEFAULT_N_SLICES,
    seed: int = 0,
) -> dict:
    import torch

    from src.m4_ink_detection.unet import InkUNet

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_frags = [load_fragment(data / n, z_start, n_slices) for n in train_names]
    val_frag = load_fragment(data / val_name, z_start, n_slices) if val_name else None
    mean, std = intensity_stats(train_frags)
    print(f"train {train_names} val {val_name} | device {device} | mean {mean:.4f} std {std:.4f}", flush=True)
    for f in train_frags + ([val_frag] if val_frag else []):
        print(f"  {f.name}: {f.shape[1]}x{f.shape[0]} px, papyrus {f.mask.mean():.0%}, ink {f.labels[f.mask].mean():.1%} of papyrus", flush=True)

    model = InkUNet(n_slices, base).to(device)
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    sampler = CropSampler(train_frags, crop, seed=seed)

    out.parent.mkdir(parents=True, exist_ok=True)
    history, best = [], {"f_beta": -1.0}
    t0 = time.time()
    running = 0.0
    for step in range(1, steps + 1):
        x, y, m = (torch.from_numpy(a).to(device) for a in make_batch(sampler, batch, mean, std))
        model.train()
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            loss = masked_loss(model(x).float(), y, m)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        running += loss.item()

        if step % 100 == 0:
            print(f"  step {step}/{steps}  loss {running / 100:.4f}  ({time.time() - t0:.0f}s)", flush=True)
            running = 0.0
        if val_frag is not None and (step % eval_every == 0 or step == steps):
            res = evaluate(model, val_frag, mean, std, crop, crop * 3 // 4)
            res.pop("_prob")
            history.append({"step": step, **res})
            print(f"  VAL step {step}: F0.5 {res['f_beta']:.3f} (precision {res['precision']:.3f}, recall {res['recall']:.3f}, thr {res['threshold']:.2f})", flush=True)
            if res["f_beta"] > best["f_beta"]:
                best = {"step": step, **res}
                _save(out, model, mean, std, z_start, n_slices, base, crop, best, train_names, val_name)
    if val_frag is None:
        _save(out, model, mean, std, z_start, n_slices, base, crop, {}, train_names, val_name)

    report = {"best": best, "history": history, "train": train_names, "val": val_name, "steps": steps, "seconds": time.time() - t0}
    if val_frag is not None:
        report["classical_baseline"] = classical_baseline(val_frag, mean, std)
        report["all_ink_baseline_f0.5"] = _all_ink_f(best["ink_fraction"])
        print(f"classical density baseline F0.5 {report['classical_baseline']['f_beta']:.3f}  |  predict-everything-is-ink F0.5 {report['all_ink_baseline_f0.5']:.3f}", flush=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    return report


def _all_ink_f(prevalence: float, beta: float = 0.5) -> float:
    """F-beta of labelling every papyrus pixel as ink (precision = prevalence, recall = 1)."""
    b2 = beta**2
    return float((1 + b2) * prevalence / (b2 * prevalence + 1))


def _save(out, model, mean, std, z_start, n_slices, base, crop, metrics, train_names, val_name) -> None:
    import torch

    torch.save(
        {"state_dict": model.state_dict(), "mean": mean, "std": std, "z_start": z_start, "n_slices": n_slices, "base": base,
         "tile": crop, "metrics": metrics, "train": train_names, "val": val_name},
        out,
    )


def load_model(path: str | Path, device: str | None = None):
    """Load a checkpoint written by :func:`train`; returns ``(model, meta)``."""
    import torch

    from src.m4_ink_detection.unet import InkUNet

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = InkUNet(ckpt["n_slices"], ckpt["base"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, {k: v for k, v in ckpt.items() if k != "state_dict"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/real_scans/fragments")
    parser.add_argument("--train", nargs="+", default=["frag1", "frag2"])
    parser.add_argument("--val", default="frag3", help="held-out fragment ('' to train on everything, unvalidated)")
    parser.add_argument("--out", default="data/processed/models/ink_unet.pt")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=1000)
    args = parser.parse_args()
    train(Path(args.data), args.train, args.val or None, Path(args.out), args.steps, args.batch, args.crop, args.lr, args.base, args.eval_every)


if __name__ == "__main__":
    main()
