"""
Day 2, Step 2: temperature scaling + calibration metrics.

Fits ONE scalar T on the validation logits (minimising negative log-likelihood),
then reports ECE, Brier score, NLL and accuracy BEFORE and AFTER on val and test.
Temperature scaling never changes the predicted class (argmax is unchanged),
so accuracy stays identical; only the confidence values change.

Inputs : eval/results/v2_logits_val.json, v2_logits_test.json
Outputs: eval/results/calibration.json, eval/results/reliability.png (if matplotlib is installed)

Run: python scripts/calibrate.py
"""
import json
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval" / "results"
N_BINS = 10


def softmax(z, T=1.0):
    z = np.asarray(z, dtype=float) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits, y, T):
    p = softmax(logits, T)
    return float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))


def fit_temperature(logits, y):
    """Grid search (log-spaced) followed by a local refinement. No scipy needed."""
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 400))
    best = min(grid, key=lambda T: nll(logits, y, T))
    fine = np.linspace(best * 0.9, best * 1.1, 200)
    return float(min(fine, key=lambda T: nll(logits, y, T)))


def ece(probs, y, n_bins=N_BINS):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def brier(probs, y):
    onehot = np.eye(probs.shape[1])[y]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def metrics(logits, y, T):
    p = softmax(logits, T)
    return {"ece": ece(p, y), "brier": brier(p, y), "nll": nll(logits, y, T),
            "accuracy": float((p.argmax(1) == y).mean()),
            "mean_confidence": float(p.max(1).mean())}


def load(name):
    with open(RESULTS_DIR / f"v2_logits_{name}.json") as f:
        d = json.load(f)
    return np.array([r["logits"] for r in d]), np.array([r["label_id"] for r in d])


def reliability_plot(logits, y, T, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping reliability.png (pip install matplotlib)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, (title, t) in zip(axes, [("Before (T=1)", 1.0), (f"After (T={T:.2f})", T)]):
        p = softmax(logits, t)
        conf, correct = p.max(1), (p.argmax(1) == y).astype(float)
        edges = np.linspace(0, 1, N_BINS + 1)
        xs, accs = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.any():
                xs.append(conf[m].mean())
                accs.append(correct[m].mean())
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.plot(xs, accs, "o-")
        ax.set_title(f"{title}  ECE={ece(p, y):.3f}")
        ax.set_xlabel("confidence")
    axes[0].set_ylabel("accuracy")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"saved {path}")


def main():
    lv, yv = load("val")
    lt, yt = load("test")
    T = fit_temperature(lv, yv)
    print(f"Temperature fitted on VAL: T = {T:.3f}  (T>1 means the model was overconfident)\n")

    out = {"temperature": T, "n_val": int(len(yv)), "n_test": int(len(yt))}
    print(f"{'split':6s} {'stage':7s} {'ECE':>7s} {'Brier':>7s} {'NLL':>7s} {'acc':>7s} {'meanConf':>9s}")
    for split, lg, y in [("val", lv, yv), ("test", lt, yt)]:
        for stage, t in [("before", 1.0), ("after", T)]:
            m = metrics(lg, y, t)
            out[f"{split}_{stage}"] = m
            print(f"{split:6s} {stage:7s} {m['ece']:7.4f} {m['brier']:7.4f} {m['nll']:7.4f} "
                  f"{m['accuracy']:7.4f} {m['mean_confidence']:9.4f}")
    with open(RESULTS_DIR / "calibration.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved eval/results/calibration.json")
    reliability_plot(lt, yt, T, RESULTS_DIR / "reliability.png")


if __name__ == "__main__":
    main()