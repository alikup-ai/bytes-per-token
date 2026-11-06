"""Fit each engine's cycle costs on its calibration runs, then predict the
held-out runs it has never seen. Exit status 1 if any held-out prediction is
off by more than TOL.

    python costmodel/validate.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).parent))
from arch import ARCHS  # noqa: E402
from model import STAGES, THETA, run_features  # noqa: E402

TOL = 0.15
HERE = Path(__file__).parent


def load():
    hw = json.loads((HERE / "hw" / "esp32s3.json").read_text())
    engines = json.loads((HERE / "engines.json").read_text())
    runs = json.loads((HERE / "measured.json").read_text())["runs"]
    return hw, engines, runs


def fit(engine, runs, hw, engines):
    rows, ys = [], []
    for r in runs:
        if r["engine"] != engine or r["role"] != "calibrate":
            continue
        A, b = run_features(r, ARCHS, engines, hw)
        for i, s in enumerate(STAGES):
            rows.append(A[i])
            ys.append(r["stages_ms"][s] / 1e3 - b[i])
    A, y = np.array(rows), np.array(ys)
    used = A.any(axis=0)             # a cost no calibration op exercises stays 0
    theta = np.zeros(len(THETA))
    # Scale columns so NNLS is well conditioned across ~1e-9..1e-3 magnitudes.
    scale = np.abs(A[:, used]).max(axis=0)
    sol, _ = nnls(A[:, used] / scale, y)
    theta[used] = sol / scale
    return theta


def predict(run, theta, hw, engines):
    A, b = run_features(run, ARCHS, engines, hw)
    return (A @ theta + b) * 1e3


def main():
    hw, engines, runs = load()
    failed = False
    for engine in sorted({r["engine"] for r in runs}):
        theta = fit(engine, runs, hw, engines)
        print(f"\n== {engine}")
        for name, v in zip(THETA, theta):
            shown = f"{1 / v / 1e6:.1f} MB/s" if name == "s_per_flash_byte" and v else f"{v:.3g}"
            print(f"   {name:18s} {shown}")
        for r in (r for r in runs if r["engine"] == engine):
            p = predict(r, theta, hw, engines)
            tag = "CAL " if r["role"] == "calibrate" else "HELD"
            if "stages_ms" in r:
                meas = np.array([r["stages_ms"][s] for s in STAGES])
                cells = "  ".join(f"{s} {m:5.1f}/{q:5.1f}" for s, m, q in zip(STAGES, meas, p))
                tot_m, tot_p = meas.sum(), p.sum()
            else:
                cells, tot_m, tot_p = "(total only)", r["total_ms"], p.sum()
            err = tot_p / tot_m - 1
            print(f"   {tag} {r['id']:24s} total {tot_m:6.1f} meas / {tot_p:6.1f} pred ({err:+.1%})")
            print(f"        measured/predicted ms: {cells}")
            if r["role"] == "validate":
                if "stages_ms" in r:
                    worst = np.max(np.abs(p / meas - 1))
                    print(f"        worst stage error {worst:.1%}")
                    failed |= worst > TOL
                failed |= abs(err) > TOL
            if "mean_pos" in r:
                sweep = []
                for mp in (8, 16, 24, 32, 48):
                    q = dict(r, mean_pos=mp)
                    sweep.append(f"pos {mp}: {predict(q, theta, hw, engines).sum():.1f}")
                print("        mean_pos sensitivity: " + ", ".join(sweep))
    print("\nRESULT:", "FAIL (held-out error > 15%)" if failed else "PASS (held-out within 15%)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
