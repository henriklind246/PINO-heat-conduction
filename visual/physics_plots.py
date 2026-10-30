"""Direct-state gate diagnostics."""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_direct_state_gate(run_dir: str | Path):
    """Compare the frozen direct-state probes without averaging unlike modes."""
    root = Path(run_dir)
    with (root / "diagnostics.csv").open() as file:
        rows = list(csv.DictReader(file))
    grid = max(int(r["grid_size"]) for r in rows)
    selected = [r for r in rows if int(r["grid_size"]) == grid and float(r["time_n"]) == 0]
    probes = tuple(dict.fromkeys(r["probe"] for r in selected))
    families = tuple(dict.fromkeys(r["ic_family"] for r in selected))
    fig, axes = plt.subplots(len(probes), 2, figsize=(11, 3 * len(probes)), squeeze=False)
    for idx, probe in enumerate(probes):
        for objective in ("raw", "exact"):
            part = [r for r in selected if r["probe"] == probe and r["objective"] == objective]
            steps = sorted({int(r["update"]) for r in part})
            for ax, metric in zip(axes[idx], ("state_rmse_K", "raw_residual_rms")):
                values = [np.median([float(r[metric]) for r in part if int(r["update"]) == step]) for step in steps]
                ax.semilogy(steps, values, label=objective)
                ax.set_xlabel("Adam updates")
                ax.set_ylabel("State RMS error (K)" if metric == "state_rmse_K" else "Raw residual RMS")
                ax.set_title(probe)
                ax.legend()
    fig.suptitle(f"First interval, {grid}×{grid}; median across {len(families)} prescribed IC families")
    fig.tight_layout()
    path = root / "direct_state_convergence.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
