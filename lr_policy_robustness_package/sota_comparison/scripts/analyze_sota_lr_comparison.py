"""Aggregate SOTA trainable-LR comparator logs.

Expected local layout after pulling remote logs:

    results/sota_lr_comparison/<server>/per_run/*.log
    results/sota_lr_comparison/<server>/top.log

The parser also accepts the direct remote-style folders copied under results:
    results/sota_adalrs_stars/per_run/*.log
    results/sota_bandit_industry/per_run/*.log
    results/sota_hypergrad_mlspace/per_run/*.log
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


EVAL_RE = re.compile(
    r"\[Eval\] Epoch:\s*(?P<epoch>\d+) \| "
    r"Teacher Test Loss:\s*(?P<teacher_loss>[-0-9.]+) \| "
    r"Teacher Test Acc:\s*(?P<teacher_acc>[-0-9.]+) \| "
    r"Student Test Loss:\s*(?P<student_loss>[-0-9.]+) \| "
    r"Student Test Acc:\s*(?P<student_acc>[-0-9.]+)"
)
LR_RE = re.compile(r"(?:next_student_lr|new_lr)=([-0-9.eE+]+)")
RUN_DONE_RE = re.compile(r"RUN_DONE\s+(\S+)\s+exit=(\d+)")

BASELINE_COSINE_SEED3 = 70.45
BASELINE_PATCHTST_BEST = 70.70
BASELINE_COSINE_MEAN = 70.197


def infer_method(name: str) -> str:
    if name.startswith("adalrs"):
        return "adalrs"
    if name.startswith("bandit_exp3"):
        return "bandit_exp3"
    if name.startswith("bandit_ucb"):
        return "bandit_ucb"
    if name.startswith("hypergrad_hb"):
        return "hypergrad_hb"
    return "unknown"


def infer_seed(name: str) -> int:
    match = re.search(r"seed(\d+)", name)
    return int(match.group(1)) if match else -1


def infer_variant(name: str) -> str:
    method = infer_method(name)
    if method == "bandit_ucb" and re.match(r"bandit_ucb_seed\d+_60ep$", name):
        return "ucb"
    prefix = {
        "adalrs": "adalrs_",
        "bandit_exp3": "bandit_exp3_",
        "bandit_ucb": "bandit_ucb_",
        "hypergrad_hb": "hypergrad_hb_",
    }.get(method, "")
    variant = name[len(prefix) :] if prefix and name.startswith(prefix) else name
    return re.sub(r"_seed\d+_60ep$", "", variant)


def parse_log(path: Path, server: str) -> dict | None:
    text = path.read_text(errors="replace")
    evals = []
    for match in EVAL_RE.finditer(text):
        evals.append(
            {
                "epoch": int(match.group("epoch")),
                "teacher_acc": float(match.group("teacher_acc")),
                "teacher_loss": float(match.group("teacher_loss")),
                "student_acc": float(match.group("student_acc")),
                "student_loss": float(match.group("student_loss")),
            }
        )
    if not evals:
        return None

    lr_values = [float(match.group(1)) for match in LR_RE.finditer(text)]
    name = path.stem
    final = max(evals, key=lambda item: item["epoch"])
    best = max(evals, key=lambda item: item["student_acc"])
    last10 = [item["student_acc"] for item in evals[-10:]]
    return {
        "server": server,
        "run": name,
        "method": infer_method(name),
        "variant": infer_variant(name),
        "seed": infer_seed(name),
        "n_epochs": len(evals),
        "final_epoch": final["epoch"],
        "final_student_acc": final["student_acc"],
        "final_student_loss": final["student_loss"],
        "best_student_acc": best["student_acc"],
        "best_student_epoch": best["epoch"],
        "last10_student_mean": float(np.mean(last10)),
        "last10_student_std": float(np.std(last10)),
        "final_teacher_acc": final["teacher_acc"],
        "last_lr": lr_values[-1] if lr_values else float("nan"),
        "min_lr": min(lr_values) if lr_values else float("nan"),
        "max_lr": max(lr_values) if lr_values else float("nan"),
        "delta_vs_cosine_seed3": final["student_acc"] - BASELINE_COSINE_SEED3,
        "delta_vs_patchtst_best": final["student_acc"] - BASELINE_PATCHTST_BEST,
        "delta_vs_cosine_mean": final["student_acc"] - BASELINE_COSINE_MEAN,
    }


def discover_logs(root: Path) -> list[tuple[str, Path]]:
    patterns = [
        ("stars", root / "sota_lr_comparison" / "stars" / "per_run"),
        ("industry", root / "sota_lr_comparison" / "industry" / "per_run"),
        ("mlspace", root / "sota_lr_comparison" / "mlspace" / "per_run"),
        ("stars", root / "sota_adalrs_stars" / "per_run"),
        ("industry", root / "sota_bandit_industry" / "per_run"),
        ("mlspace", root / "sota_hypergrad_mlspace" / "per_run"),
    ]
    logs = []
    for server, directory in patterns:
        if directory.exists():
            logs.extend((server, path) for path in sorted(directory.glob("*.log")))
    return logs


def write_outputs(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: (row["final_student_acc"], row["best_student_acc"]), reverse=True)
    if not rows:
        (out_dir / "SUMMARY.md").write_text("# SOTA LR comparison\n\nNo logs parsed.\n", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["variant"])].append(row)

    method_rows = []
    for (method, variant), items in sorted(grouped.items()):
        finals = np.array([item["final_student_acc"] for item in items], dtype=np.float64)
        last10 = np.array([item["last10_student_mean"] for item in items], dtype=np.float64)
        method_rows.append(
            {
                "method": method,
                "variant": variant,
                "n": len(items),
                "final_mean": float(finals.mean()),
                "final_std": float(finals.std(ddof=1)) if len(items) > 1 else 0.0,
                "last10_mean": float(last10.mean()),
                "last10_std": float(last10.std(ddof=1)) if len(items) > 1 else 0.0,
            }
        )
    method_rows.sort(key=lambda row: row["final_mean"], reverse=True)

    with (out_dir / "method_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(method_rows[0].keys()))
        writer.writeheader()
        writer.writerows(method_rows)

    lines = [
        "# SOTA Trainable LR Comparison",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "Baselines: cosine seed3 = 70.45; PatchTST headline = 70.70; cosine 3-seed mean = 70.197.",
        "",
        "## Verdict",
    ]
    best = rows[0]
    if best["final_student_acc"] >= BASELINE_PATCHTST_BEST:
        lines.append(
            f"Best run `{best['run']}` reached {best['final_student_acc']:.2f}, matching or beating the PatchTST headline."
        )
    elif best["final_student_acc"] >= BASELINE_COSINE_SEED3:
        lines.append(
            f"Best run `{best['run']}` reached {best['final_student_acc']:.2f}, beating cosine seed3 but not PatchTST headline."
        )
    else:
        lines.append(
            f"No parsed run beats cosine seed3 yet. Best run `{best['run']}` reached {best['final_student_acc']:.2f}."
        )
    lines.extend(
        [
            "",
            "## Method Summary",
            "",
            "| method | variant | n | final mean | final std | last10 mean | delta vs cosine mean |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in method_rows:
        lines.append(
            f"| {row['method']} | `{row['variant']}` | {row['n']} | "
            f"{row['final_mean']:.3f} | {row['final_std']:.3f} | "
            f"{row['last10_mean']:.3f} | {row['final_mean'] - BASELINE_COSINE_MEAN:+.3f} |"
        )

    lines.extend(
        [
            "",
            "## Top Runs",
            "",
            "| rank | server | run | final acc | final loss | best acc | last10 | vs cosine seed3 | vs PatchTST |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(rows[:20], start=1):
        lines.append(
            f"| {rank} | {row['server']} | `{row['run']}` | "
            f"{row['final_student_acc']:.2f} | {row['final_student_loss']:.3f} | "
            f"{row['best_student_acc']:.2f} | {row['last10_student_mean']:.2f} | "
            f"{row['delta_vs_cosine_seed3']:+.2f} | {row['delta_vs_patchtst_best']:+.2f} |"
        )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        labels = [f"{row['method']}\n{row['variant']}" for row in method_rows]
        values = [row["final_mean"] for row in method_rows]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(range(len(values)), values)
        ax.axhline(BASELINE_COSINE_SEED3, color="black", linestyle="--", linewidth=1, label="cosine seed3")
        ax.axhline(BASELINE_PATCHTST_BEST, color="gray", linestyle=":", linewidth=1.2, label="PatchTST headline")
        ax.set_ylabel("Final student test accuracy")
        ax.set_title("SOTA LR comparator variants")
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "method_final_accuracy.png", dpi=180)
        plt.close(fig)
    except Exception as exc:  # plotting is optional on headless environments
        (out_dir / "plot_error.txt").write_text(str(exc), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/sota_lr_comparison"))
    args = parser.parse_args()

    rows = []
    for server, path in discover_logs(args.results_root):
        parsed = parse_log(path, server)
        if parsed is not None:
            rows.append(parsed)
    write_outputs(rows, args.out_dir)
    print(f"parsed {len(rows)} runs -> {args.out_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
