"""Regenerate SOTA comparator figures from archived CSV tables."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    method_rows = load_rows(ROOT / "tables" / "method_summary.csv")
    run_rows = load_rows(ROOT / "tables" / "summary.csv")
    for row in method_rows:
        for key in ["final_mean", "final_std", "last10_mean"]:
            row[key] = float(row[key])
    for row in run_rows:
        for key in ["final_student_acc", "best_student_acc", "last10_student_mean"]:
            row[key] = float(row[key])

    labels = [f"{row['method']}\n{row['variant']}" for row in method_rows]
    values = [row["final_mean"] for row in method_rows]
    errors = [row["final_std"] for row in method_rows]
    fig, ax = plt.subplots(figsize=(13, 6), dpi=160)
    ax.bar(range(len(values)), values, yerr=errors, capsize=3)
    ax.axhline(70.45, color="black", linestyle="--", linewidth=1.2, label="cosine seed3 70.45")
    ax.axhline(70.70, color="tab:red", linestyle=":", linewidth=1.4, label="PatchTST headline 70.70")
    ax.axhline(70.197, color="gray", linestyle="-.", linewidth=1.0, label="cosine 3-seed mean 70.197")
    ax.set_ylabel("Final student test accuracy")
    ax.set_title("External adaptive LR comparators vs PatchTST output ensemble")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylim(20, 73)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sota_method_final_accuracy.png", bbox_inches="tight")
    plt.close(fig)

    sorted_runs = sorted(run_rows, key=lambda row: row["final_student_acc"], reverse=True)[:12]
    fig, ax = plt.subplots(figsize=(13, 6), dpi=160)
    y_positions = list(range(len(sorted_runs)))[::-1]
    ax.barh(y_positions, [row["final_student_acc"] for row in sorted_runs])
    ax.axvline(70.45, color="black", linestyle="--", linewidth=1.2, label="cosine seed3")
    ax.axvline(70.70, color="tab:red", linestyle=":", linewidth=1.4, label="PatchTST headline")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([row["run"].replace("_60ep", "") for row in sorted_runs], fontsize=8)
    ax.set_xlabel("Final student test accuracy")
    ax.set_title("Top SOTA-comparator runs")
    ax.set_xlim(30, 72)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sota_top_runs.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {FIG_DIR / 'sota_method_final_accuracy.png'}")
    print(f"Wrote {FIG_DIR / 'sota_top_runs.png'}")


if __name__ == "__main__":
    main()
