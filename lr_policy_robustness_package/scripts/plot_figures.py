#!/usr/bin/env python
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
fig_dir = ROOT / 'figures/robust12h'
fig_dir.mkdir(parents=True, exist_ok=True)
try:
    import matplotlib.pyplot as plt
except Exception as exc:
    print(f'matplotlib unavailable: {exc}')
    raise SystemExit(0)
per_run = pd.read_csv(ROOT / 'data/robust12h/per_run.csv')
pairs = pd.read_csv(ROOT / 'data/robust12h/pairs.csv')
shock = per_run[(per_run.experiment == 'shock') & (per_run.perturb_key == 'shock_lr=0.10')]
pivot = shock.pivot_table(index='seed', columns='method', values='final_student_acc', aggfunc='first')
ax = pivot[['cosine','patchtst']].plot(kind='bar', figsize=(7,4))
ax.set_ylabel('Student test accuracy')
ax.set_title('LR-down shock01: paired seeds')
plt.tight_layout()
plt.savefig(fig_dir / 'shock01_winning_seeds.png', dpi=200)
plt.close()
ax = pairs.sort_values('delta_pp').plot(kind='barh', x='perturb_key', y='delta_pp', legend=False, figsize=(8,6))
ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel('PatchTST/residual policy delta vs cosine (pp)')
ax.set_title('Robustness paired deltas')
plt.tight_layout()
plt.savefig(fig_dir / 'paired_deltas.png', dpi=200)
plt.close()
print(f'wrote figures to {fig_dir}')
