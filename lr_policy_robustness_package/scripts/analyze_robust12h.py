#!/usr/bin/env python
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
pairs = pd.read_csv(ROOT / 'data/robust12h/pairs.csv')
per_run = pd.read_csv(ROOT / 'data/robust12h/per_run.csv')
out = ROOT / 'reports/robust12h_PACKAGE_SUMMARY.md'


def markdown_table(df):
    cols = list(df.columns)
    rows = ['| ' + ' | '.join(cols) + ' |']
    rows.append('| ' + ' | '.join(['---'] * len(cols)) + ' |')
    for _, row in df.iterrows():
        rows.append('| ' + ' | '.join(str(row[col]) for col in cols) + ' |')
    return '\n'.join(rows)


lines = ['# Package Robust12h Summary', '']
lines.append('## Paired deltas')
paired_cols = [
    'experiment', 'perturb_key', 'method', 'cosine_final_mean',
    'method_final_mean', 'delta_pp', 'n_cosine', 'n_method',
]
paired = pairs.sort_values('delta_pp', ascending=False)[paired_cols].round(3)
lines.append(markdown_table(paired))
lines.append('')
shock = per_run[(per_run.experiment == 'shock') & (per_run.perturb_key == 'shock_lr=0.10')]
pivot = shock.pivot_table(index='seed', columns='method', values='final_student_acc', aggfunc='first')
pivot['patchtst_minus_cosine'] = pivot['patchtst'] - pivot['cosine']
lines.append('## Winning shock01 seeds')
winning = pivot[['cosine','patchtst','patchtst_minus_cosine']].reset_index().round(3)
lines.append(markdown_table(winning))
out.write_text('\n'.join(lines), encoding='utf-8')
print(f'wrote {out}')
