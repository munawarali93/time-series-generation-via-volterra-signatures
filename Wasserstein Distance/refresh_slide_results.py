"""Refresh LaTeX tables from the measured JSON results, without rerunning data."""
import json
from pathlib import Path


def refresh(root=None):
    root = Path(root) if root else Path(__file__).resolve().parent
    data = json.loads((root / 'results' / 'metrics.json').read_text())
    lines = [r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Change & Exact $W_1$ & Sample $W_1$ & Exact $W_2$ & Sample $W_2$\\', r'\midrule']
    for row in data['gaussians']:
        lines.append(row['case'] + ' & ' + ' & '.join(f"{row[k]:.3f}" for k in
                     ['w1', 'empirical_w1', 'w2', 'empirical_w2']) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (root / 'results' / 'gaussian_table.tex').write_text('\n'.join(lines)+'\n')
    lines = [r'\begin{tabular}{rrrr}', r'\toprule',
             r'$\epsilon$ & Transport cost & Full objective & $S_\epsilon$\\', r'\midrule']
    for row in data['sinkhorn']['runs']:
        lines.append(' & '.join(f"{row[k]:.3f}" for k in
                     ['epsilon', 'transport_cost', 'objective', 'divergence']) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (root / 'results' / 'sinkhorn_table.tex').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    refresh()
