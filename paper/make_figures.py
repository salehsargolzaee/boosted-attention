"""
Generate publication-quality figures for the gradient-boosted attention paper.

Figures:
  1. Architecture diagram (standard vs boosted attention)
  2. Results (perplexity bars + ablation rounds)
  3. Gate analysis (raincloud plot from saved JSON)
  4. Convex hull escape (raincloud plot from saved JSON)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / 'results'
PAPER_DIR = Path(__file__).parent

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

C_STD = '#c0392b'
C_BOOST = '#27ae60'
C_FAIR = '#e67e22'

def layer_color(i, n=4):
    cmap = plt.cm.Blues
    return cmap(0.35 + 0.55 * i / (n - 1))


def _save(name):
    for ext in ['pdf', 'png']:
        plt.savefig(PAPER_DIR / f'{name}.{ext}')
    plt.close()
    print(f'Saved {name}')


# ============================================================
# Figure 1: Architecture — clean vertical flowcharts
# ============================================================

def fig_architecture():
    fig = plt.figure(figsize=(10, 4.2))

    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.2], wspace=0.15)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    def box(ax, cx, cy, w, h, text, fc='white', ec='#2c3e50', fs=9, fw='normal'):
        r = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle="round,pad=0.06", fc=fc, ec=ec, lw=1.2)
        ax.add_patch(r)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=fs, fontweight=fw)

    def arrow(ax, x1, y1, x2, y2, color='#7f8c8d', lw=1.3):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw))

    def arrow_L(ax, x1, y1, x2, y2, color='#7f8c8d', lw=1.3, vh=True):
        if vh:
            angleA, angleB = 90, 0
        else:
            angleA, angleB = 0, 90
        patch = FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=f'angle,angleA={angleA},angleB={angleB},rad=0',
            arrowstyle='->', color=color, lw=lw, mutation_scale=12)
        ax.add_patch(patch)

    ylo, yhi = -0.2, 6.8

    # ---- (a) Standard Attention ----
    ax = ax1
    ax.set_xlim(0, 4)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect('auto')
    ax.axis('off')
    ax.text(2, yhi - 0.3, '(a) Standard Attention', fontsize=11,
            fontweight='bold', ha='center', va='top')

    box(ax, 2, 0.5, 2.4, 0.7, r'Input $\mathbf{x}$', '#ecf0f1', fs=10)
    arrow(ax, 2, 0.85, 2, 1.65)
    box(ax, 2, 2.1, 2.4, 0.7, r'$\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$', '#dfe6e9')
    arrow(ax, 2, 2.45, 2, 3.25)
    box(ax, 2, 3.7, 2.4, 0.7, 'Softmax Attention', '#d5f5e3')
    arrow(ax, 2, 4.05, 2, 4.85)
    box(ax, 2, 5.3, 2.4, 0.7, r'Output $\hat{\mathbf{y}}_0$', '#aed6f1', fs=10, fw='bold')

    # ---- (b) Boosted Attention (M=2) ----
    ax = ax2
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect('auto')
    ax.axis('off')
    ax.text(4, yhi - 0.3, '(b) Boosted Attention ($M{=}2$)', fontsize=11,
            fontweight='bold', ha='center', va='top')

    box(ax, 1.8, 0.5, 2, 0.55, r'Input $\mathbf{x}$', '#ecf0f1', fs=10)

    arrow(ax, 1.8, 0.78, 1.8, 1.35)
    box(ax, 1.8, 1.7, 2.2, 0.5, r'$\mathbf{W}_Q^{(0)}, \mathbf{W}_K^{(0)}, \mathbf{W}_V^{(0)}$', '#dfe6e9', fs=8)
    arrow(ax, 1.8, 1.95, 1.8, 2.4)
    box(ax, 1.8, 2.7, 2.2, 0.5, r'Attn$_0$($\mathbf{x}$)', '#d5f5e3', fs=9)
    arrow(ax, 1.8, 2.95, 1.8, 3.45)
    box(ax, 1.8, 3.7, 1.5, 0.4, r'$\hat{\mathbf{y}}_0$', '#a9dfbf', fs=10, fw='bold')

    arrow_L(ax, 2.8, 0.5, 4.2, 1.45, color='#8e44ad', lw=1.5, vh=False)
    arrow_L(ax, 2.55, 3.7, 4.2, 1.95, color='#8e44ad', lw=1.5, vh=False)
    box(ax, 4.7, 1.7, 1.6, 0.5, r'$\mathbf{r}{=}\mathbf{x}{-}\hat{\mathbf{y}}_0$',
        '#e8daef', ec='#8e44ad', fs=8, fw='bold')

    arrow(ax, 5.5, 1.7, 5.85, 1.7, color='#8e44ad', lw=1.5)
    box(ax, 6.8, 1.7, 1.7, 0.5, r'$\mathbf{W}_Q^{(1)},\mathbf{W}_K^{(1)},\mathbf{W}_V^{(1)}$', '#fdebd0', fs=7)
    arrow(ax, 6.8, 1.95, 6.8, 2.4)
    box(ax, 6.8, 2.7, 2.0, 0.5, r'Attn$_1$($\mathbf{r}$)', '#fadbd8', fs=9)
    arrow(ax, 6.8, 2.95, 6.8, 3.45)
    box(ax, 6.8, 3.7, 1.2, 0.4, r'$\mathbf{c}_1$', '#f5b7b1', fs=10, fw='bold')

    arrow_L(ax, 1.8, 3.9, 2.75, 4.68, color='#2980b9', lw=1.5, vh=True)
    arrow_L(ax, 6.8, 3.9, 5.25, 4.68, color='#2980b9', lw=1.5, vh=True)
    box(ax, 4, 4.9, 2.5, 0.45, r'Gate $\mathbf{g}{=}\sigma(\mathbf{W}_g[\hat{\mathbf{y}}_0 \,\Vert\, \mathbf{c}_1])$',
        '#d6eaf8', ec='#2980b9', fs=8)

    arrow(ax, 4, 5.13, 4, 5.55)
    box(ax, 4, 5.8, 3, 0.45,
        r'Output: $\hat{\mathbf{y}}_0 + \mathbf{g} \odot \mathbf{c}_1$',
        '#aed6f1', fs=9, fw='bold')

    for ext in ['pdf', 'png']:
        plt.savefig(PAPER_DIR / f'fig_architecture.{ext}')
    plt.close()
    print('Saved fig_architecture')


# ============================================================
# Figure 2: Results
# ============================================================

def fig_results():
    with open(RESULTS_DIR / 'exp_v2_small.json') as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.5),
                                    gridspec_kw={'width_ratios': [1.3, 1]})

    C_TWIC = '#8e44ad'

    # ---- Left: perplexity bars, zoomed ----
    ax = ax1
    keys = ['small/Standard', 'small/Twicing', 'small/Std-fair(d=288)', 'small/Boosted-2']
    labels = ['Standard\n(7.4M)', 'Twicing\n(7.4M)',
              'Std-fair\n(8.8M)', 'Ours\n(8.7M)']
    means = [data[k]['test_mean'] for k in keys]
    stds = [data[k]['test_std'] for k in keys]
    colors = [C_STD, C_TWIC, C_FAIR, C_BOOST]

    bars = ax.bar(range(4), means, yerr=stds, color=colors, alpha=0.85,
                  capsize=5, edgecolor='white', lw=1.5, width=0.6)
    ax.set_ylim(66, 74)
    ax.set_ylabel('Test Perplexity (lower is better)')
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=8)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.15,
                f'{m:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    y_ann = means[2] + stds[2] + 0.6
    ax.plot([2, 3], [y_ann, y_ann], color=C_BOOST, lw=1.5)
    ax.plot([2, 2], [y_ann - 0.3, y_ann], color=C_BOOST, lw=1.5)
    ax.plot([3, 3], [means[3] + stds[3] + 0.2, y_ann], color=C_BOOST, lw=1.5)
    ax.text(2.5, y_ann + 0.15, '$-$1.1 ppl\n(architectural)',
            ha='center', va='bottom', fontsize=7.5, color=C_BOOST, fontweight='bold')

    ax.grid(axis='y', alpha=0.2, ls='--')
    ax.set_title('WikiText-103 Test Perplexity', fontsize=11, fontweight='bold')

    # ---- Right: ablation rounds (read from JSON) ----
    ax = ax2
    with open(RESULTS_DIR / 'exp_ablations.json') as f:
        abl = json.load(f)
    abl_rounds = abl['ablation_rounds']
    rounds = sorted(int(k) for k in abl_rounds if k != 'bayes_optimal')
    accs = [abl_rounds[str(r)]['acc'] for r in rounds]
    bayes_acc = abl_rounds['bayes_optimal']['acc']

    ax.plot(rounds, accs, 'o-', color=C_BOOST, lw=2.5, ms=8, zorder=3)
    ax.fill_between(rounds, 38, accs, alpha=0.08, color=C_BOOST)
    ax.axhline(accs[0], color=C_STD, ls='--', lw=1, alpha=0.6, label='1-round baseline')
    ax.axhline(bayes_acc, color='#8e44ad', ls=':', lw=1.5, alpha=0.8,
               label=f'Bayes optimal ({bayes_acc:.1f}%)')

    delta_max = max(accs) - accs[0]
    ax.annotate(f'+{delta_max:.1f} pp', xy=(1.55, (accs[0] + accs[1]) / 2),
                fontsize=9, fontweight='bold', color=C_BOOST)

    ax.set_xlabel('Boosting Rounds ($M$)')
    ax.set_ylabel('Retrieval Accuracy (%)')
    ax.set_xticks(rounds)
    ax.set_ylim(38, 62)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.2, ls='--')
    ax.set_title('Ablation: Rounds (pattern retrieval)', fontsize=11, fontweight='bold')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(PAPER_DIR / f'fig_results.{ext}')
    plt.close()
    print('Saved fig_results')


# ============================================================
# Figure 3: Gate Analysis — raincloud plot
# ============================================================

def _raincloud(ax, data_list, positions, color_func, show_mean=True,
               max_scatter=None):
    """Half-violin + jittered strip + box plot."""
    from scipy.stats import gaussian_kde
    for i, (pos, d) in enumerate(zip(positions, data_list)):
        c = color_func(i)
        kde = gaussian_kde(d, bw_method=0.3)
        y_grid = np.linspace(d.min() - 0.05 * d.ptp(), d.max() + 0.05 * d.ptp(), 200)
        density = kde(y_grid)
        density = density / density.max() * 0.32
        ax.fill_betweenx(y_grid, pos, pos + density, alpha=0.5, color=c,
                         linewidth=0, zorder=1)
        ax.plot(pos + density, y_grid, color=c, lw=0.5, zorder=2)

        rng = np.random.default_rng(42 + i)
        if max_scatter and len(d) > max_scatter:
            d_scatter = d[rng.choice(len(d), size=max_scatter, replace=False)]
        else:
            d_scatter = d
        jitter = rng.uniform(-0.18, -0.02, size=len(d_scatter))
        ax.scatter(pos + jitter, d_scatter, s=2.5, alpha=0.45, color=c,
                   edgecolors='none', zorder=2)

        q25, med, q75 = np.percentile(d, [25, 50, 75])
        ax.plot([pos - 0.02, pos - 0.02], [q25, q75], color='#2c3e50',
                lw=1.2, solid_capstyle='round', zorder=4)
        ax.scatter(pos - 0.02, med, color='white', s=8, zorder=5,
                   edgecolors='#2c3e50', linewidths=0.5)

        if show_mean:
            mu = d.mean()
            ax.text(pos + 0.38, mu, f'{mu:.2f}', ha='left', va='center',
                    fontsize=7, color='#2c3e50')


def fig_gate_analysis():
    with open(RESULTS_DIR / 'analysis_gate_values.json') as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    n_layers = len(data)
    positions = list(range(n_layers))
    gate_data = [np.array(data[f'layer_{i}']) for i in positions]

    _raincloud(ax, gate_data, positions, layer_color)

    ax.axhline(0.5, color='#aaaaaa', ls=':', lw=0.5)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel('Layer', fontsize=9)
    ax.set_ylabel('Gate value (per dimension)', fontsize=9)
    ax.set_xticks(positions)
    ax.set_xlim(-0.35, n_layers - 0.4)
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    _save('fig_gate_analysis')


# ============================================================
# Figure 4: Convex Hull Escape — raincloud plot
# ============================================================

def fig_convex_hull():
    with open(RESULTS_DIR / 'analysis_convex_hull.json') as f:
        raw = json.load(f)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    layers = sorted([k for k in raw if k.startswith('layer_')],
                    key=lambda k: int(k.split('_')[1]))
    n_layers = len(layers)
    positions = list(range(n_layers))

    def get_dists(entry):
        if isinstance(entry, dict):
            return np.array(entry['distances'])
        return np.array(entry)

    dist_data = [get_dists(raw[k]) for k in layers]

    _raincloud(ax, dist_data, positions, layer_color, max_scatter=120)

    ax.axhline(0, color='#aaaaaa', ls='-', lw=0.5)
    ax.set_xlabel('Layer', fontsize=9)
    ax.set_ylabel('$\\ell_2$ distance to conv($\\mathbf{V}^{(0)}$)', fontsize=9)
    ax.set_xticks(positions)
    ax.set_xlim(-0.35, n_layers - 0.4)
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    _save('fig_convex_hull')


# ============================================================
# Figure 5: Convergence speed
# ============================================================

C_TWICE = '#8e44ad'

def fig_convergence():
    import torch

    models = [
        ('Standard', C_STD, '--'),
        ('Boosted-2', C_BOOST, '-'),
        ('Twicing', C_TWICE, '-.'),
        ('Std-fair(d=288)', C_FAIR, ':'),
    ]
    labels = ['Standard', 'Boosted ($M{=}2$)', 'Twicing', 'Std-fair ($d{=}288$)']

    fig, ax = plt.subplots(figsize=(5, 3.2))

    for (name, color, ls), label in zip(models, labels):
        curves = []
        for seed in [42, 123]:
            path = RESULTS_DIR / f'checkpoints/small_{name}_seed{seed}.pt'
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            curves.append(ckpt['history']['val_ppl'])
        mean = np.mean(curves, axis=0)
        epochs = np.arange(1, len(mean) + 1)
        ax.plot(epochs, mean, ls, color=color, lw=2, label=label, marker='o', ms=4)

    fair_final = 69.6
    ax.axhline(fair_final, color=C_FAIR, ls='--', lw=0.8, alpha=0.5)
    ax.annotate('Std-fair final (69.6)', xy=(3, fair_final - 0.6),
                fontsize=7, color=C_FAIR, alpha=0.8)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Perplexity')
    ax.set_xticks(range(1, 16))
    ax.set_ylim(65, 120)
    ax.legend(fontsize=7.5, loc='upper right')
    ax.grid(True, alpha=0.2, ls='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    _save('fig_convergence')


# ============================================================
# Figure 6: Scaling ablation across (d, K, sigma) configs
# ============================================================

def fig_scaling_ablation():
    with open(RESULTS_DIR / 'exp_ablations.json') as f:
        data = json.load(f)

    configs = data['ablation_configs']

    dims = [32, 64, 128]
    dim_configs = {d: {} for d in dims}
    for key, val in configs.items():
        for d in dims:
            if key.startswith(f'd={d},'):
                sigma = float(key.split('s=')[1])
                dim_configs[d][sigma] = val

    C_BAYES = '#8e44ad'
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, d in zip(axes, dims):
        sigmas = sorted(dim_configs[d].keys())
        baselines = [dim_configs[d][s]['baseline'] for s in sigmas]
        boosteds = [dim_configs[d][s]['boosted'] for s in sigmas]
        bayes_opts = [dim_configs[d][s].get('bayes_optimal', 0) for s in sigmas]
        deltas = [dim_configs[d][s]['delta'] for s in sigmas]

        for key in configs:
            if key.startswith(f'd={d},'):
                K_val = int(key.split('K=')[1].split(',')[0])
                break

        x = np.arange(len(sigmas))
        w = 0.25
        ax.bar(x - w, baselines, w, label='Standard', color=C_STD, alpha=0.8)
        ax.bar(x, boosteds, w, label='Boosted', color=C_BOOST, alpha=0.8)
        ax.bar(x + w, bayes_opts, w, label='Bayes opt.', color=C_BAYES, alpha=0.5)

        for i, delta in enumerate(deltas):
            y_top = max(baselines[i], boosteds[i], bayes_opts[i])
            ax.text(x[i], y_top + 1.0, f'+{delta:.1f}',
                    ha='center', va='bottom', fontsize=7, fontweight='bold',
                    color=C_BOOST)

        ax.set_xticks(x)
        ax.set_xticklabels([f'$\\sigma$={s}' for s in sigmas], fontsize=8)
        ax.set_title(f'$d$={d}, $K$={K_val}', fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.2, ls='--')
        if ax == axes[0]:
            ax.set_ylabel('Retrieval Accuracy (%)', fontsize=9)
            ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    _save('fig_scaling_ablation')


# ============================================================

if __name__ == '__main__':
    fig_architecture()
    fig_results()
    if (RESULTS_DIR / 'analysis_gate_values.json').exists():
        fig_gate_analysis()
    if (RESULTS_DIR / 'analysis_convex_hull.json').exists():
        fig_convex_hull()
    if (RESULTS_DIR / 'exp_ablations.json').exists():
        fig_scaling_ablation()
    if (RESULTS_DIR / 'checkpoints').exists():
        fig_convergence()
    print('\nAll figures generated.')
