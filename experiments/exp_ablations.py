"""
Ablation Studies for Boosted Attention

1. Number of boosting rounds (1, 2, 3, 4, 5)
2. Gate type (MLP, scalar, none)
3. Shared vs separate projections
4. Contribution analysis: which round helps most?
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time

from boosted_attention import BoostedAttention, StandardAttention, BoostedAttentionOutput

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cpu"
if torch.backends.mps.is_available():
    try:
        _ = torch.randn(2, 2, device="mps") @ torch.randn(2, 2, device="mps")
        DEVICE = "mps"
    except Exception:
        pass
print(f"Device: {DEVICE}")


def train_and_eval(model, patterns, d, K, noise_std, epochs=150, lr=3e-3,
                   batch_size=512, n_val=5000, device=DEVICE):
    """Train a model on denoising and return best val accuracy + details."""
    model = model.to(device)
    patterns = patterns.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_acc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        tidx = torch.randint(K, (batch_size,))
        targets = patterns[tidx]
        queries = targets + noise_std * torch.randn(batch_size, d, device=device)

        output, weights, entropy = model(queries, patterns)
        cos_loss = 1 - F.cosine_similarity(output, targets).mean()
        cls_loss = F.cross_entropy(weights, tidx.to(device))
        loss = cos_loss + 0.5 * cls_loss

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if (epoch + 1) % 50 == 0:
            acc = eval_acc(model, patterns, d, K, noise_std, n_val, device)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    # Final detailed eval
    final_acc = eval_acc(model, patterns, d, K, noise_std, n_val, device)
    details = eval_detailed(model, patterns, d, K, noise_std, n_val, device)
    return max(best_acc, final_acc), details


@torch.no_grad()
def eval_acc(model, patterns, d, K, noise_std, n=5000, device=DEVICE):
    model.eval()
    tidx = torch.randint(K, (n,))
    targets = patterns[tidx]
    queries = targets + noise_std * torch.randn(n, d, device=device)
    output, _, _ = model(queries, patterns)
    dists = torch.cdist(output.unsqueeze(0), patterns.unsqueeze(0)).squeeze(0)
    return (dists.argmin(dim=-1) == tidx.to(device)).float().mean().item() * 100


@torch.no_grad()
def eval_detailed(model, patterns, d, K, noise_std, n=5000, device=DEVICE):
    """Detailed eval including per-round analysis for boosted models."""
    model.eval()
    tidx = torch.randint(K, (n,))
    targets = patterns[tidx]
    queries = targets + noise_std * torch.randn(n, d, device=device)

    result = model(queries, patterns, return_details=True)
    details = {}

    if isinstance(result, BoostedAttentionOutput):
        # Per-round accuracy (cumulative)
        cumulative = torch.zeros_like(result.round_outputs[0])
        for r, ro in enumerate(result.round_outputs):
            if r == 0:
                cumulative = ro
            else:
                gate = result.gates[r - 1]
                cumulative = cumulative + gate * ro
            dists = torch.cdist(cumulative.unsqueeze(0), patterns.unsqueeze(0)).squeeze(0)
            acc = (dists.argmin(dim=-1) == tidx.to(device)).float().mean().item() * 100
            details[f"acc_after_round_{r}"] = acc

        # Gate statistics
        for r, g in enumerate(result.gates):
            details[f"gate_{r+1}_mean"] = g.mean().item()
            details[f"gate_{r+1}_std"] = g.std().item()

        # Entropy
        details["entropy_round_0"] = result.entropies[0].mean().item()

        # Residual norms (how much error remains after each round)
        cumulative = result.round_outputs[0]
        for r, res in enumerate(result.residuals):
            details[f"residual_{r+1}_norm"] = res.norm(dim=-1).mean().item()

        details["output"] = result.output
    else:
        output, weights, entropy = result
        dists = torch.cdist(output.unsqueeze(0), patterns.unsqueeze(0)).squeeze(0)
        details["acc_after_round_0"] = (dists.argmin(dim=-1) == tidx.to(device)).float().mean().item() * 100
        details["entropy_round_0"] = entropy.mean().item()
        details["output"] = output

    # Final accuracy
    out = details["output"]
    dists = torch.cdist(out.unsqueeze(0), patterns.unsqueeze(0)).squeeze(0)
    details["final_acc"] = (dists.argmin(dim=-1) == tidx.to(device)).float().mean().item() * 100
    details["cosine"] = F.cosine_similarity(out, targets).mean().item()
    del details["output"]

    return details


# ============================================================
# Ablation 1: Number of boosting rounds
# ============================================================

def ablation_rounds(d=64, K=16, noise_std=0.5):
    """Test 1, 2, 3, 4, 5 boosting rounds."""
    print(f"\n{'='*60}")
    print(f"  ABLATION 1: Number of Boosting Rounds")
    print(f"  d={d}, K={K}, σ={noise_std}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    patterns = F.normalize(torch.randn(K, d), dim=-1)

    results = {}
    for n_rounds in [1, 2, 3, 4, 5]:
        label = f"{n_rounds} round{'s' if n_rounds > 1 else ''}"
        print(f"\n  Training: {label}...")
        if n_rounds == 1:
            model = StandardAttention(d, beta_init=2.0)
        else:
            model = BoostedAttention(d, n_rounds=n_rounds, beta_init=2.0,
                                     gate_type="mlp")
        acc, details = train_and_eval(model, patterns, d, K, noise_std)
        results[n_rounds] = {"acc": acc, "details": details}
        print(f"  -> {label}: {acc:.1f}%")

        # Print per-round cumulative accuracy
        for key, val in sorted(details.items()):
            if key.startswith("acc_after"):
                print(f"     {key}: {val:.1f}%")
            elif key.startswith("gate"):
                print(f"     {key}: {val:.3f}")
            elif key.startswith("residual"):
                print(f"     {key}: {val:.4f}")

    return results


# ============================================================
# Ablation 2: Gate type
# ============================================================

def ablation_gate(d=64, K=16, noise_std=0.5):
    """Test different gate types: MLP, scalar (shrinkage), none (pure additive)."""
    print(f"\n{'='*60}")
    print(f"  ABLATION 2: Gate Type")
    print(f"  d={d}, K={K}, σ={noise_std}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    patterns = F.normalize(torch.randn(K, d), dim=-1)

    results = {}
    for gate_type in ["mlp", "scalar", "none"]:
        print(f"\n  Training: gate={gate_type}...")
        model = BoostedAttention(d, n_rounds=2, beta_init=2.0, gate_type=gate_type)
        acc, details = train_and_eval(model, patterns, d, K, noise_std)
        results[gate_type] = {"acc": acc, "details": details}
        print(f"  -> gate={gate_type}: {acc:.1f}%")

    # Also test baseline
    print(f"\n  Training: baseline (1 round, no correction)...")
    model = StandardAttention(d, beta_init=2.0)
    acc, details = train_and_eval(model, patterns, d, K, noise_std)
    results["baseline"] = {"acc": acc, "details": details}
    print(f"  -> baseline: {acc:.1f}%")

    return results


# ============================================================
# Ablation 3: Across multiple (d, K, noise) configs
# ============================================================

def ablation_configs():
    """Test boosted attention (2 rounds, MLP gate) across configs."""
    print(f"\n{'='*60}")
    print(f"  ABLATION 3: Across Configurations")
    print(f"{'='*60}")

    configs = [
        (32, 8, 0.3), (32, 8, 0.5), (32, 8, 0.8), (32, 8, 1.2),
        (64, 16, 0.3), (64, 16, 0.5), (64, 16, 0.8), (64, 16, 1.2),
        (128, 32, 0.3), (128, 32, 0.5), (128, 32, 0.8),
    ]

    results = {}
    for d, K, noise_std in configs:
        label = f"d={d}, K={K}, σ={noise_std}"
        print(f"\n  Config: {label}")

        torch.manual_seed(42)
        patterns = F.normalize(torch.randn(K, d), dim=-1)

        # Baseline
        model_base = StandardAttention(d, beta_init=2.0)
        acc_base, det_base = train_and_eval(model_base, patterns, d, K, noise_std)

        # Boosted (2 rounds)
        model_boost = BoostedAttention(d, n_rounds=2, beta_init=2.0, gate_type="mlp")
        acc_boost, det_boost = train_and_eval(model_boost, patterns, d, K, noise_std)

        delta = acc_boost - acc_base
        results[label] = {
            "baseline": acc_base, "boosted": acc_boost, "delta": delta,
            "d": d, "K": K, "noise_std": noise_std,
        }
        print(f"    Baseline: {acc_base:.1f}%  Boosted: {acc_boost:.1f}%  Delta: {delta:+.1f}%")

    return results


# ============================================================
# Plotting
# ============================================================

def plot_ablations(rounds_results, gate_results, config_results, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Number of rounds
    ax = axes[0, 0]
    n_rounds_list = sorted(rounds_results.keys())
    accs = [rounds_results[n]["acc"] for n in n_rounds_list]
    ax.plot(n_rounds_list, accs, 'o-', color='#2ecc71', linewidth=2.5, markersize=10)
    ax.axhline(rounds_results[1]["acc"], color='#e74c3c', linestyle='--',
               label=f'Baseline (1 round): {rounds_results[1]["acc"]:.1f}%')
    ax.set_xlabel('Number of Boosting Rounds', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy vs Number of Boosting Rounds', fontsize=14, fontweight='bold')
    ax.set_xticks(n_rounds_list)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2: Per-round cumulative accuracy (for 5-round model)
    ax = axes[0, 1]
    if 5 in rounds_results:
        det = rounds_results[5]["details"]
        round_accs = [det.get(f"acc_after_round_{r}", 0) for r in range(5)]
        ax.bar(range(5), round_accs, color=['#e74c3c'] + ['#2ecc71'] * 4, alpha=0.8)
        ax.set_xlabel('After Round #', fontsize=12)
        ax.set_ylabel('Cumulative Accuracy (%)', fontsize=12)
        ax.set_title('Cumulative Accuracy After Each Round\n(5-round model)',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(range(5))
        ax.set_xticklabels(['0\n(base)', '1\n(+corr)', '2\n(+corr)', '3\n(+corr)', '4\n(+corr)'])
        for i, v in enumerate(round_accs):
            ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')

    # Panel 3: Gate type comparison
    ax = axes[1, 0]
    gate_types = ["baseline", "none", "scalar", "mlp"]
    gate_labels = ["No correction\n(baseline)", "No gate\n(pure add)", "Scalar gate\n(shrinkage)", "MLP gate\n(learned)"]
    gate_accs = [gate_results[gt]["acc"] for gt in gate_types]
    colors_gate = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
    bars = ax.bar(range(len(gate_types)), gate_accs, color=colors_gate, alpha=0.85)
    ax.set_xticks(range(len(gate_types)))
    ax.set_xticklabels(gate_labels, fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Gate Type Ablation', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, gate_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Panel 4: Improvement across configs
    ax = axes[1, 1]
    cfg_labels = list(config_results.keys())
    deltas = [config_results[c]["delta"] for c in cfg_labels]
    noise_vals = [config_results[c]["noise_std"] for c in cfg_labels]
    colors_cfg = plt.cm.YlOrRd(np.array(noise_vals) / max(noise_vals))
    bars = ax.barh(range(len(cfg_labels)), deltas, color=colors_cfg, alpha=0.85)
    ax.set_yticks(range(len(cfg_labels)))
    ax.set_yticklabels(cfg_labels, fontsize=9)
    ax.set_xlabel('Accuracy Improvement (% points)', fontsize=12)
    ax.set_title('Boosted Attention Improvement Across Configs',
                 fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    for i, (bar, val) in enumerate(zip(bars, deltas)):
        ax.text(max(val, 0) + 0.3, i, f'{val:+.1f}%', va='center',
                fontsize=9, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    save_path = save_path or RESULTS_DIR / "exp13_ablations.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    t0 = time.time()

    rounds_results = ablation_rounds(d=64, K=16, noise_std=0.5)
    gate_results = ablation_gate(d=64, K=16, noise_std=0.5)
    config_results = ablation_configs()

    plot_ablations(rounds_results, gate_results, config_results)

    print(f"\nTotal time: {time.time() - t0:.1f}s ({(time.time() - t0)/60:.1f}min)")
