"""
Experiment: Learned Routing Gate

The v2 results show:
- Oracle routing (knowing ground truth) can improve accuracy by 4-10%
- But scalar divergence ||one_step - converged|| can't capture this

This experiment asks: can a LEARNED gate, given access to both output
vectors, learn to route better than scalar divergence?

Setup:
1. Train a standard attention model (single-step)
2. Generate (out_1step, out_converged, ground_truth_which_is_better) tuples
3. Train a small MLP gate to predict which output is better
4. Test the gate on held-out data
5. Also test: what features does the gate learn to use?
   - Just divergence magnitude?
   - Direction of divergence?
   - Similarity to specific patterns?
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

RESULTS_DIR = Path(__file__).parent.parent / "results"


class AttentionDenoiser(nn.Module):
    """Standard single-step attention for denoising."""
    def __init__(self, d, beta_init=2.0):
        super().__init__()
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.log_beta = nn.Parameter(torch.tensor(float(beta_init)).log())

    @property
    def beta(self):
        return self.log_beta.exp()

    def forward(self, query, patterns):
        q = self.W_q(query)
        k = self.W_k(patterns)
        v = self.W_v(patterns)
        logits = self.beta * (q @ k.T)
        weights = F.softmax(logits, dim=-1)
        return weights @ v, weights

    @torch.no_grad()
    def iterate(self, query, patterns, n_steps=30, tol=1e-7):
        out, w = self.forward(query, patterns)
        q = out
        for step in range(n_steps - 1):
            # Re-project and attend
            q_proj = self.W_q(q)
            k = self.W_k(patterns)
            v = self.W_v(patterns)
            logits = self.beta * (q_proj @ k.T)
            w = F.softmax(logits, dim=-1)
            q_new = w @ v
            if (q_new - q).norm(dim=-1).max() < tol:
                break
            q = q_new
        return q, w


class RoutingGate(nn.Module):
    """Learns to predict which output (1-step or converged) is better."""
    def __init__(self, d, hidden=64, feature_set="full"):
        super().__init__()
        self.feature_set = feature_set
        if feature_set == "full":
            # Full: both outputs + divergence vector + scalar features
            in_dim = 2 * d + d + 3  # out1, out2, diff, [|diff|, entropy, cos_sim]
        elif feature_set == "outputs_only":
            in_dim = 2 * d
        elif feature_set == "div_scalar_only":
            in_dim = 1
        elif feature_set == "div_vector":
            in_dim = d + 1  # diff vector + magnitude
        elif feature_set == "entropy_only":
            in_dim = 1

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, out_1step, out_conv, entropy=None):
        diff = out_conv - out_1step
        div_scalar = diff.norm(dim=-1, keepdim=True)
        cos_sim = F.cosine_similarity(out_1step, out_conv, dim=-1).unsqueeze(-1)
        if entropy is None:
            entropy = torch.zeros_like(div_scalar)

        if self.feature_set == "full":
            x = torch.cat([out_1step, out_conv, diff, div_scalar, entropy, cos_sim], dim=-1)
        elif self.feature_set == "outputs_only":
            x = torch.cat([out_1step, out_conv], dim=-1)
        elif self.feature_set == "div_scalar_only":
            x = div_scalar
        elif self.feature_set == "div_vector":
            x = torch.cat([diff, div_scalar], dim=-1)
        elif self.feature_set == "entropy_only":
            x = entropy

        return self.net(x).squeeze(-1)  # (B,) logit: positive = trust converged


def run_routing_experiment(d=32, K=8, noise_std=0.8, device='cpu'):
    """Full routing experiment for one (d, K, noise) config."""

    print(f"\n  [Step 1] Training attention denoiser (d={d}, K={K}, σ={noise_std})...")

    torch.manual_seed(42)
    patterns = F.normalize(torch.randn(K, d), dim=-1).to(device)

    # Train denoiser
    model = AttentionDenoiser(d, beta_init=2.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    for epoch in range(100):
        model.train()
        target_idx = torch.randint(K, (512,))
        targets = patterns[target_idx]
        queries = targets + noise_std * torch.randn(512, d, device=device)

        output, weights = model(queries, patterns)
        cos_loss = 1 - F.cosine_similarity(output, targets).mean()
        cls_loss = F.cross_entropy(weights, target_idx.to(device))
        loss = cos_loss + 0.5 * cls_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    # Quick accuracy check
    with torch.no_grad():
        tidx = torch.randint(K, (2000,))
        tgt = patterns[tidx]
        qry = tgt + noise_std * torch.randn(2000, d, device=device)
        out, _ = model(qry, patterns)
        dists = torch.cdist(out.unsqueeze(0), patterns.unsqueeze(0)).squeeze(0)
        base_acc = (dists.argmin(dim=-1) == tidx.to(device)).float().mean().item() * 100
    print(f"  Base 1-step accuracy: {base_acc:.1f}%")

    # [Step 2] Generate routing dataset
    print(f"  [Step 2] Generating routing data...")
    n_data = 20000
    with torch.no_grad():
        target_idx = torch.randint(K, (n_data,))
        targets = patterns[target_idx]
        queries = targets + noise_std * torch.randn(n_data, d, device=device)

        out_1step, weights_1step = model(queries, patterns)
        out_conv, _ = model.iterate(queries, patterns, n_steps=30)

        # Which is better? (binary label: 1 = converged is closer to target)
        mse_1step = (out_1step - targets).pow(2).sum(dim=-1)
        mse_conv = (out_conv - targets).pow(2).sum(dim=-1)
        conv_is_better = (mse_conv < mse_1step).float()

        # Entropy
        entropy = -(weights_1step * (weights_1step + 1e-10).log()).sum(dim=-1, keepdim=True)

        # Correctness
        correct_1step = (torch.cdist(out_1step.unsqueeze(0), patterns.unsqueeze(0)).squeeze(0).argmin(dim=-1) == target_idx.to(device))
        correct_conv = (torch.cdist(out_conv.unsqueeze(0), patterns.unsqueeze(0)).squeeze(0).argmin(dim=-1) == target_idx.to(device))

        acc_1step = correct_1step.float().mean().item() * 100
        acc_conv = correct_conv.float().mean().item() * 100
        acc_oracle = (correct_1step | correct_conv).float().mean().item() * 100

    print(f"  1-step acc: {acc_1step:.1f}%, conv acc: {acc_conv:.1f}%, oracle: {acc_oracle:.1f}%")
    print(f"  Conv better (by MSE): {conv_is_better.mean().item()*100:.1f}% of queries")

    # [Step 3] Train routing gates with different feature sets
    print(f"  [Step 3] Training routing gates...")

    # Split
    n_train = 15000
    perm = torch.randperm(n_data)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    feature_sets = ["full", "outputs_only", "div_vector", "div_scalar_only", "entropy_only"]
    gate_results = {}

    for fs in feature_sets:
        gate = RoutingGate(d, hidden=64, feature_set=fs).to(device)
        gate_opt = torch.optim.Adam(gate.parameters(), lr=1e-3)

        # Train the gate
        for gate_epoch in range(200):
            idx = train_idx[torch.randperm(n_train)[:512]]
            logits = gate(out_1step[idx], out_conv[idx], entropy[idx])
            gate_loss = F.binary_cross_entropy_with_logits(logits, conv_is_better[idx])
            gate_opt.zero_grad()
            gate_loss.backward()
            gate_opt.step()

        # Evaluate
        gate.eval()
        with torch.no_grad():
            test_logits = gate(out_1step[test_idx], out_conv[test_idx], entropy[test_idx])
            test_pred = (test_logits > 0).float()

            # Routing accuracy: does the gate correctly predict which output is better?
            routing_acc = (test_pred == conv_is_better[test_idx]).float().mean().item() * 100

            # Retrieval accuracy with routing
            use_conv = test_pred.bool()
            routed_out = torch.where(use_conv.unsqueeze(-1), out_conv[test_idx], out_1step[test_idx])
            routed_dists = torch.cdist(routed_out.unsqueeze(0), patterns.unsqueeze(0)).squeeze(0)
            routed_retrieved = routed_dists.argmin(dim=-1)
            routed_acc = (routed_retrieved == target_idx[test_idx].to(device)).float().mean().item() * 100

            # Baselines on test set
            test_acc_1step = correct_1step[test_idx].float().mean().item() * 100
            test_acc_conv = correct_conv[test_idx].float().mean().item() * 100
            test_acc_oracle = (correct_1step[test_idx] | correct_conv[test_idx]).float().mean().item() * 100

        gate_results[fs] = {
            "routing_acc": routing_acc,
            "retrieval_acc": routed_acc,
        }
        improvement = routed_acc - test_acc_1step
        print(f"    {fs:20s}: routing_acc={routing_acc:.1f}%  "
              f"retrieval={routed_acc:.1f}%  (1-step={test_acc_1step:.1f}%, "
              f"conv={test_acc_conv:.1f}%, oracle={test_acc_oracle:.1f}%, "
              f"delta={improvement:+.1f}%)")

    return {
        "d": d, "K": K, "noise_std": noise_std,
        "acc_1step": acc_1step, "acc_conv": acc_conv, "acc_oracle": acc_oracle,
        "gates": gate_results,
    }


def plot_routing_results(all_results, save_path=None):
    """Plot routing experiment results."""
    configs = list(all_results.keys())
    feature_sets = list(all_results[configs[0]]["gates"].keys())

    fig, axes = plt.subplots(1, len(configs), figsize=(7*len(configs), 6))
    if len(configs) == 1:
        axes = [axes]

    for ax, cfg in zip(axes, configs):
        r = all_results[cfg]
        x = np.arange(len(feature_sets))

        # Baselines
        ax.axhline(r["acc_1step"], color='red', linestyle='--', linewidth=1.5,
                    label=f'One-step ({r["acc_1step"]:.1f}%)')
        ax.axhline(r["acc_conv"], color='blue', linestyle='--', linewidth=1.5,
                    label=f'Converged ({r["acc_conv"]:.1f}%)')
        ax.axhline(r["acc_oracle"], color='green', linestyle='--', linewidth=1.5,
                    label=f'Oracle ({r["acc_oracle"]:.1f}%)')

        # Gate results
        accs = [r["gates"][fs]["retrieval_acc"] for fs in feature_sets]
        bars = ax.bar(x, accs, color=['#9b59b6', '#e67e22', '#3498db', '#e74c3c', '#f39c12'],
                      alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([fs.replace('_', '\n') for fs in feature_sets],
                           fontsize=9)
        ax.set_ylabel('Retrieval Accuracy (%)', fontsize=11)
        ax.set_title(f'd={r["d"]}, K={r["K"]}, σ={r["noise_std"]}',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=8)

        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    save_path = save_path or RESULTS_DIR / "exp10_learned_routing.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    t0 = time.time()

    device = "cpu"
    if torch.backends.mps.is_available():
        try:
            _ = torch.randn(2, 2, device="mps") @ torch.randn(2, 2, device="mps")
            device = "mps"
        except Exception:
            pass
    print(f"Using device: {device}")

    configs = [
        (16, 4, 0.8),
        (32, 8, 0.8),
        (64, 16, 0.8),
    ]

    all_results = {}
    for d, K, noise_std in configs:
        key = f"d{d}_K{K}_s{noise_std}"
        all_results[key] = run_routing_experiment(d, K, noise_std, device=device)

    plot_routing_results(all_results)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
