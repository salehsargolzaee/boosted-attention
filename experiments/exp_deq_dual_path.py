"""
Experiment: DEQ-based Dual-Path Attention

The key insight: previous experiments failed because the converged path
was trained by backpropagating through 20 iteration steps (vanishing gradients),
or not trained at all (post-hoc iteration of a one-step model).

DEQ (Deep Equilibrium Models) solve this: they find the fixed point in the
forward pass using an iterative solver, then compute the gradient at the
fixed point using the implicit function theorem. This means:
- Forward: iterate attention to convergence (proper fixed point)
- Backward: compute dz*/dtheta = (I - df/dz)^{-1} df/dtheta via a linear solve
- Result: proper gradients for the converged path, O(1) memory

Architecture:
1. Shared W_q, W_k, W_v, beta
2. Shallow path: one attention step (standard)
3. Deep path: DEQ fixed point of attention iteration
4. Both paths supervised: combined loss on both outputs
5. Learned gate: merge based on [shallow, deep, divergence]
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
from torchdeq import get_deq
import time
import json

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cpu"
# MPS has issues with some torchdeq ops, test carefully
if torch.backends.mps.is_available():
    try:
        t = torch.randn(4, 4, device="mps")
        _ = torch.linalg.solve(t, t)  # DEQ needs linear solves
        DEVICE = "mps"
    except Exception:
        DEVICE = "cpu"
print(f"Using device: {DEVICE}")


class DEQDualPathDenoiser(nn.Module):
    """
    Dual-path denoiser with DEQ-trained converged path.

    Both paths share projections and are jointly trained.
    The gate sees both outputs and learns when to trust each.
    """
    def __init__(self, d, d_hidden=None, beta_init=2.0, gate_hidden=32,
                 deq_max_iter=30, deq_tol=1e-5):
        super().__init__()
        d_hidden = d_hidden or d
        self.d = d
        self.d_hidden = d_hidden

        # Shared projections
        self.W_q = nn.Linear(d, d_hidden, bias=False)
        self.W_k = nn.Linear(d, d_hidden, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.log_beta = nn.Parameter(torch.tensor(float(beta_init)).log())

        # Learned gate: takes [shallow_out, deep_out, div_scalar] -> alpha
        self.gate = nn.Sequential(
            nn.Linear(2 * d + 1, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )

        # DEQ solver for the converged path
        self.deq = get_deq(
            f_solver='fixed_point_iter',
            f_max_iter=deq_max_iter,
            f_tol=deq_tol,
            b_solver='fixed_point_iter',
            b_max_iter=deq_max_iter,
            b_tol=deq_tol,
        )

    @property
    def beta(self):
        return self.log_beta.exp()

    def _attention_step(self, z, k_proj, v_proj):
        """Single attention step in projected space. z: (B, d)."""
        q_proj = self.W_q(z)  # (B, d_hidden)
        logits = self.beta * (q_proj @ k_proj.T)  # (B, K)
        weights = F.softmax(logits, dim=-1)
        return weights @ v_proj  # (B, d)

    def forward(self, query, patterns, return_details=False):
        """
        Args:
            query: (B, d) noisy queries
            patterns: (K, d) stored patterns
            return_details: whether to return per-path outputs
        """
        # Pre-compute key/value projections (shared, computed once)
        k_proj = self.W_k(patterns)  # (K, d_hidden)
        v_proj = self.W_v(patterns)  # (K, d)

        # === Shallow path: one attention step ===
        q_proj = self.W_q(query)  # (B, d_hidden)
        logits_shallow = self.beta * (q_proj @ k_proj.T)  # (B, K)
        weights_shallow = F.softmax(logits_shallow, dim=-1)
        out_shallow = weights_shallow @ v_proj  # (B, d)

        # === Deep path: DEQ fixed point ===
        # Define the attention iteration as a function for DEQ
        def attn_iter(z):
            qp = self.W_q(z)
            logits = self.beta * (qp @ k_proj.T)
            w = F.softmax(logits, dim=-1)
            return w @ v_proj

        # Use one-step output as initialization (warm start)
        z0 = out_shallow.detach().clone()

        # DEQ finds z* such that z* = attn_iter(z*)
        z_star_list, info = self.deq(attn_iter, z0)
        out_deep = z_star_list[0]  # (B, d)

        # === Divergence ===
        div = (out_shallow - out_deep).norm(dim=-1, keepdim=True)  # (B, 1)

        # === Learned gate ===
        gate_input = torch.cat([out_shallow, out_deep, div], dim=-1)
        alpha = torch.sigmoid(self.gate(gate_input))  # (B, 1)
        # alpha=1 -> trust shallow, alpha=0 -> trust deep
        output = alpha * out_shallow + (1 - alpha) * out_deep

        if return_details:
            return {
                "output": output,
                "out_shallow": out_shallow,
                "out_deep": out_deep,
                "divergence": div,
                "gate_alpha": alpha,
                "weights_shallow": weights_shallow,
                "info": info,
            }
        return output, out_shallow, out_deep


class OneStepBaseline(nn.Module):
    """Standard single-step attention baseline."""
    def __init__(self, d, d_hidden=None, beta_init=2.0):
        super().__init__()
        d_hidden = d_hidden or d
        self.W_q = nn.Linear(d, d_hidden, bias=False)
        self.W_k = nn.Linear(d, d_hidden, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.log_beta = nn.Parameter(torch.tensor(float(beta_init)).log())

    @property
    def beta(self):
        return self.log_beta.exp()

    def forward(self, query, patterns, return_details=False):
        q = self.W_q(query)
        k = self.W_k(patterns)
        v = self.W_v(patterns)
        logits = self.beta * (q @ k.T)
        weights = F.softmax(logits, dim=-1)
        out = weights @ v
        return out, weights


class DEQConvergedBaseline(nn.Module):
    """Converged-only baseline using DEQ."""
    def __init__(self, d, d_hidden=None, beta_init=2.0, deq_max_iter=30, deq_tol=1e-5):
        super().__init__()
        d_hidden = d_hidden or d
        self.W_q = nn.Linear(d, d_hidden, bias=False)
        self.W_k = nn.Linear(d, d_hidden, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.log_beta = nn.Parameter(torch.tensor(float(beta_init)).log())
        self.deq = get_deq(
            f_solver='fixed_point_iter', f_max_iter=deq_max_iter, f_tol=deq_tol,
            b_solver='fixed_point_iter', b_max_iter=deq_max_iter, b_tol=deq_tol,
        )

    @property
    def beta(self):
        return self.log_beta.exp()

    def forward(self, query, patterns, return_details=False):
        k = self.W_k(patterns)
        v = self.W_v(patterns)

        def attn_iter(z):
            qp = self.W_q(z)
            logits = self.beta * (qp @ k.T)
            w = F.softmax(logits, dim=-1)
            return w @ v

        # Initialize with one-step
        q0 = self.W_q(query)
        logits0 = self.beta * (q0 @ k.T)
        w0 = F.softmax(logits0, dim=-1)
        z0 = (w0 @ v).detach()

        z_star_list, info = self.deq(attn_iter, z0)
        return z_star_list[0], info


def train_and_evaluate(d, K, noise_std, epochs=120, lr=3e-3, batch_size=512,
                       n_train=20000, n_val=5000, device=DEVICE):
    """Train all three models and compare."""

    torch.manual_seed(42)
    patterns = F.normalize(torch.randn(K, d), dim=-1).to(device)

    results = {}

    # ====== Model 1: One-step baseline ======
    print(f"\n  [1/3] Training one-step baseline...")
    model_1step = OneStepBaseline(d, beta_init=2.0).to(device)
    opt1 = torch.optim.Adam(model_1step.parameters(), lr=lr)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=epochs)

    for epoch in range(epochs):
        model_1step.train()
        tidx = torch.randint(K, (batch_size,))
        targets = patterns[tidx]
        queries = targets + noise_std * torch.randn(batch_size, d, device=device)

        out, weights = model_1step(queries, patterns)
        cos_loss = 1 - F.cosine_similarity(out, targets).mean()
        cls_loss = F.cross_entropy(weights, tidx.to(device))
        loss = cos_loss + 0.5 * cls_loss

        opt1.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model_1step.parameters(), 1.0)
        opt1.step()
        sched1.step()

        if (epoch + 1) % 30 == 0:
            model_1step.eval()
            with torch.no_grad():
                tidx_v = torch.randint(K, (n_val,))
                tgt_v = patterns[tidx_v]
                qry_v = tgt_v + noise_std * torch.randn(n_val, d, device=device)
                out_v, _ = model_1step(qry_v, patterns)
                dists = torch.cdist(out_v.unsqueeze(0), patterns.unsqueeze(0)).squeeze(0)
                acc = (dists.argmin(dim=-1) == tidx_v.to(device)).float().mean().item() * 100
            print(f"    Epoch {epoch+1}: acc={acc:.1f}%, beta={model_1step.beta.item():.2f}")

    results["one_step"] = {"model": model_1step}

    # ====== Model 2: DEQ converged baseline ======
    print(f"\n  [2/3] Training DEQ converged baseline...")
    model_conv = DEQConvergedBaseline(d, beta_init=2.0).to(device)
    opt2 = torch.optim.Adam(model_conv.parameters(), lr=lr)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=epochs)

    for epoch in range(epochs):
        model_conv.train()
        tidx = torch.randint(K, (batch_size,))
        targets = patterns[tidx]
        queries = targets + noise_std * torch.randn(batch_size, d, device=device)

        out, info = model_conv(queries, patterns)
        loss = 1 - F.cosine_similarity(out, targets).mean()

        opt2.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model_conv.parameters(), 1.0)
        opt2.step()
        sched2.step()

        if (epoch + 1) % 30 == 0:
            model_conv.eval()
            with torch.no_grad():
                tidx_v = torch.randint(K, (n_val,))
                tgt_v = patterns[tidx_v]
                qry_v = tgt_v + noise_std * torch.randn(n_val, d, device=device)
                out_v, _ = model_conv(qry_v, patterns)
                dists = torch.cdist(out_v.unsqueeze(0), patterns.unsqueeze(0)).squeeze(0)
                acc = (dists.argmin(dim=-1) == tidx_v.to(device)).float().mean().item() * 100
            print(f"    Epoch {epoch+1}: acc={acc:.1f}%, beta={model_conv.beta.item():.2f}")

    results["converged_deq"] = {"model": model_conv}

    # ====== Model 3: DEQ Dual-Path with learned gate ======
    print(f"\n  [3/3] Training DEQ dual-path (our method)...")
    model_dual = DEQDualPathDenoiser(d, beta_init=2.0, gate_hidden=32).to(device)
    opt3 = torch.optim.Adam(model_dual.parameters(), lr=lr)
    sched3 = torch.optim.lr_scheduler.CosineAnnealingLR(opt3, T_max=epochs)

    for epoch in range(epochs):
        model_dual.train()
        tidx = torch.randint(K, (batch_size,))
        targets = patterns[tidx]
        queries = targets + noise_std * torch.randn(batch_size, d, device=device)

        output, out_shallow, out_deep = model_dual(queries, patterns)

        # Combined loss: supervise all three outputs
        cos_merged = 1 - F.cosine_similarity(output, targets).mean()
        cos_shallow = 1 - F.cosine_similarity(out_shallow, targets).mean()
        cos_deep = 1 - F.cosine_similarity(out_deep, targets).mean()
        loss = cos_merged + 0.3 * cos_shallow + 0.3 * cos_deep

        opt3.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model_dual.parameters(), 1.0)
        opt3.step()
        sched3.step()

        if (epoch + 1) % 30 == 0:
            model_dual.eval()
            with torch.no_grad():
                tidx_v = torch.randint(K, (n_val,))
                tgt_v = patterns[tidx_v]
                qry_v = tgt_v + noise_std * torch.randn(n_val, d, device=device)
                details = model_dual(qry_v, patterns, return_details=True)
                out_v = details["output"]
                dists = torch.cdist(out_v.unsqueeze(0), patterns.unsqueeze(0)).squeeze(0)
                acc = (dists.argmin(dim=-1) == tidx_v.to(device)).float().mean().item() * 100
                gate_mean = details["gate_alpha"].mean().item()
                div_mean = details["divergence"].mean().item()
            print(f"    Epoch {epoch+1}: acc={acc:.1f}%, beta={model_dual.beta.item():.2f}, "
                  f"gate={gate_mean:.3f}, div={div_mean:.4f}")

    results["dual_path_deq"] = {"model": model_dual}

    # ====== Evaluation ======
    print(f"\n  --- Final Evaluation ---")
    eval_results = {}

    for name, res in results.items():
        model = res["model"]
        model.eval()
        with torch.no_grad():
            tidx_v = torch.randint(K, (n_val,))
            tgt_v = patterns[tidx_v]
            qry_v = tgt_v + noise_std * torch.randn(n_val, d, device=device)

            if name == "one_step":
                out_v, weights_v = model(qry_v, patterns)
            elif name == "converged_deq":
                out_v, _ = model(qry_v, patterns)
            elif name == "dual_path_deq":
                details = model(qry_v, patterns, return_details=True)
                out_v = details["output"]

            # Accuracy
            dists = torch.cdist(out_v.unsqueeze(0), patterns.unsqueeze(0)).squeeze(0)
            acc = (dists.argmin(dim=-1) == tidx_v.to(device)).float().mean().item() * 100
            # MSE
            mse = (out_v - tgt_v).pow(2).mean().item()
            # Cosine
            cos = F.cosine_similarity(out_v, tgt_v).mean().item()

            eval_results[name] = {"acc": acc, "mse": mse, "cosine": cos}
            print(f"  {name:20s}: acc={acc:.1f}%  mse={mse:.6f}  cos={cos:.4f}")

    # Detailed dual-path analysis
    if "dual_path_deq" in results:
        model = results["dual_path_deq"]["model"]
        model.eval()
        with torch.no_grad():
            tidx_v = torch.randint(K, (n_val,))
            tgt_v = patterns[tidx_v]
            qry_v = tgt_v + noise_std * torch.randn(n_val, d, device=device)
            details = model(qry_v, patterns, return_details=True)

            # Per-path accuracy
            dists_s = torch.cdist(details["out_shallow"].unsqueeze(0),
                                   patterns.unsqueeze(0)).squeeze(0)
            dists_d = torch.cdist(details["out_deep"].unsqueeze(0),
                                   patterns.unsqueeze(0)).squeeze(0)
            dists_m = torch.cdist(details["output"].unsqueeze(0),
                                   patterns.unsqueeze(0)).squeeze(0)

            acc_s = (dists_s.argmin(dim=-1) == tidx_v.to(device)).float().mean().item() * 100
            acc_d = (dists_d.argmin(dim=-1) == tidx_v.to(device)).float().mean().item() * 100
            acc_m = (dists_m.argmin(dim=-1) == tidx_v.to(device)).float().mean().item() * 100
            acc_oracle = ((dists_s.argmin(dim=-1) == tidx_v.to(device)) |
                          (dists_d.argmin(dim=-1) == tidx_v.to(device))).float().mean().item() * 100

            gate_mean = details["gate_alpha"].mean().item()
            gate_std = details["gate_alpha"].std().item()
            div_mean = details["divergence"].mean().item()

            print(f"\n  Dual-path breakdown:")
            print(f"    Shallow path acc:  {acc_s:.1f}%")
            print(f"    Deep path acc:     {acc_d:.1f}%")
            print(f"    Merged acc:        {acc_m:.1f}%")
            print(f"    Oracle acc:        {acc_oracle:.1f}%")
            print(f"    Gate alpha:        {gate_mean:.3f} +/- {gate_std:.3f}")
            print(f"    Mean divergence:   {div_mean:.4f}")

            eval_results["dual_breakdown"] = {
                "acc_shallow": acc_s, "acc_deep": acc_d,
                "acc_merged": acc_m, "acc_oracle": acc_oracle,
                "gate_mean": gate_mean, "gate_std": gate_std,
                "div_mean": div_mean,
            }

    return eval_results, patterns


def run_full_comparison():
    """Run across multiple configs."""
    configs = [
        (16, 4, 0.5, "d=16, K=4, σ=0.5"),
        (16, 4, 0.8, "d=16, K=4, σ=0.8"),
        (32, 8, 0.5, "d=32, K=8, σ=0.5"),
        (32, 8, 0.8, "d=32, K=8, σ=0.8"),
        (64, 16, 0.5, "d=64, K=16, σ=0.5"),
        (64, 16, 0.8, "d=64, K=16, σ=0.8"),
    ]

    all_results = {}
    for d, K, noise_std, label in configs:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        eval_results, _ = train_and_evaluate(d, K, noise_std, epochs=120)
        all_results[label] = eval_results

    return all_results


def plot_comparison(all_results, save_path=None):
    """Plot final comparison."""
    labels = list(all_results.keys())
    n = len(labels)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Accuracy comparison
    ax = axes[0]
    x = np.arange(n)
    w = 0.2

    methods = ["one_step", "converged_deq", "dual_path_deq"]
    colors = {"one_step": "#e74c3c", "converged_deq": "#3498db", "dual_path_deq": "#2ecc71"}
    method_labels = {"one_step": "One-step", "converged_deq": "DEQ Converged",
                     "dual_path_deq": "DEQ Dual-Path (ours)"}

    for i, method in enumerate(methods):
        accs = [all_results[l].get(method, {}).get("acc", 0) for l in labels]
        ax.bar(x + (i - 1) * w, accs, w, color=colors[method],
               label=method_labels[method], alpha=0.85)

    # Oracle line
    oracle_accs = [all_results[l].get("dual_breakdown", {}).get("acc_oracle", 0) for l in labels]
    ax.plot(x, oracle_accs, 'k*--', markersize=10, label='Oracle routing', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Retrieval Accuracy: DEQ Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Panel 2: Gate behavior and divergence
    ax = axes[1]
    gate_means = [all_results[l].get("dual_breakdown", {}).get("gate_mean", 0.5) for l in labels]
    gate_stds = [all_results[l].get("dual_breakdown", {}).get("gate_std", 0) for l in labels]
    div_means = [all_results[l].get("dual_breakdown", {}).get("div_mean", 0) for l in labels]

    ax2 = ax.twinx()
    bars = ax.bar(x - 0.15, gate_means, 0.3, yerr=gate_stds, color='#e67e22',
                  alpha=0.7, capsize=3, label='Gate alpha (1=shallow)')
    line = ax2.plot(x + 0.15, div_means, 'D-', color='#9b59b6', markersize=8,
                    linewidth=2, label='Mean divergence')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Gate Alpha (1=trust shallow, 0=trust deep)', fontsize=10, color='#e67e22')
    ax2.set_ylabel('Mean Divergence', fontsize=10, color='#9b59b6')
    ax.set_title('Gate Behavior & Divergence', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')

    plt.tight_layout()
    save_path = save_path or RESULTS_DIR / "exp11_deq_dual_path.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    t0 = time.time()
    all_results = run_full_comparison()
    plot_comparison(all_results)

    # Save summary
    summary = {}
    for label, res in all_results.items():
        summary[label] = {k: v for k, v in res.items() if k != "model" and not isinstance(v, dict) or k == "dual_breakdown"}
    # Clean for JSON serialization
    clean_summary = {}
    for label, res in all_results.items():
        clean_summary[label] = {}
        for method in ["one_step", "converged_deq", "dual_path_deq", "dual_breakdown"]:
            if method in res:
                clean_summary[label][method] = {k: v for k, v in res[method].items()
                                                  if not isinstance(v, torch.nn.Module)}
    with open(RESULTS_DIR / "exp11_summary.json", "w") as f:
        json.dump(clean_summary, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
