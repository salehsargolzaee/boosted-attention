"""
Post-hoc analysis of trained gradient-boosted attention models.

Analyses:
  1. Gate analysis — per-dimension gate values across layers
  2. Attention entropy — round 0 vs round 1 entropy distributions

All analyses use saved checkpoints, no retraining needed.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from exp_lm_v2 import (TransformerLM, CausalAttention, BoostedCausalAttention,
                        TwicingCausalAttention, get_wikitext_data)

RESULTS_DIR = Path(__file__).parent.parent / 'results'
CKPT_DIR = RESULTS_DIR / 'checkpoints'
PAPER_DIR = Path(__file__).parent.parent / 'paper'

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})


def load_model(label, seed=42):
    """Load a trained model from checkpoint."""
    ckpt_path = CKPT_DIR / f'small_{label}_seed{seed}.pt'
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    if 'Boosted' in label:
        attn_type, n_rounds, d_model = 'boosted', 2, 256
    elif 'Twicing' in label:
        attn_type, n_rounds, d_model = 'twicing', 1, 256
    elif 'fair' in label:
        attn_type, n_rounds, d_model = 'standard', 1, 288
    else:
        attn_type, n_rounds, d_model = 'standard', 1, 256

    vocab_size = 16384
    model = TransformerLM(vocab_size, d_model, 4, 4, 256, attn_type, n_rounds)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


class BoostedCausalAttentionHooked(BoostedCausalAttention):
    """Boosted attention that stores intermediate attention weights and gate values."""

    def forward(self, x):
        self._round_attns = []
        self._gate_values = []

        B, T, D = x.shape
        # Round 0
        qkv = self.W_qkvs[0](x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn.masked_fill_(mask, float('-inf'))
        attn_weights = F.softmax(attn, dim=-1)
        self._round_attns.append(attn_weights.detach())
        pred = (attn_weights @ v).transpose(1, 2).reshape(B, T, D)

        output = pred
        cumulative = pred

        for i in range(1, self.n_rounds):
            residual = x - cumulative

            qkv = self.W_qkvs[i](residual).reshape(B, T, 3, self.n_heads, self.d_head)
            q, k, v = qkv.unbind(dim=2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn.masked_fill_(mask, float('-inf'))
            attn_weights = F.softmax(attn, dim=-1)
            self._round_attns.append(attn_weights.detach())
            correction = (attn_weights @ v).transpose(1, 2).reshape(B, T, D)

            gate = self.gates[i - 1](torch.cat([cumulative, correction], dim=-1))
            self._gate_values.append(gate.detach())

            gated = gate * correction
            output = output + gated
            cumulative = cumulative + gated

        return self.W_out(output)


def hook_boosted_model(model):
    """Replace BoostedCausalAttention layers with hooked versions."""
    for layer in model.layers:
        attn = layer['attn']
        if isinstance(attn, BoostedCausalAttention):
            hooked = BoostedCausalAttentionHooked(
                attn.W_qkvs[0].in_features, attn.n_heads, attn.n_rounds)
            hooked.load_state_dict(attn.state_dict())
            hooked.eval()
            layer['attn'] = hooked
    return model


# ============================================================
# Analysis 1: Gate values across layers
# ============================================================

def analysis_gate_values(model, test_data):
    """Per-dimension gate values averaged over test sequences, per layer."""
    print('\n=== Analysis 1: Gate Values ===')

    gate_stats = {i: [] for i in range(4)}
    n_batches = min(50, len(test_data))

    with torch.no_grad():
        for b in range(n_batches):
            x = test_data[b:b+1]
            _ = model(x[:, :-1])
            for i, layer in enumerate(model.layers):
                attn = layer['attn']
                if hasattr(attn, '_gate_values') and attn._gate_values:
                    g = attn._gate_values[0].numpy()  # (1, T, d)
                    gate_stats[i].append(g.mean(axis=(0, 1)))

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i in range(4):
        ax = axes[i]
        if gate_stats[i]:
            all_gates = np.stack(gate_stats[i])
            mean_per_dim = all_gates.mean(axis=0)

            ax.bar(range(len(mean_per_dim)), mean_per_dim, alpha=0.7,
                   color='#2980b9', width=1.0)
            ax.set_ylim(0, 1)
            ax.set_xlabel('Dimension')
            ax.set_title(f'Layer {i}', fontsize=10, fontweight='bold')
            ax.axhline(0.5, color='gray', ls='--', lw=0.8, alpha=0.5)

            mu = mean_per_dim.mean()
            sigma = mean_per_dim.std()
            ax.text(0.95, 0.95, f'$\\mu$={mu:.2f}\n$\\sigma$={sigma:.2f}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            print(f'  Layer {i}: mean={mu:.3f}, std={sigma:.3f}')

    axes[0].set_ylabel('Gate value')
    plt.suptitle('Learned Gate Values per Layer (averaged over test sequences)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(PAPER_DIR / f'fig_gate_analysis.{ext}')
    plt.close()
    print('  Saved fig_gate_analysis')


# ============================================================
# Analysis 2: Attention entropy — round 0 vs round 1
# ============================================================

def analysis_attention_entropy(model, test_data):
    """Compare entropy of attention distributions between round 0 and round 1."""
    print('\n=== Analysis 2: Attention Entropy ===')

    entropy_r0_all = []
    entropy_r1_all = []
    n_batches = min(50, len(test_data))

    with torch.no_grad():
        for b in range(n_batches):
            x = test_data[b:b+1]
            _ = model(x[:, :-1])
            for layer in model.layers:
                attn = layer['attn']
                if hasattr(attn, '_round_attns') and len(attn._round_attns) >= 2:
                    # attn weights shape: (B, n_heads, T, T)
                    a0 = attn._round_attns[0]
                    a1 = attn._round_attns[1]
                    # Entropy per query position: -sum(p log p), averaged over heads
                    # Clamp to avoid log(0)
                    e0 = -(a0 * torch.log(a0.clamp(min=1e-10))).sum(dim=-1)  # (B, H, T)
                    e1 = -(a1 * torch.log(a1.clamp(min=1e-10))).sum(dim=-1)
                    entropy_r0_all.append(e0.reshape(-1).numpy())
                    entropy_r1_all.append(e1.reshape(-1).numpy())

    entropy_r0 = np.concatenate(entropy_r0_all)
    entropy_r1 = np.concatenate(entropy_r1_all)

    print(f'  Round 0 entropy: mean={entropy_r0.mean():.3f}, median={np.median(entropy_r0):.3f}')
    print(f'  Round 1 entropy: mean={entropy_r1.mean():.3f}, median={np.median(entropy_r1):.3f}')
    print(f'  Entropy reduction: {(entropy_r0.mean() - entropy_r1.mean()) / entropy_r0.mean() * 100:.1f}% relative')

    # Only produce figure if the difference is meaningful
    diff = entropy_r0.mean() - entropy_r1.mean()
    if abs(diff) < 0.01 * entropy_r0.mean():
        print('  Entropy difference < 1% — not significant enough to plot.')
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    # Left: overlaid histograms
    bins = np.linspace(0, max(entropy_r0.max(), entropy_r1.max()), 60)
    ax1.hist(entropy_r0, bins=bins, alpha=0.6, color='#2980b9', label='Round 0 (initial)', density=True)
    ax1.hist(entropy_r1, bins=bins, alpha=0.6, color='#e74c3c', label='Round 1 (correction)', density=True)
    ax1.set_xlabel('Attention entropy (nats)')
    ax1.set_ylabel('Density')
    ax1.set_title('Attention Entropy Distribution', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2, ls='--')

    # Right: per-layer breakdown
    # Re-collect per layer
    layer_entropies = {i: {'r0': [], 'r1': []} for i in range(4)}
    with torch.no_grad():
        for b in range(n_batches):
            x = test_data[b:b+1]
            _ = model(x[:, :-1])
            for i, layer in enumerate(model.layers):
                attn = layer['attn']
                if hasattr(attn, '_round_attns') and len(attn._round_attns) >= 2:
                    a0 = attn._round_attns[0]
                    a1 = attn._round_attns[1]
                    e0 = -(a0 * torch.log(a0.clamp(min=1e-10))).sum(dim=-1).mean().item()
                    e1 = -(a1 * torch.log(a1.clamp(min=1e-10))).sum(dim=-1).mean().item()
                    layer_entropies[i]['r0'].append(e0)
                    layer_entropies[i]['r1'].append(e1)

    layers = range(4)
    r0_means = [np.mean(layer_entropies[i]['r0']) for i in layers]
    r1_means = [np.mean(layer_entropies[i]['r1']) for i in layers]
    r0_stds = [np.std(layer_entropies[i]['r0']) for i in layers]
    r1_stds = [np.std(layer_entropies[i]['r1']) for i in layers]

    x_pos = np.arange(4)
    w = 0.35
    ax2.bar(x_pos - w/2, r0_means, w, yerr=r0_stds, color='#2980b9', alpha=0.8,
            label='Round 0', capsize=3)
    ax2.bar(x_pos + w/2, r1_means, w, yerr=r1_stds, color='#e74c3c', alpha=0.8,
            label='Round 1', capsize=3)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Mean entropy (nats)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'Layer {i}' for i in range(4)], fontsize=9)
    ax2.set_title('Entropy by Layer', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.2, ls='--')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(PAPER_DIR / f'fig_attention_entropy.{ext}')
    plt.close()
    print('  Saved fig_attention_entropy')
    return True


# ============================================================
# Analysis 3: Example-level correction cases
# ============================================================

def analysis_example_corrections(model_std, model_boosted, test_data, tokenizer):
    """Find tokens where the correction round fixes a prediction error,
    and visualize what round 0 vs round 1 attend to."""
    print('\n=== Analysis 3: Example Corrections ===')

    # Step 1: Find positions with largest per-token loss improvement
    candidates = []  # (batch_idx, token_pos, loss_std, loss_boost, target_id)
    n_batches = min(200, len(test_data))

    with torch.no_grad():
        for b in range(n_batches):
            x = test_data[b:b+1]
            logits_std = model_std(x[:, :-1])
            logits_boost = model_boosted(x[:, :-1])
            targets = x[:, 1:]

            loss_std = F.cross_entropy(
                logits_std.reshape(-1, logits_std.size(-1)),
                targets.reshape(-1), reduction='none')
            loss_boost = F.cross_entropy(
                logits_boost.reshape(-1, logits_boost.size(-1)),
                targets.reshape(-1), reduction='none')

            improvement = loss_std - loss_boost  # positive = boosted is better
            for pos in range(len(improvement)):
                if improvement[pos] > 1.5:  # substantial improvement (>1.5 nats)
                    # Get top-1 predictions
                    pred_std = logits_std[0, pos].argmax().item()
                    pred_boost = logits_boost[0, pos].argmax().item()
                    target = targets[0, pos].item()
                    # Only keep if boosted gets it right and standard doesn't,
                    # or boosted is much closer
                    candidates.append({
                        'batch': b, 'pos': pos,
                        'loss_std': loss_std[pos].item(),
                        'loss_boost': loss_boost[pos].item(),
                        'improvement': improvement[pos].item(),
                        'pred_std': pred_std, 'pred_boost': pred_boost,
                        'target': target,
                        'boost_correct': pred_boost == target,
                        'std_correct': pred_std == target,
                    })

    # Sort by improvement, prefer cases where boosted is correct and standard isn't
    candidates.sort(key=lambda c: (c['boost_correct'] and not c['std_correct'],
                                    c['improvement']), reverse=True)

    print(f'  Found {len(candidates)} positions with >1.5 nat improvement')
    print(f'  Of those, {sum(c["boost_correct"] and not c["std_correct"] for c in candidates)} '
          f'have boosted correct & standard wrong')

    # Step 2: Select up to 3 good examples from DIFFERENT sequences
    selected = []
    used_batches = set()
    for c in candidates:
        if len(selected) >= 3:
            break
        b, pos = c['batch'], c['pos']
        if any(abs(b - ub) < 50 for ub in used_batches):
            continue  # enforce diversity: skip nearby sequences (likely same article)
        if pos < 10:
            continue
        x = test_data[b]
        context_ids = x[max(0, pos-11):pos+2].tolist()
        context_text = tokenizer.decode(context_ids)
        if len(context_text.strip()) < 20:
            continue
        c['context_ids'] = context_ids
        c['context_start'] = max(0, pos-11)
        selected.append(c)
        used_batches.add(b)

    if len(selected) < 2:
        print('  Not enough good examples found.')
        return

    # Step 3: Overlaid attention bars for each example
    fig, axes = plt.subplots(len(selected), 1, figsize=(7, 2.5 * len(selected)))
    if len(selected) == 1:
        axes = [axes]

    for row, c in enumerate(selected):
        b, pos = c['batch'], c['pos']
        x = test_data[b:b+1]

        with torch.no_grad():
            _ = model_boosted(x[:, :-1])

        # Use layer 1 (strongest correction per entropy/gate analysis)
        layer = model_boosted.layers[1]['attn']
        attn_r0 = layer._round_attns[0][0].mean(dim=0).numpy()
        attn_r1 = layer._round_attns[1][0].mean(dim=0).numpy()

        ctx_start = c['context_start']
        ctx_end = pos + 1
        n_ctx = ctx_end - ctx_start

        attn_r0_row = attn_r0[pos, ctx_start:ctx_end]
        attn_r1_row = attn_r1[pos, ctx_start:ctx_end]

        # Decode token labels, joining BPE fragments with previous token
        raw_labels = []
        for tid in x[0, ctx_start:ctx_end].tolist():
            tok = tokenizer.decode([tid])
            if len(tok) > 12:
                tok = tok[:11] + '.'
            raw_labels.append(tok.replace('\n', ' '))

        target_tok = tokenizer.decode([c['target']]).strip()
        pred_std_tok = tokenizer.decode([c['pred_std']]).strip()
        pred_boost_tok = tokenizer.decode([c['pred_boost']]).strip()

        ax = axes[row]
        x_pos = np.arange(n_ctx)
        w = 0.35
        ax.bar(x_pos - w/2, attn_r0_row, w, color='#2980b9', alpha=0.8, label='Round 0')
        ax.bar(x_pos + w/2, attn_r1_row, w, color='#e74c3c', alpha=0.8, label='Round 1')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(raw_labels, rotation=40, ha='right', fontsize=7)
        ax.set_ylabel('Attn weight', fontsize=8)
        ax.grid(axis='y', alpha=0.15, ls='--')
        if row == 0:
            ax.legend(fontsize=7, loc='upper left', ncol=2)

        # Title with prediction info
        marker = '\u2713' if c['boost_correct'] else ''
        ax.set_title(
            f'Target: "{target_tok}"    '
            f'Standard: "{pred_std_tok}" (loss {c["loss_std"]:.1f})    '
            f'Boosted: "{pred_boost_tok}" {marker} (loss {c["loss_boost"]:.1f})',
            fontsize=8, fontfamily='monospace', pad=8)

    plt.suptitle('Example Corrections: Attention Redistribution in Layer 1 (head-averaged)',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(PAPER_DIR / f'fig_example_corrections.{ext}')
    plt.close()
    print('  Saved fig_example_corrections')

    # Print details for verification
    for i, c in enumerate(selected):
        target_tok = tokenizer.decode([c['target']]).strip()
        pred_std_tok = tokenizer.decode([c['pred_std']]).strip()
        pred_boost_tok = tokenizer.decode([c['pred_boost']]).strip()
        ctx = tokenizer.decode(c['context_ids']).strip()
        print(f'\n  Example {i+1}:')
        print(f'    Context: ...{ctx}')
        print(f'    Target: "{target_tok}", Std pred: "{pred_std_tok}", Boost pred: "{pred_boost_tok}"')
        print(f'    Loss std={c["loss_std"]:.2f}, boost={c["loss_boost"]:.2f}, '
              f'improvement={c["improvement"]:.2f} nats')


# ============================================================

if __name__ == '__main__':
    print('Loading data...')
    _, _, test_data, tokenizer, actual_vocab = get_wikitext_data(
        seq_len=256, vocab_size=16384, max_train_tokens=100_000)

    print('Loading models...')
    model_std = load_model('Standard', seed=42)
    model_boosted = load_model('Boosted-2', seed=42)
    model_boosted = hook_boosted_model(model_boosted)
    print(f'Standard params: {sum(p.numel() for p in model_std.parameters()):,}')
    print(f'Boosted params: {sum(p.numel() for p in model_boosted.parameters()):,}')

    analysis_gate_values(model_boosted, test_data)
    entropy_significant = analysis_attention_entropy(model_boosted, test_data)
    analysis_example_corrections(model_std, model_boosted, test_data, tokenizer)

    if not entropy_significant:
        print('\nNote: Entropy analysis not significant — only gate figure produced.')

    print('\nDone.')
