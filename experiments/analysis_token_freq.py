"""
Analyze whether gradient-boosted attention helps more on rare tokens.

For each token in the test set, compute the per-token cross-entropy loss
under both the standard and boosted models. Then bin tokens by their
frequency in the training corpus and compare the mean improvement per bin.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn.functional as F
import numpy as np
import json
from collections import Counter
from pathlib import Path

from exp_lm_v2 import TransformerLM, get_wikitext_data

RESULTS_DIR = Path(__file__).parent.parent / 'results'
CKPT_DIR = RESULTS_DIR / 'checkpoints'


def find_checkpoint(name):
    """Find checkpoint file, trying new naming format first then old."""
    for prefix in ['wikitext103_', '']:
        path = CKPT_DIR / f'{prefix}{name}'
        if path.exists():
            return path
    raise FileNotFoundError(f'No checkpoint found for {name}')


def load_model(path, vocab_size, attn_type, d_model=256, n_layers=4,
               n_heads=4, n_rounds=2):
    model = TransformerLM(vocab_size, d_model, n_layers, n_heads, 256,
                          attn_type, n_rounds)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def per_token_loss(model, data, batch_size=16):
    """Return per-token cross-entropy losses and corresponding token ids."""
    all_losses = []
    all_token_ids = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            x = data[i:i+batch_size]
            logits = model(x[:, :-1])  # (B, T-1, V)
            targets = x[:, 1:]        # (B, T-1)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction='none'
            )
            all_losses.append(loss.cpu().numpy())
            all_token_ids.append(targets.reshape(-1).cpu().numpy())
            if (i // batch_size) % 20 == 0:
                print(f'  batch {i // batch_size}/{len(data) // batch_size}')
    return np.concatenate(all_losses), np.concatenate(all_token_ids)


def main():
    print('Loading data and tokenizer...')
    train_data, val_data, test_data, tokenizer, vocab_size = \
        get_wikitext_data(seq_len=256, vocab_size=16384, max_train_tokens=10_000_000)

    # Token frequencies from training data
    print('Computing token frequencies from training data...')
    train_flat = train_data.reshape(-1).numpy()
    freq = Counter(train_flat)
    total_train = len(train_flat)

    # Load models (seed 42)
    print('\nLoading standard model...')
    model_std = load_model(find_checkpoint('small_Standard_seed42.pt'),
                           vocab_size, 'standard')
    print('Computing per-token losses (standard)...')
    loss_std, token_ids = per_token_loss(model_std, test_data)
    del model_std

    print('\nLoading boosted model...')
    model_boost = load_model(find_checkpoint('small_Boosted-2_seed42.pt'),
                             vocab_size, 'boosted')
    print('Computing per-token losses (boosted)...')
    loss_boost, _ = per_token_loss(model_boost, test_data)
    del model_boost

    # Per-token improvement (positive = boosted is better)
    improvement = loss_std - loss_boost

    # Assign frequency to each test token
    token_freqs = np.array([freq.get(int(tid), 0) for tid in token_ids])

    # Bin by frequency (log scale)
    # Bins: [0,1], [2,10], [11,100], [101,1000], [1001,10000], [10001+]
    bin_edges = [0, 1, 10, 100, 1000, 10000, float('inf')]
    bin_labels = ['0-1', '2-10', '11-100', '101-1k', '1k-10k', '10k+']

    print('\n' + '='*65)
    print(f'{"Freq bin":>10} | {"N tokens":>10} | {"Mean Δloss":>10} | '
          f'{"Std loss":>10} | {"Boost loss":>10} | {"Improv %":>10}')
    print('-'*65)

    results = {}
    for i in range(len(bin_labels)):
        lo, hi = bin_edges[i], bin_edges[i+1]
        if i == 0:
            mask = token_freqs <= hi
        else:
            mask = (token_freqs > lo) & (token_freqs <= hi)

        n = mask.sum()
        if n == 0:
            continue
        mean_imp = improvement[mask].mean()
        mean_std = loss_std[mask].mean()
        mean_boost = loss_boost[mask].mean()
        pct = 100 * mean_imp / mean_std if mean_std > 0 else 0

        print(f'{bin_labels[i]:>10} | {n:>10,} | {mean_imp:>+10.4f} | '
              f'{mean_std:>10.4f} | {mean_boost:>10.4f} | {pct:>+9.2f}%')

        results[bin_labels[i]] = {
            'n_tokens': int(n),
            'mean_improvement': float(mean_imp),
            'mean_loss_std': float(mean_std),
            'mean_loss_boost': float(mean_boost),
            'relative_improvement_pct': float(pct),
        }

    print('='*65)
    print(f'\nOverall: mean improvement = {improvement.mean():.4f} nats')
    print(f'Correlation(log_freq, improvement) = '
          f'{np.corrcoef(np.log1p(token_freqs), improvement)[0,1]:.4f}')

    results['overall'] = {
        'mean_improvement': float(improvement.mean()),
        'correlation_logfreq_improvement': float(
            np.corrcoef(np.log1p(token_freqs), improvement)[0, 1]),
    }

    out_path = RESULTS_DIR / 'analysis_token_freq.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
