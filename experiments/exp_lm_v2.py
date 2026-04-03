"""
WikiText-103 Language Modeling: Standard vs Boosted vs Twicing.

Multi-scale experiment for the Boosted Attention paper.
Scales: small (d=256, 4L), medium (d=384, 6L), optionally large (d=512, 8L).
Each config runs with 2 seeds. Includes param-fair baselines.

Usage:
  python experiments/exp_lm_v2.py --scale small          # ~2h on A100
  python experiments/exp_lm_v2.py --scale medium         # ~8h on A100
  python experiments/exp_lm_v2.py --scale all            # ~12h on A100
  python experiments/exp_lm_v2.py --scale small --dry    # print configs only
"""

import sys, os, json, math, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================
# Data
# ============================================================

def get_wikitext_data(seq_len=256, vocab_size=16384, max_train_tokens=None):
    """Load WikiText-103 with BPE tokenization."""
    print('Loading WikiText-103...')
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', trust_remote_code=True)

    print(f'Training BPE tokenizer (vocab={vocab_size})...')
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=['<pad>', '<unk>'])
    texts = [t for t in ds['train']['text'] if len(t.strip()) > 0]
    tokenizer.train_from_iterator(texts[:100000], trainer)

    actual_vocab = tokenizer.get_vocab_size()
    print(f'Actual vocab size: {actual_vocab}')

    def encode_split(split_name, max_tokens=None):
        texts = [t for t in ds[split_name]['text'] if len(t.strip()) > 0]
        all_ids = []
        for t in texts:
            ids = tokenizer.encode(t).ids
            all_ids.extend(ids)
            if max_tokens and len(all_ids) >= max_tokens:
                all_ids = all_ids[:max_tokens]
                break
        return torch.tensor(all_ids, dtype=torch.long)

    train_ids = encode_split('train', max_tokens=max_train_tokens)
    val_ids = encode_split('validation', max_tokens=1_000_000)
    test_ids = encode_split('test', max_tokens=1_000_000)

    def batchify(ids, seq_len):
        n = len(ids) // seq_len
        return ids[:n * seq_len].reshape(n, seq_len)

    train_data = batchify(train_ids, seq_len)
    val_data = batchify(val_ids, seq_len)
    test_data = batchify(test_ids, seq_len)
    print(f'Train: {len(train_data)*seq_len:,} tokens, '
          f'Val: {len(val_data)*seq_len:,}, Test: {len(test_data)*seq_len:,}')
    return train_data, val_data, test_data, tokenizer, actual_vocab


# ============================================================
# Attention Modules
# ============================================================

class CausalAttention(nn.Module):
    """Standard multi-head causal attention."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.W_qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn.masked_fill_(mask, float('-inf'))
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.W_out(out)


class BoostedCausalAttention(nn.Module):
    """Boosted Attention: M rounds with separate QKV and learned gate."""
    def __init__(self, d_model, n_heads, n_rounds=2, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.n_rounds = n_rounds
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.W_qkvs = nn.ModuleList([
            nn.Linear(d_model, 3 * d_model) for _ in range(n_rounds)
        ])
        self.W_out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())
            for _ in range(n_rounds - 1)
        ])

    def _attend(self, x, round_idx):
        B, T, D = x.shape
        qkv = self.W_qkvs[round_idx](x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn.masked_fill_(mask, float('-inf'))
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        return (attn @ v).transpose(1, 2).reshape(B, T, D)

    def forward(self, x):
        pred = self._attend(x, 0)
        output = pred
        cumulative = pred
        for i in range(1, self.n_rounds):
            residual = x - cumulative
            correction = self._attend(residual, i)
            gate = self.gates[i - 1](torch.cat([cumulative, correction], dim=-1))
            gated = gate * correction
            output = output + gated
            cumulative = cumulative + gated
        return self.W_out(output)


class TwicingCausalAttention(nn.Module):
    """
    Twicing Attention (Abdullaev & Nguyen, ICLR 2025).

    Output = (2A - A^2)V = AV + A(V - AV)
    Uses the SAME attention matrix A for both passes.
    No learned gate, no separate projections.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.W_qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Compute attention weights (shared for both passes)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn.masked_fill_(mask, float('-inf'))
        A = self.attn_drop(F.softmax(attn, dim=-1))

        # Twicing: AV + A(V - AV) = (2A - A^2)V
        Av = A @ v                   # first pass
        residual = v - Av            # what attention missed
        correction = A @ residual    # smooth the residual with same A
        out = (Av + correction).transpose(1, 2).reshape(B, T, D)
        return self.W_out(out)


# ============================================================
# Transformer LM
# ============================================================

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, max_seq=256,
                 attn_type='standard', n_rounds=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq, d_model)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            if attn_type == 'standard':
                attn = CausalAttention(d_model, n_heads, dropout)
            elif attn_type == 'boosted':
                attn = BoostedCausalAttention(d_model, n_heads, n_rounds, dropout)
            elif attn_type == 'twicing':
                attn = TwicingCausalAttention(d_model, n_heads, dropout)
            else:
                raise ValueError(f'Unknown attn_type: {attn_type}')
            self.layers.append(nn.ModuleDict({
                'attn': attn,
                'n1': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, 4 * d_model), nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(4 * d_model, d_model), nn.Dropout(dropout)),
                'n2': nn.LayerNorm(d_model),
            }))

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.shape
        h = self.drop(self.embed(x) + self.pos_embed(torch.arange(T, device=x.device)))
        for layer in self.layers:
            h = h + layer['attn'](layer['n1'](h))
            h = h + layer['ffn'](layer['n2'](h))
        return self.head(self.ln_f(h))


# ============================================================
# Training
# ============================================================

def train_lm(model, train_data, val_data, test_data,
             epochs=15, batch_size=32, lr=3e-4, warmup_steps=1500,
             device=DEVICE, save_path=None):
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {n_params:,}')

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps_per_epoch = len(train_data) // batch_size
    total_steps = epochs * steps_per_epoch
    print(f'  Steps/epoch: {steps_per_epoch}, Total: {total_steps}, Warmup: {warmup_steps}')

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    history = {'train_loss': [], 'val_ppl': [], 'epoch': []}
    best_val_ppl = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_data))
        total_loss, n_batches = 0.0, 0
        for i in range(0, len(train_data) - batch_size, batch_size):
            idx = perm[i:i+batch_size]
            x = train_data[idx].to(device)
            logits = model(x[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   x[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / n_batches

        # Validation
        model.eval()
        val_loss, val_n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val_data) - batch_size, batch_size):
                x = val_data[i:i+batch_size].to(device)
                logits = model(x[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       x[:, 1:].reshape(-1))
                val_loss += loss.item()
                val_n += 1
        val_ppl = math.exp(val_loss / val_n)

        # Track best model
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history['train_loss'].append(train_loss)
        history['val_ppl'].append(val_ppl)
        history['epoch'].append(epoch + 1)

        lr_now = sched.get_last_lr()[0]
        print(f'  Epoch {epoch+1:2d}: train={train_loss:.1f}  val={val_ppl:.1f}  lr={lr_now:.2e}')

    # Restore best model for test eval
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    # Test evaluation
    model.eval()
    test_loss, test_n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(test_data) - batch_size, batch_size):
            x = test_data[i:i+batch_size].to(device)
            logits = model(x[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   x[:, 1:].reshape(-1))
            test_loss += loss.item()
            test_n += 1
    test_ppl = math.exp(test_loss / test_n)
    print(f'  Test perplexity: {test_ppl:.1f} (best val: {best_val_ppl:.1f})')

    # Save checkpoint
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': best_state,
            'n_params': n_params,
            'test_ppl': test_ppl,
            'best_val_ppl': best_val_ppl,
            'history': history,
        }, save_path)
        print(f'  Saved checkpoint: {save_path}')

    return history, n_params, test_ppl


# ============================================================
# Scale configs
# ============================================================

SCALES = {
    'small': {
        'd_model': 256, 'n_layers': 4, 'n_heads': 4,
        'epochs': 15, 'batch_size': 32, 'lr': 3e-4,
        'warmup_steps': 1500, 'max_train_tokens': 10_000_000,
    },
    'medium': {
        'd_model': 384, 'n_layers': 6, 'n_heads': 6,
        'epochs': 10, 'batch_size': 24, 'lr': 2e-4,
        'warmup_steps': 2000, 'max_train_tokens': 30_000_000,
    },
}


def find_param_fair_d(vocab_size, target_params, n_layers, n_heads, seq_len,
                      attn_type_target='boosted'):
    """Find d_model for standard attention that matches target param count."""
    for d in range(256, 768, 4):
        if d % n_heads != 0:
            continue
        m = TransformerLM(vocab_size, d, n_layers, n_heads, seq_len, 'standard', 1)
        n = sum(p.numel() for p in m.parameters())
        del m
        if n >= target_params:
            return d, n
    return None, None


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', default='small', choices=['small', 'medium', 'all'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123])
    parser.add_argument('--dry', action='store_true', help='Print configs and exit')
    args = parser.parse_args()

    scales = list(SCALES.keys()) if args.scale == 'all' else [args.scale]

    t0 = time.time()
    seq_len = 256

    # Determine max tokens needed
    max_tokens = max(SCALES[s]['max_train_tokens'] for s in scales)
    train_data, val_data, test_data, tokenizer, actual_vocab = get_wikitext_data(
        seq_len, vocab_size=16384, max_train_tokens=max_tokens)

    all_results = {}

    for scale_name in scales:
        cfg = SCALES[scale_name]
        d, nl, nh = cfg['d_model'], cfg['n_layers'], cfg['n_heads']

        # Trim training data for this scale
        n_keep = min(len(train_data), cfg['max_train_tokens'] // seq_len)
        train_sub = train_data[:n_keep]
        print(f'\n{"="*70}')
        print(f'  SCALE: {scale_name}  (d={d}, {nl}L, {nh}H, '
              f'{len(train_sub)*seq_len:,} tokens)')
        print(f'{"="*70}')

        # Build config list
        configs = [
            (f'{scale_name}/Standard', 'standard', d, nl, nh, 1),
            (f'{scale_name}/Boosted-2', 'boosted', d, nl, nh, 2),
            (f'{scale_name}/Twicing', 'twicing', d, nl, nh, 1),
        ]

        # Find param-fair d for Boosted-2
        m_b = TransformerLM(actual_vocab, d, nl, nh, seq_len, 'boosted', 2)
        n_boost = sum(p.numel() for p in m_b.parameters())
        del m_b
        fair_d, fair_n = find_param_fair_d(actual_vocab, n_boost, nl, nh, seq_len)
        if fair_d:
            configs.append(
                (f'{scale_name}/Std-fair(d={fair_d})', 'standard', fair_d, nl, nh, 1))
            print(f'  Param-fair: d={fair_d}, {fair_n:,} params '
                  f'(boosted={n_boost:,})')

        if args.dry:
            for label, atype, dm, nlr, nhr, nr in configs:
                m = TransformerLM(actual_vocab, dm, nlr, nhr, seq_len, atype, nr)
                n = sum(p.numel() for p in m.parameters())
                print(f'  {label}: {n:,} params')
                del m
            continue

        # Run all configs
        for label, attn_type, d_model, n_layers, n_heads, n_rounds in configs:
            seed_results = []
            for seed in args.seeds:
                print(f'\n===== {label} [seed={seed}] =====')
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

                model = TransformerLM(
                    actual_vocab, d_model, n_layers, n_heads,
                    seq_len, attn_type, n_rounds)

                # Save checkpoint for future downstream eval
                ckpt_name = f'{label.replace("/", "_")}_seed{seed}.pt'
                ckpt_path = RESULTS_DIR / 'checkpoints' / ckpt_name

                history, n_params, test_ppl = train_lm(
                    model, train_sub, val_data, test_data,
                    epochs=cfg['epochs'], batch_size=cfg['batch_size'],
                    lr=cfg['lr'], warmup_steps=cfg['warmup_steps'],
                    save_path=str(ckpt_path))

                seed_results.append({
                    'test_ppl': test_ppl,
                    'val_ppl': history['val_ppl'][-1],
                    'n_params': n_params,
                    'history': {k: v for k, v in history.items()},
                })
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            test_ppls = [r['test_ppl'] for r in seed_results]
            val_ppls = [r['val_ppl'] for r in seed_results]
            np_ = seed_results[0]['n_params']
            all_results[label] = {
                'test_mean': float(np.mean(test_ppls)),
                'test_std': float(np.std(test_ppls)),
                'val_mean': float(np.mean(val_ppls)),
                'val_std': float(np.std(val_ppls)),
                'n_params': np_,
            }
            tmean = np.mean(test_ppls)
            tstd = np.std(test_ppls)
            print(f'>>> {label}: test={tmean:.1f}+-{tstd:.1f}  params={np_:,}')

    if not args.dry:
        # Print summary
        print(f'\n{"="*70}')
        print('FINAL SUMMARY')
        print(f'{"="*70}')
        for l, r in all_results.items():
            print(f'  {l:<35s} test={r["test_mean"]:6.1f}+-{r["test_std"]:.1f}  '
                  f'val={r["val_mean"]:6.1f}+-{r["val_std"]:.1f}  '
                  f'params={r["n_params"]:,}')
        elapsed = time.time() - t0
        print(f'\nTotal: {elapsed:.0f}s ({elapsed/60:.1f}min)')

        # Save
        summary = {k: v for k, v in all_results.items()}
        outfile = RESULTS_DIR / f'exp_v2_{"_".join(scales)}.json'
        with open(outfile, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'Saved: {outfile}')
