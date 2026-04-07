"""
Scaled training on OpenWebText: Standard vs Boosted vs Twicing vs Param-fair.

Supports 125M and 350M parameter configurations.
Designed for long unattended runs on RunPod with auto-resume.

Usage:
  # Single run
  python train_openwebtext.py --scale 125m --attn standard --seed 42

  # See all configs
  python train_openwebtext.py --list-configs
"""

import os, sys, json, math, time, argparse, signal, datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import GradScaler, autocast

# ============================================================
# Config
# ============================================================

CONFIGS = {
    '125m': {
        'd_model': 768, 'n_layers': 12, 'n_heads': 12, 'd_ff': 3072,
        'seq_len': 1024, 'lr': 6e-4, 'min_lr': 6e-5,
        'micro_batch': 32, 'grad_accum': 16,  # effective batch = 32*1024*16 = 524K tokens
        'warmup_steps': 2000, 'dropout': 0.0,
    },
    '350m': {
        'd_model': 1024, 'n_layers': 24, 'n_heads': 16, 'd_ff': 4096,
        'seq_len': 1024, 'lr': 3e-4, 'min_lr': 3e-5,
        'micro_batch': 16, 'grad_accum': 32,  # effective batch = 16*1024*32 = 524K tokens
        'warmup_steps': 2000, 'dropout': 0.0,
    },
}

# Param-fair d_model: chosen so standard model matches boosted param count
PARAM_FAIR_D = {
    '125m': 896,   # ~160M params to match boosted-125M
    '350m': 1184,  # ~455M params to match boosted-350M
}

DATA_DIR = Path(os.environ.get('DATA_DIR', '/workspace/data'))
CKPT_DIR = Path(os.environ.get('CKPT_DIR', '/workspace/checkpoints'))
LOG_DIR = Path(os.environ.get('LOG_DIR', '/workspace/logs'))

VOCAB_SIZE = 50257  # GPT-2 tokenizer


# ============================================================
# Model
# ============================================================

class CausalAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.W_qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, T, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.attn_drop.p if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.resid_drop(self.W_out(out))


class BoostedCausalAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_rounds=2, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_rounds = n_rounds
        self.d_head = d_model // n_heads
        self.W_qkvs = nn.ModuleList([
            nn.Linear(d_model, 3 * d_model, bias=False) for _ in range(n_rounds)
        ])
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * d_model, d_model, bias=False), nn.Sigmoid())
            for _ in range(n_rounds - 1)
        ])
        self.W_out = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop_p = dropout
        self.resid_drop = nn.Dropout(dropout)

    def _attend(self, qkv_proj, x_query):
        B, T, D = x_query.shape
        qkv = qkv_proj(x_query).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                             dropout_p=self.attn_drop_p if self.training else 0.0)
        return out.transpose(1, 2).reshape(B, T, D)

    def forward(self, x):
        # Round 0
        pred = self._attend(self.W_qkvs[0], x)
        cumulative = pred

        for i in range(1, self.n_rounds):
            residual = x - cumulative
            correction = self._attend(self.W_qkvs[i], residual)
            gate = self.gates[i - 1](torch.cat([cumulative, correction], dim=-1))
            cumulative = cumulative + gate * correction

        return self.resid_drop(self.W_out(cumulative))


class TwicingCausalAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop_p = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.W_qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        drop_p = self.attn_drop_p if self.training else 0.0

        # Compute A and AV manually for twicing: (2A - A^2)V = 2*AV - A*(AV)
        scale = 1.0 / math.sqrt(self.d_head)
        scores = q @ k.transpose(-2, -1) * scale
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(causal_mask, float('-inf'))
        A = F.softmax(scores, dim=-1)
        if drop_p > 0 and self.training:
            A = F.dropout(A, p=drop_p)

        AV = A @ v                  # first pass
        AAV = A @ AV                # second pass (reuses A)
        out = 2 * AV - AAV          # (2A - A^2)V

        out = out.transpose(1, 2).reshape(B, T, D)
        return self.resid_drop(self.W_out(out))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, attn_type='standard', n_rounds=2, dropout=0.0):
        super().__init__()
        if attn_type == 'standard':
            self.attn = CausalAttention(d_model, n_heads, dropout)
        elif attn_type == 'boosted':
            self.attn = BoostedCausalAttention(d_model, n_heads, n_rounds, dropout)
        elif attn_type == 'twicing':
            self.attn = TwicingCausalAttention(d_model, n_heads, dropout)
        else:
            raise ValueError(f'Unknown attn_type: {attn_type}')

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, seq_len,
                 attn_type='standard', n_rounds=2, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, attn_type, n_rounds, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def build_model(scale, attn_type, device):
    cfg = CONFIGS[scale].copy()
    d_model = cfg['d_model']
    n_rounds = 2

    if attn_type == 'param_fair':
        d_model = PARAM_FAIR_D[scale]
        actual_attn = 'standard'
    else:
        actual_attn = attn_type

    # Adjust FFN proportionally if d_model changed
    d_ff = 4 * d_model

    model = TransformerLM(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        n_layers=cfg['n_layers'],
        n_heads=cfg['n_heads'] if attn_type != 'param_fair' else d_model // (cfg['d_model'] // cfg['n_heads']),
        d_ff=d_ff,
        seq_len=cfg['seq_len'],
        attn_type=actual_attn,
        n_rounds=n_rounds,
        dropout=cfg['dropout'],
    ).to(device)

    return model


# ============================================================
# Data
# ============================================================

def prepare_data(seq_len):
    """Tokenize OpenWebText and save as memory-mapped numpy array."""
    token_file = DATA_DIR / f'openwebtext_gpt2_{seq_len}.bin'
    meta_file = DATA_DIR / f'openwebtext_gpt2_{seq_len}_meta.json'

    if token_file.exists() and meta_file.exists():
        meta = json.loads(meta_file.read_text())
        tokens = np.memmap(token_file, dtype=np.uint16, mode='r',
                          shape=(meta['n_sequences'], seq_len + 1))
        print(f'Loaded cached data: {meta["n_sequences"]:,} sequences, '
              f'{meta["total_tokens"]:,} tokens')
        return tokens

    print('Preparing OpenWebText data (this takes ~30-60 min first time)...')
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    ds = load_dataset('openwebtext', split='train')

    # Tokenize all documents and concatenate
    print(f'Tokenizing {len(ds):,} documents...')
    all_tokens = []
    total = 0
    for i, doc in enumerate(ds):
        toks = tokenizer.encode(doc['text'])
        all_tokens.extend(toks)
        total += len(toks)
        if (i + 1) % 500_000 == 0:
            print(f'  {i+1:,} docs, {total:,} tokens so far...')

    print(f'Total tokens: {total:,}')

    # Split into sequences of (seq_len + 1) for input/target
    chunk_size = seq_len + 1
    n_sequences = len(all_tokens) // chunk_size
    all_tokens = all_tokens[:n_sequences * chunk_size]

    # Save as memory-mapped file
    arr = np.array(all_tokens, dtype=np.uint16).reshape(n_sequences, chunk_size)
    fp = np.memmap(token_file, dtype=np.uint16, mode='w+', shape=arr.shape)
    fp[:] = arr[:]
    fp.flush()

    meta = {'n_sequences': n_sequences, 'total_tokens': total, 'seq_len': seq_len}
    meta_file.write_text(json.dumps(meta))
    print(f'Saved {n_sequences:,} sequences to {token_file}')

    return np.memmap(token_file, dtype=np.uint16, mode='r', shape=(n_sequences, chunk_size))


class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, data, start_frac=0.0, end_frac=1.0):
        n = len(data)
        self.data = data[int(n * start_frac):int(n * end_frac)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = torch.from_numpy(self.data[idx].astype(np.int64))
        return chunk[:-1], chunk[1:]  # input, target


# ============================================================
# Training
# ============================================================

def get_lr(step, warmup_steps, max_lr, min_lr, total_steps):
    """Cosine decay with linear warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def save_checkpoint(model, optimizer, scaler, step, best_val_loss, run_name):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = CKPT_DIR / f'{run_name}_latest.pt'
    torch.save({
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'best_val_loss': best_val_loss,
    }, path)
    return path


def load_checkpoint(run_name, model, optimizer, scaler, device):
    path = CKPT_DIR / f'{run_name}_latest.pt'
    if not path.exists():
        return 0, float('inf')
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scaler.load_state_dict(ckpt['scaler'])
    print(f'Resumed from step {ckpt["step"]}')
    return ckpt['step'], ckpt['best_val_loss']


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=100):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with autocast(dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    model.train()
    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)


class Logger:
    def __init__(self, run_name, use_wandb=False):
        self.run_name = run_name
        self.use_wandb = use_wandb
        self.log_file = LOG_DIR / f'{run_name}.jsonl'
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        if use_wandb:
            import wandb
            wandb.init(project='boosted-attention', name=run_name, resume='allow')

    def log(self, data, step):
        data['step'] = step
        data['timestamp'] = datetime.datetime.now().isoformat()
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(data) + '\n')
        if self.use_wandb:
            import wandb
            wandb.log(data, step=step)

    def print_status(self, data, step, total_steps):
        elapsed = data.get('elapsed_min', 0)
        pct = step / total_steps * 100
        parts = [
            f'step {step}/{total_steps} ({pct:.1f}%)',
            f'loss={data.get("train_loss", 0):.4f}',
            f'lr={data.get("lr", 0):.2e}',
            f'tok/s={data.get("tokens_per_sec", 0):.0f}',
            f'elapsed={elapsed:.0f}min',
        ]
        if 'val_ppl' in data:
            parts.append(f'val_ppl={data["val_ppl"]:.1f}')
        print(f'  [{self.run_name}] {" | ".join(parts)}', flush=True)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = CONFIGS[args.scale]
    run_name = f'{args.scale}_{args.attn}_seed{args.seed}'

    print(f'\n{"="*60}')
    print(f'  {run_name}')
    print(f'  Scale: {args.scale}, Attention: {args.attn}, Seed: {args.seed}')
    print(f'  Device: {device}')
    print(f'{"="*60}')

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Data
    data = prepare_data(cfg['seq_len'])
    # Use last 0.5% as validation (~45K sequences)
    train_ds = TokenDataset(data, 0.0, 0.995)
    val_ds = TokenDataset(data, 0.995, 1.0)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg['micro_batch'], shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg['micro_batch'], shuffle=False,
        num_workers=2, pin_memory=True)

    total_steps = len(train_ds) // (cfg['micro_batch'] * cfg['grad_accum'])
    print(f'  Train sequences: {len(train_ds):,}')
    print(f'  Val sequences: {len(val_ds):,}')
    print(f'  Total steps: {total_steps:,}')
    print(f'  Tokens per step: {cfg["micro_batch"] * cfg["grad_accum"] * cfg["seq_len"]:,}')

    # Model
    model = build_model(args.scale, args.attn, device)
    n_params = count_params(model)
    print(f'  Parameters: {n_params:,} ({n_params/1e6:.1f}M)')

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg['lr'], betas=(0.9, 0.95),
        weight_decay=0.1, fused=True)
    scaler = GradScaler()

    # Resume
    start_step, best_val_loss = load_checkpoint(run_name, model, optimizer, scaler, device)

    # Logger
    logger = Logger(run_name, use_wandb=args.wandb)
    logger.log({
        'event': 'start', 'params': n_params, 'scale': args.scale,
        'attn': args.attn, 'seed': args.seed, 'total_steps': total_steps,
    }, step=start_step)

    # Training loop
    model.train()
    grad_accum = cfg['grad_accum']
    step = start_step
    last_ckpt_time = time.time()
    t0 = time.time()
    running_loss = 0.0
    micro_steps = 0
    data_iter = iter(train_loader)

    # Handle graceful shutdown
    shutdown_requested = False
    def handle_signal(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True
        print('\nShutdown signal received, saving checkpoint...')
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    print(f'  Training from step {start_step}...\n')

    while step < total_steps and not shutdown_requested:
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for micro in range(grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x, y = x.to(device), y.to(device)
            with autocast(dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            step_loss += loss.item()
            micro_steps += 1

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # LR schedule
        lr = get_lr(step, cfg['warmup_steps'], cfg['lr'], cfg['min_lr'], total_steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        scaler.step(optimizer)
        scaler.update()
        step += 1
        running_loss += step_loss

        # Log every 50 steps
        if step % 50 == 0:
            elapsed_min = (time.time() - t0) / 60
            tokens_seen = micro_steps * cfg['micro_batch'] * cfg['seq_len']
            tok_per_sec = tokens_seen / (time.time() - t0)
            avg_loss = running_loss / 50
            data = {
                'train_loss': avg_loss, 'lr': lr,
                'tokens_per_sec': tok_per_sec, 'elapsed_min': elapsed_min,
            }
            logger.print_status(data, step, total_steps)
            logger.log(data, step)
            running_loss = 0.0

        # Evaluate every 1000 steps
        if step % 1000 == 0 or step == total_steps:
            val_loss, val_ppl = evaluate(model, val_loader, device)
            data = {'val_loss': val_loss, 'val_ppl': val_ppl}
            logger.log(data, step)
            print(f'  ** Validation: loss={val_loss:.4f}, ppl={val_ppl:.2f}', flush=True)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = CKPT_DIR / f'{run_name}_best.pt'
                torch.save({'step': step, 'model': model.state_dict(),
                            'val_loss': val_loss, 'val_ppl': val_ppl,
                            'params': n_params}, best_path)
                print(f'  ** New best model saved (ppl={val_ppl:.2f})')

        # Checkpoint every 30 minutes
        if time.time() - last_ckpt_time > 1800 or step == total_steps:
            save_checkpoint(model, optimizer, scaler, step, best_val_loss, run_name)
            last_ckpt_time = time.time()
            print(f'  ** Checkpoint saved at step {step}')

    # Final save
    save_checkpoint(model, optimizer, scaler, step, best_val_loss, run_name)
    val_loss, val_ppl = evaluate(model, val_loader, device)
    logger.log({'event': 'done', 'final_val_loss': val_loss, 'final_val_ppl': val_ppl,
                'total_elapsed_min': (time.time() - t0) / 60}, step=step)
    print(f'\n  Done: {run_name}, val_ppl={val_ppl:.2f}, '
          f'elapsed={(time.time()-t0)/3600:.1f}h\n')

    return val_ppl


# ============================================================
# Main
# ============================================================

def list_configs():
    print('\nAll experiment configurations:\n')
    for scale in ['125m', '350m']:
        cfg = CONFIGS[scale]
        for attn in ['standard', 'boosted', 'twicing', 'param_fair']:
            d = PARAM_FAIR_D[scale] if attn == 'param_fair' else cfg['d_model']
            label = f'{scale}_{attn}'
            print(f'  {label:<25s}  d={d}, layers={cfg["n_layers"]}, '
                  f'heads={cfg["n_heads"] if attn != "param_fair" else d // (cfg["d_model"] // cfg["n_heads"])}')
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on OpenWebText')
    parser.add_argument('--scale', choices=['125m', '350m'], default='125m')
    parser.add_argument('--attn', choices=['standard', 'boosted', 'twicing', 'param_fair'],
                        default='standard')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--list-configs', action='store_true')
    args = parser.parse_args()

    if args.list_configs:
        list_configs()
        sys.exit(0)

    train(args)
