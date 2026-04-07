"""
Post-training evaluation of scaled models.

Evaluates:
  1. WikiText-103 test perplexity
  2. Zero-shot benchmarks via lm-evaluation-harness (HellaSwag, PIQA, LAMBADA, ARC-E)

Usage:
  python eval_benchmarks.py --checkpoint /workspace/checkpoints/125m_boosted_seed42_best.pt \
                            --scale 125m --attn boosted

  # Evaluate all checkpoints in a directory
  python eval_benchmarks.py --eval-all --scale 125m
"""

import os, sys, json, math, argparse, glob
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

sys.path.insert(0, os.path.dirname(__file__))
from train_openwebtext import TransformerLM, CONFIGS, PARAM_FAIR_D, VOCAB_SIZE, build_model

CKPT_DIR = Path(os.environ.get('CKPT_DIR', '/workspace/checkpoints'))
RESULTS_DIR = Path(os.environ.get('RESULTS_DIR', '/workspace/results'))


def load_model_from_checkpoint(ckpt_path, scale, attn, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(scale, attn, device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Loaded {ckpt_path}, params={n_params:,}')
    if 'val_ppl' in ckpt:
        print(f'  Training val_ppl={ckpt["val_ppl"]:.2f} at step {ckpt.get("step", "?")}')
    return model


# ============================================================
# WikiText-103 perplexity
# ============================================================

@torch.no_grad()
def eval_wikitext103(model, device, seq_len=1024):
    """Evaluate perplexity on WikiText-103 test set."""
    print('\n  Evaluating WikiText-103 test perplexity...')
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')

    # Tokenize and concatenate
    all_tokens = []
    for doc in ds:
        text = doc['text'].strip()
        if text:
            all_tokens.extend(tokenizer.encode(text))

    print(f'  Test tokens: {len(all_tokens):,}')

    # Evaluate with sliding window
    total_loss = 0.0
    total_tokens = 0
    stride = seq_len // 2  # 50% overlap

    for start in range(0, len(all_tokens) - seq_len, stride):
        chunk = all_tokens[start:start + seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(chunk[1:], dtype=torch.long, device=device).unsqueeze(0)

        with autocast(dtype=torch.bfloat16):
            logits = model(x)
            # Only count loss for the second half (non-overlapping part)
            offset = stride if start > 0 else 0
            loss = F.cross_entropy(
                logits[:, offset:].reshape(-1, logits.size(-1)),
                y[:, offset:].reshape(-1),
                reduction='sum')

        count = y[:, offset:].numel()
        total_loss += loss.item()
        total_tokens += count

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    print(f'  WikiText-103 test: loss={avg_loss:.4f}, ppl={ppl:.2f}')
    return {'wt103_loss': avg_loss, 'wt103_ppl': ppl}


# ============================================================
# lm-evaluation-harness benchmarks
# ============================================================

def eval_lm_harness(model, device, scale, attn, tasks='hellaswag,piqa,lambada_openai,arc_easy'):
    """Run zero-shot evaluation using lm-evaluation-harness."""
    print(f'\n  Running lm-eval-harness: {tasks}')

    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print('  lm-evaluation-harness not installed. Skipping.')
        print('  Install with: pip install lm-eval')
        return {}

    # Wrap our model for lm-eval
    # lm-eval expects a HuggingFace-compatible model, so we use a wrapper
    from transformers import GPT2TokenizerFast

    class LMEvalWrapper(torch.nn.Module):
        """Minimal wrapper to make our model compatible with lm-eval."""
        def __init__(self, model, tokenizer, device):
            super().__init__()
            self.model = model
            self.tokenizer = tokenizer
            self.device = device

        @torch.no_grad()
        def generate(self, *args, **kwargs):
            raise NotImplementedError

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    seq_len = CONFIGS[scale]['seq_len']

    # Save model temporarily in HF format for lm-eval
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write a simple eval script instead
        # lm-eval has a simple_evaluate API
        results = lm_eval.simple_evaluate(
            model='hf',
            model_args=f'pretrained={tmpdir}',
            tasks=tasks.split(','),
            batch_size=8,
        )

    # If the HF wrapper approach doesn't work, fall back to manual eval
    print(f'  lm-eval results: {json.dumps(results.get("results", {}), indent=2)}')
    return results.get('results', {})


# ============================================================
# Manual zero-shot evaluations (fallback if lm-eval is tricky)
# ============================================================

@torch.no_grad()
def eval_lambada(model, device, seq_len=1024):
    """LAMBADA last-word accuracy."""
    print('\n  Evaluating LAMBADA accuracy...')
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    ds = load_dataset('lambada', split='test')

    correct = 0
    total = 0

    for item in ds:
        text = item['text']
        tokens = tokenizer.encode(text)
        if len(tokens) > seq_len:
            tokens = tokens[-seq_len:]

        # Last word
        last_word = text.split()[-1]
        last_word_tokens = tokenizer.encode(' ' + last_word)

        x = torch.tensor(tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)
        with autocast(dtype=torch.bfloat16):
            logits = model(x)

        # Check if model predicts the last token correctly
        pred = logits[0, -1].argmax().item()
        if pred == tokens[-1]:
            correct += 1
        total += 1

    acc = correct / total * 100
    print(f'  LAMBADA: accuracy={acc:.2f}% ({correct}/{total})')
    return {'lambada_acc': acc}


# ============================================================
# Main
# ============================================================

def eval_single(ckpt_path, scale, attn, device):
    model = load_model_from_checkpoint(ckpt_path, scale, attn, device)

    results = {'checkpoint': str(ckpt_path), 'scale': scale, 'attn': attn}

    # WikiText-103
    wt103 = eval_wikitext103(model, device, CONFIGS[scale]['seq_len'])
    results.update(wt103)

    # LAMBADA
    lambada = eval_lambada(model, device, CONFIGS[scale]['seq_len'])
    results.update(lambada)

    return results


def eval_all(scale, device):
    """Evaluate all best checkpoints for a given scale."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for attn in ['standard', 'boosted', 'twicing', 'param_fair']:
        pattern = f'{scale}_{attn}_seed*_best.pt'
        ckpts = sorted(glob.glob(str(CKPT_DIR / pattern)))
        for ckpt_path in ckpts:
            seed = ckpt_path.split('seed')[1].split('_')[0]
            print(f'\n{"="*60}')
            print(f'  Evaluating: {scale}/{attn}/seed{seed}')
            print(f'{"="*60}')

            results = eval_single(ckpt_path, scale, attn, device)
            results['seed'] = int(seed)
            all_results.append(results)

    # Save
    out_path = RESULTS_DIR / f'eval_{scale}.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to {out_path}')

    # Summary table
    print(f'\n{"="*60}')
    print(f'  Summary: {scale}')
    print(f'{"="*60}')
    print(f'  {"Config":<25s} {"WT-103 PPL":>12s} {"LAMBADA":>10s}')
    print(f'  {"-"*50}')
    for r in all_results:
        label = f'{r["attn"]}/seed{r["seed"]}'
        ppl = r.get('wt103_ppl', float('nan'))
        lam = r.get('lambada_acc', float('nan'))
        print(f'  {label:<25s} {ppl:>12.2f} {lam:>9.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Single checkpoint to evaluate')
    parser.add_argument('--scale', choices=['125m', '350m'], required=True)
    parser.add_argument('--attn', choices=['standard', 'boosted', 'twicing', 'param_fair'])
    parser.add_argument('--eval-all', action='store_true', help='Evaluate all checkpoints for scale')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.eval_all:
        eval_all(args.scale, device)
    elif args.checkpoint:
        if not args.attn:
            parser.error('--attn required when evaluating a single checkpoint')
        results = eval_single(args.checkpoint, args.scale, args.attn, device)
        print(json.dumps(results, indent=2))
    else:
        parser.error('Specify --checkpoint or --eval-all')
