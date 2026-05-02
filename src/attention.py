"""
Boosted Attention: Gradient Boosting within a Single Attention Layer

Architecture:
  Round 0: prediction = attention_0(query, keys, values)
  Round 1: residual_1 = query - prediction
            correction_1 = attention_1(residual_1, keys, values)
  Round 2: residual_2 = query - prediction - correction_1
            correction_2 = attention_2(residual_2, keys, values)
  ...
  Output = prediction + gate_1 * correction_1 + gate_2 * correction_2 + ...

Each round has separate learned projections (W_q, W_k, W_v).
Gates are learned per-round, conditioned on the current state and uncertainty.

This is Friedman's gradient boosting (MART, 2001) with attention as the
base learner, applied within a single layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class BoostedAttentionOutput:
    output: torch.Tensor              # (B, d) final merged output
    round_outputs: List[torch.Tensor]  # per-round outputs
    residuals: List[torch.Tensor]      # per-round residuals
    gates: List[torch.Tensor]          # per-round gate values
    weights: List[torch.Tensor]        # per-round attention weights
    entropies: List[torch.Tensor]      # per-round entropies


class AttentionRound(nn.Module):
    """One round of attention (one boosting stage / one weak learner)."""

    def __init__(self, d_in: int, d_key: int, d_out: int, beta_init: float = 2.0):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_key, bias=False)
        self.W_k = nn.Linear(d_in, d_key, bias=False)
        self.W_v = nn.Linear(d_in, d_out, bias=False)
        self.log_beta = nn.Parameter(torch.tensor(float(beta_init)).log())

    @property
    def beta(self):
        return self.log_beta.exp()

    def forward(self, query, keys, values):
        """
        Args:
            query: (B, d_in) — original query (round 0) or residual (round 1+)
            keys: (K, d_in) or (B, K, d_in) — stored patterns
            values: (K, d_in) or (B, K, d_in) — stored patterns (may equal keys)
        Returns:
            output: (B, d_out), weights: (B, K), entropy: (B,)
        """
        q = self.W_q(query)
        k = self.W_k(keys)
        v = self.W_v(values)

        if k.dim() == 3:
            logits = self.beta * torch.bmm(k, q.unsqueeze(-1)).squeeze(-1)
        else:
            logits = self.beta * (q @ k.T)

        weights = F.softmax(logits, dim=-1)

        if v.dim() == 3:
            output = torch.bmm(weights.unsqueeze(1), v).squeeze(1)
        else:
            output = weights @ v

        entropy = -(weights * (weights + 1e-10).log()).sum(dim=-1)
        return output, weights, entropy


class BoostedAttention(nn.Module):
    """
    K-round boosted attention layer.

    Round 0: Fast retrieval (standard attention)
    Round 1..K-1: Error correction (each attends to cumulative residual)

    Each round has its own projections. A learned gate controls
    how much each correction contributes.
    """

    def __init__(
        self,
        d_model: int,
        n_rounds: int = 2,
        d_key: Optional[int] = None,
        beta_init: float = 2.0,
        gate_type: str = "mlp",  # "mlp", "scalar", "none"
        gate_hidden: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_rounds = n_rounds
        d_key = d_key or d_model

        # One attention module per boosting round
        self.rounds = nn.ModuleList([
            AttentionRound(d_model, d_key, d_model, beta_init)
            for _ in range(n_rounds)
        ])

        # Gates for rounds 1+ (round 0 is the base prediction, no gate)
        self.gate_type = gate_type
        if gate_type == "mlp":
            self.gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2 * d_model + 1, gate_hidden),
                    nn.GELU(),
                    nn.Linear(gate_hidden, d_model),
                    nn.Sigmoid(),
                )
                for _ in range(n_rounds - 1)
            ])
        elif gate_type == "scalar":
            # One learned scalar per round (like the shrinkage parameter in MART)
            self.gate_params = nn.ParameterList([
                nn.Parameter(torch.tensor(0.5))
                for _ in range(n_rounds - 1)
            ])
        # gate_type == "none": fixed weight of 1.0 (pure additive boosting)

    def forward(self, query, keys, values=None, return_details=False):
        """
        Args:
            query: (B, d_model)
            keys: (K, d_model) or (B, K, d_model)
            values: optional, defaults to keys
        """
        if values is None:
            values = keys

        round_outputs = []
        residuals = []
        gates_out = []
        weights_out = []
        entropies_out = []

        # Round 0: base prediction
        pred, w0, ent0 = self.rounds[0](query, keys, values)
        round_outputs.append(pred)
        weights_out.append(w0)
        entropies_out.append(ent0)

        cumulative = pred
        output = pred.clone()

        # Rounds 1..K-1: corrections on cumulative residual
        for i in range(1, self.n_rounds):
            residual = query - cumulative
            residuals.append(residual)

            correction, wi, enti = self.rounds[i](residual, keys, values)
            round_outputs.append(correction)
            weights_out.append(wi)
            entropies_out.append(enti)

            # Compute gate
            if self.gate_type == "mlp":
                gate_input = torch.cat([
                    cumulative,
                    correction,
                    entropies_out[0].unsqueeze(-1),  # uncertainty from round 0
                ], dim=-1)
                gate = self.gates[i - 1](gate_input)
            elif self.gate_type == "scalar":
                gate = torch.sigmoid(self.gate_params[i - 1]).expand_as(correction)
            else:  # "none"
                gate = torch.ones_like(correction)

            gates_out.append(gate)

            gated_correction = gate * correction
            output = output + gated_correction
            cumulative = cumulative + gated_correction

        if return_details:
            return BoostedAttentionOutput(
                output=output,
                round_outputs=round_outputs,
                residuals=residuals,
                gates=gates_out,
                weights=weights_out,
                entropies=entropies_out,
            )
        return output, weights_out[0], entropies_out[0]


class StandardAttention(nn.Module):
    """Baseline: single-round attention (equivalent to BoostedAttention with n_rounds=1)."""

    def __init__(self, d_model, d_key=None, beta_init=2.0):
        super().__init__()
        self.attn = AttentionRound(d_model, d_key or d_model, d_model, beta_init)

    def forward(self, query, keys, values=None, return_details=False):
        if values is None:
            values = keys
        output, weights, entropy = self.attn(query, keys, values)
        return output, weights, entropy


# ============================================================
# Causal Multi-Head Variants (for language modeling)
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
    """Gradient-boosted causal attention: M rounds with separate QKV and learned gate.

    Round 0: Q, K, V all from x.
    Round 1+: Q, K, V all from residual (default, kv_source='residual')
              or Q from residual, K/V from x (kv_source='input').
    """
    def __init__(self, d_model, n_heads, n_rounds=2, dropout=0.1,
                 kv_source='residual'):
        super().__init__()
        self.n_heads = n_heads
        self.n_rounds = n_rounds
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.kv_source = kv_source
        self.scale = self.d_head ** -0.5
        self.W_qkvs = nn.ModuleList([nn.Linear(d_model, 3 * d_model)
                                     for _ in range(n_rounds)])
        self.W_out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())
            for _ in range(n_rounds - 1)
        ])
        self.capturing = False
        self._cached = {}

    def _attend(self, x_q, x_kv, round_idx):
        B, T, D = x_q.shape
        W = self.W_qkvs[round_idx]
        q = F.linear(x_q, W.weight[:D], W.bias[:D]).reshape(
            B, T, self.n_heads, self.d_head).transpose(1, 2)
        kv = F.linear(x_kv, W.weight[D:], W.bias[D:]).reshape(
            B, T, 2, self.n_heads, self.d_head)
        k, v = kv.unbind(dim=2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x_q.device), diagonal=1).bool()
        attn.masked_fill_(mask, float('-inf'))
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        if self.capturing:
            self._cached['attn'].append(attn.detach())
            self._cached['v'].append(v.detach())
        return out

    def enable_capture(self):
        self.capturing = True
        self._cached = {'attn': [], 'v': [], 'gate': [], 'pred_r0': None,
                        'output_pre_proj': None}

    def disable_capture(self):
        self.capturing = False
        self._cached = {}

    def forward(self, x):
        pred = self._attend(x, x, 0)
        if self.capturing:
            self._cached['pred_r0'] = pred.detach()
        output = pred
        cumulative = pred
        for i in range(1, self.n_rounds):
            residual = x - cumulative
            x_kv = residual if self.kv_source == 'residual' else x
            correction = self._attend(residual, x_kv, i)
            gate = self.gates[i - 1](torch.cat([cumulative, correction], dim=-1))
            if self.capturing:
                self._cached['gate'].append(gate.detach())
            gated = gate * correction
            output = output + gated
            cumulative = cumulative + gated
        if self.capturing:
            self._cached['output_pre_proj'] = output.detach()
        return self.W_out(output)


class TwicingCausalAttention(nn.Module):
    """Twicing Attention (Abdullaev & Nguyen, ICLR 2025).

    Output = (2A - A^2)V = AV + A(V - AV).
    Uses the SAME attention matrix A for both passes.
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
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn.masked_fill_(mask, float('-inf'))
        A = self.attn_drop(F.softmax(attn, dim=-1))
        Av = A @ v
        residual = v - Av
        correction = A @ residual
        out = (Av + correction).transpose(1, 2).reshape(B, T, D)
        return self.W_out(out)
