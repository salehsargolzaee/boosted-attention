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
