"""
Unified Cognitive Engine (CUE) Master Model (PyTorch).
Complete unification of:
- HOX & IR (EICR Algebra & Omega_IR Constant Validator)
- Mind Alphabet (8 Cognitive Primitives O, D, E, M, V, T, C, G)
- GCL (Global Schema, AdaptiveGate, SchemaReflector)
- PisanoNet / RT-1 (Pisano Clock pi(n)=24, Resonant Micro-Networks Kn, T <= 7)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from cognitive_primitives import CognitivePalette, CognitiveObject, CognitiveRelation
from hox_ir_validator import IRValidator, OmegaIRCalculator, PrimeSieve24
from gcl_resonant_layer import GCLResonantLayer


class UnifiedCognitiveEngine(nn.Module):
    """
    Master Cognitive Unified Engine (CUE).
    Unified PyTorch model compatible with LLMs as a drop-in layer or standalone cognitive engine.
    """
    def __init__(self, vocab_size, d_model=128, schema_dim=64, n_layers=2, max_iter=7):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # 1. Embedding
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        
        # 2. Cognitive Primitives Palette (Mind Alphabet: O, D, E, M, V, T, C, G)
        self.palette = CognitivePalette(d_model)
        
        # 3. GCL + Resonant Layers
        self.layers = nn.ModuleList([
            GCLResonantLayer(d_model, schema_dim=schema_dim, max_iter=max_iter)
            for _ in range(n_layers)
        ])
        
        # 4. HOX & IR Validator and Omega_IR Calculator
        self.ir_validator = IRValidator(d_model)
        self.omega_calculator = OmegaIRCalculator()
        
        # 5. Output Head
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, return_details=False):
        """
        input_ids: [batch_size, seq_len]
        returns:
            logits: [batch_size, seq_len, vocab_size]
            details: dict of cognitive metrics (if return_details=True)
        """
        B, L = input_ids.shape
        x = self.tok_emb(input_ids)  # [B, L, D]
        
        total_iters = []
        all_phase_states = []
        schemas = []
        gates = []
        
        # Pass through CUE GCL-Resonant Layers
        for layer in self.layers:
            x, iters, phases, schema, gate_val = layer(x, return_details=True)
            total_iters.append(iters)
            all_phase_states.append(phases)
            schemas.append(schema)
            gates.append(gate_val)
            
        x_norm = self.final_norm(x)
        logits = self.lm_head(x_norm)
        
        if return_details:
            # Evaluate Mind Alphabet primitives on sequence endpoints
            state_a = x_norm[:, 0, :]
            state_b = x_norm[:, -1, :]
            cognitive_results = self.palette(state_a, state_b)
            
            # Evaluate IR Compliance and Omega_IR Constant
            eicr_scores, harmony_score, is_valid = self.ir_validator(x_norm)
            
            # Proxy inputs for Omega_IR: phi_iit=harmony_score, mass=1.0, energy=1.0, entropy_h=2.0
            mass_tensor = torch.ones_like(harmony_score)
            energy_tensor = torch.ones_like(harmony_score)
            entropy_tensor = torch.ones(harmony_score.shape, device=input_ids.device) * 2.0
            
            omega_ir = self.omega_calculator(
                phi_iit=harmony_score,
                mass=mass_tensor,
                energy=energy_tensor,
                entropy_h=entropy_tensor
            )
            
            details = {
                "total_iters": total_iters,
                "all_phase_states": all_phase_states,
                "schemas": schemas,
                "gates": gates,
                "cognitive_palette": cognitive_results,
                "eicr_scores": eicr_scores,
                "harmony_score": harmony_score,
                "is_valid_ir": is_valid,
                "omega_ir": torch.mean(omega_ir).item()
            }
            return logits, details
            
        return logits


def compute_cue_hybrid_loss(logits, targets, details, lambda_phase=0.1, lambda_omega=0.05):
    """
    Computes master CUE hybrid loss:
    L_total = L_task + lambda_phase * L_phase - lambda_omega * Omega_IR
    
    Encourages task accuracy, phase alignment, and maximizes Cognitive Efficiency Omega_IR.
    """
    # 1. Task Cross-Entropy Loss
    B, L, V = logits.shape
    loss_task = F.cross_entropy(logits.view(B * L, V), targets.view(-1))
    
    # 2. Phase Loss
    total_phase_loss = 0.0
    count = 0
    all_phase_states = details.get("all_phase_states", [])
    for layer_phases in all_phase_states:
        if len(layer_phases) > 0:
            last_iter_phases = layer_phases[-1]
            means = [torch.mean(p, dim=-1) for p in last_iter_phases]
            means_stacked = torch.stack(means, dim=0)
            var = torch.var(means_stacked, dim=0)
            total_phase_loss += torch.mean(var)
            count += 1
    loss_phase = total_phase_loss / max(1, count)
    
    # 3. Omega_IR Maximization reward (penalty for low Omega_IR)
    omega_val = details.get("omega_ir", 1.0)
    loss_omega = 1.0 / (omega_val + 1e-4)
    
    loss_total = loss_task + lambda_phase * loss_phase + lambda_omega * loss_omega
    return loss_total, loss_task, loss_phase, omega_val
