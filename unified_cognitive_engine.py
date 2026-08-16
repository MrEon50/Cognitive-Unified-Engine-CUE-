"""
Unified Cognitive Engine (CUE) Master Model (PyTorch).
Complete unification of:
- HOX & IR (EICR Algebra & Differentiable Omega_IR Constant Validator)
- Mind Alphabet (8 Cognitive Primitives O, D, E, M, V, T, C, G)
- GCL (Global Schema, AdaptiveGate, SchemaReflector, SchemaMemory)
- PisanoNet / RT-1 (Pisano Clock pi(n)=24, Resonant Kn Micro-Networks, T <= 7)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from cognitive_primitives import CognitivePalette, CognitiveObject, CognitiveRelation, CognitiveRelationLayer
from hox_ir_validator import IRValidator, OmegaIRCalculator, PrimeSieve24
from gcl_resonant_layer import GCLResonantLayer


class UnifiedCognitiveEngine(nn.Module):
    """
    Master Cognitive Unified Engine (CUE).
    Unified PyTorch model compatible with LLMs as a drop-in layer or standalone cognitive engine.
    """
    def __init__(self, vocab_size: int, d_model: int = 128, schema_dim: int = 64, n_layers: int = 2, max_iter: int = 7, causal: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.causal = causal
        
        # 1. Embedding
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        
        # 2. Cognitive Primitives Palette (Mind Alphabet: O, D, E, M, V, T, C, G)
        self.palette = CognitivePalette(d_model)
        
        # 3. GCL + Resonant Layers (Complete Graph Kn, T <= 7)
        self.layers = nn.ModuleList([
            GCLResonantLayer(d_model, schema_dim=schema_dim, max_iter=max_iter, causal=causal)
            for _ in range(n_layers)
        ])
        
        # 4. HOX & IR Validator and Omega_IR Calculator
        self.ir_validator = IRValidator(d_model)
        self.omega_calculator = OmegaIRCalculator()
        
        # 5. Output Head
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, return_details: bool = False):
        """
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len] optional mask
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
            if return_details:
                x, layer_details = layer(x, attention_mask=attention_mask, return_details=True)
                total_iters.append(layer_details["iters"])
                all_phase_states.append(layer_details["phases"])
                schemas.append(layer_details["schema"])
                gates.append(layer_details["gate"])
            else:
                x = layer(x, attention_mask=attention_mask, return_details=False)
            
        x_norm = self.final_norm(x)
        logits = self.lm_head(x_norm)
        
        if return_details:
            # 1. Evaluate Mind Alphabet primitives on sequence endpoints
            state_a = x_norm[:, 0, :]
            state_b = x_norm[:, -1, :]
            cognitive_results = self.palette(state_a, state_b)
            
            # 2. Evaluate IR Compliance and Differentiable Omega_IR Constant
            eicr_scores, harmony_score, is_valid = self.ir_validator(x_norm)
            
            # Proxy inputs for Omega_IR: phi_iit=harmony_score, mass=1.0, energy=1.0, entropy_h=2.0
            mass_tensor = torch.ones_like(harmony_score)
            energy_tensor = torch.ones_like(harmony_score)
            entropy_tensor = torch.ones(harmony_score.shape, device=input_ids.device, dtype=harmony_score.dtype) * 2.0
            
            omega_ir_tensor = self.omega_calculator(
                phi_iit=harmony_score,
                mass=mass_tensor,
                energy=energy_tensor,
                entropy_h=entropy_tensor
            )
            
            mean_omega_tensor = torch.mean(omega_ir_tensor)
            
            details = {
                "total_iters": total_iters,
                "all_phase_states": all_phase_states,
                "schemas": schemas,
                "gates": gates,
                "cognitive_palette": cognitive_results,
                "eicr_scores": eicr_scores,
                "harmony_score": harmony_score,
                "is_valid_ir": is_valid,
                "omega_ir": mean_omega_tensor.item(),
                "omega_ir_tensor": mean_omega_tensor
            }
            return logits, details
            
        return logits


def compute_cue_hybrid_loss(logits: torch.Tensor, targets: torch.Tensor, details: dict, lambda_phase: float = 0.1, lambda_omega: float = 0.01, lambda_cog: float = 0.01):
    """
    Computes master CUE hybrid loss:
    L_total = L_task + lambda_phase * L_phase + lambda_omega * L_omega + lambda_cog * L_cog
    
    Fully differentiable: gradients backpropagate through all model submodules including
    GCL layers, Resonant Micro-networks Kn, IR Validator, Omega Calculator, and Cognitive Palette.
    """
    # 1. Task Cross-Entropy Loss
    B, L, V = logits.shape
    loss_task = F.cross_entropy(logits.view(B * L, V), targets.view(-1))
    
    # 2. Phase Synchronization Loss (Minimizes variance across 4 resonant micro-networks)
    total_phase_loss = torch.tensor(0.0, device=logits.device)
    count = 0
    all_phase_states = details.get("all_phase_states", [])
    for layer_phases in all_phase_states:
        if len(layer_phases) > 0:
            last_iter_phases = layer_phases[-1]
            means = [torch.mean(p, dim=-1) for p in last_iter_phases]
            means_stacked = torch.stack(means, dim=0)  # [4, B, L]
            var = torch.var(means_stacked, dim=0)
            total_phase_loss = total_phase_loss + torch.mean(var)
            count += 1
            
    loss_phase = total_phase_loss / max(1, count) if count > 0 else torch.tensor(0.0, device=logits.device)
    
    # 3. Omega_IR Differentiable Optimization Loss
    if "omega_ir_tensor" in details:
        omega_tensor = details["omega_ir_tensor"]
        loss_omega = 1.0 / (omega_tensor + 1e-4)
        omega_val = omega_tensor.item()
    else:
        omega_val = details.get("omega_ir", 1.0)
        loss_omega = torch.tensor(1.0 / (omega_val + 1e-4), device=logits.device)
        
    # 4. Cognitive Palette Regularization (Trains Palette primitives & Relation Synthesis)
    loss_cog = torch.tensor(0.0, device=logits.device)
    if "cognitive_palette" in details:
        cog = details["cognitive_palette"]
        causal = cog.get("C_causality", None)
        similarity = cog.get("E_similarity", None)
        o_prime = cog.get("O_prime", None)
        diff = cog.get("D_diff", None)
        
        cog_terms = []
        if causal is not None and similarity is not None:
            cog_terms.append(-torch.mean(torch.log(causal + 1e-6) + torch.log(similarity + 1e-6)) * 0.1)
        if o_prime is not None and diff is not None:
            # Cognitive coherence: relational state O' captures non-trivial delta dynamics
            rel_norm = torch.mean(torch.norm(o_prime, dim=-1))
            cog_terms.append(rel_norm * 0.01)
            
        if cog_terms:
            loss_cog = sum(cog_terms)
            
    loss_total = loss_task + lambda_phase * loss_phase + lambda_omega * loss_omega + lambda_cog * loss_cog
    return loss_total, loss_task, loss_phase, omega_val
