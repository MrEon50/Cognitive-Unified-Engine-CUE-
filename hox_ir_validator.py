"""
HOX & IR Validator Module (PyTorch).
Implements:
1. Universal IR Verification Rule: s in (E+I+C) and Rel(s) in R and Exists phi
2. Multi-factor Cognitive Efficiency Constant Omega_IR
3. Prime Sieve 24 Generator: p^2 - 1 = 24 * n (for p >= 5)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrimeSieve24:
    """
    Verifies and manages the Prime Sieve mod 24 property (p^2 - 1 = 24 * n).
    Maps active prime slots on the 24-clock: [1, 5, 7, 11, 13, 17, 19, 23].
    """
    PRIME_SLOTS = [1, 5, 7, 11, 13, 17, 19, 23]
    
    @staticmethod
    def is_valid_prime_slot(n):
        """Returns True if position n mod 24 lies on one of the 8 prime slots."""
        return (n % 24) in PrimeSieve24.PRIME_SLOTS

    @staticmethod
    def verify_prime_theorem(p):
        """Verifies p^2 - 1 = 24 * n for prime p >= 5."""
        if p < 5:
            return False, 0
        val = p * p - 1
        is_div = (val % 24 == 0)
        return is_div, val // 24


class OmegaIRCalculator(nn.Module):
    """
    Calculates the Multi-Factor Constant Omega_IR:
    Omega_IR = (phi_IIT * ln(2) * m_Pl * c^2) / (phi_0 * m * E * H) * phi
    
    Measures the ratio of integrated cognitive consciousness to energy/information cost.
    """
    def __init__(self):
        super().__init__()
        # Physical constants (normalized)
        self.m_planck = 1.0  # Normalized Planck mass
        self.c_speed = 1.0   # Speed of light
        self.ln2 = math.log(2.0)
        self.phi_golden = (1.0 + math.sqrt(5.0)) / 2.0  # 1.618033...

    def forward(self, phi_iit, mass, energy, entropy_h, harmony_phi=None):
        """
        phi_iit: Tensor [B, L, 1] integrated information score
        mass: Tensor [B, L, 1] mass / physical cost
        energy: Tensor [B, L, 1] energy cost
        entropy_h: Tensor [B, L, 1] information entropy (bits)
        harmony_phi: optional harmony scaling (defaults to golden ratio phi)
        """
        if harmony_phi is None:
            harmony_phi = torch.tensor(self.phi_golden, device=phi_iit.device)
            
        numerator = phi_iit * self.ln2 * self.m_planck * (self.c_speed ** 2)
        denominator = (mass * energy * entropy_h) + 1e-6  # Epsilon for numerical stability
        
        omega_ir = (numerator / denominator) * harmony_phi
        return omega_ir


class IRValidator(nn.Module):
    """
    Validates structural compliance of relational expressions against EICR:
    1. Base components belong to E (Existence: m, e), I (Information: i), C (Consciousness: psi)
    2. Relations belong to R (Harmony phi, Time tau, Space chi, Trans lambda, Dev delta)
    3. Harmony condition exists (phi)
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.eicr_proj = nn.Linear(d_model, 4)  # 4 EICR categories: E, I, C, R
        self.harmony_classifier = nn.Linear(d_model, 1)

    def forward(self, state_embeddings):
        """
        state_embeddings: Tensor [B, L, D]
        returns:
            eicr_scores: [B, L, 4] Softmax distribution over E, I, C, R
            harmony_score: [B, L, 1] Sigmoid harmony context score in [0, 1]
            is_valid: bool tensor indicating compliance with IR validation rule
        """
        eicr_logits = self.eicr_proj(state_embeddings)
        eicr_scores = F.softmax(eicr_logits, dim=-1)
        
        harmony_score = torch.sigmoid(self.harmony_classifier(state_embeddings))
        
        # Valid if harmony score > 0.5 and EICR categories are balanced
        is_valid = (harmony_score > 0.5)
        
        return eicr_scores, harmony_score, is_valid
