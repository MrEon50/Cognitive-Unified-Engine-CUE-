"""
Cognitive Primitives & Relational Axioms Module (PyTorch).
Implements the 8 primitives of the Mind Alphabet (O, D, E, M, V, T, C, G)
and the fundamental recursion axiom: Relation R(A, B) becomes new Object O'.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CognitiveObject(nn.Module):
    """
    Represents a Cognitive State Object (O).
    Can be a single state vector or a batch of representations [B, L, D] or [B, D].
    """
    def __init__(self, state_tensor: torch.Tensor, name="CognitiveObject"):
        super().__init__()
        self.name = name
        self.register_buffer("state", state_tensor)

    @property
    def shape(self):
        return self.state.shape

    def __repr__(self):
        return f"CognitiveObject(name='{self.name}', shape={list(self.state.shape)})"


class CognitiveRelation(CognitiveObject):
    """
    Fundament Aksjomatu Rekurencji:
    Relacja R(A, B) w momencie powstania staje się nowym Obiektem O'.
    """
    def __init__(self, obj_a: CognitiveObject, obj_b: CognitiveObject, rel_type="Relation", projection: nn.Module = None):
        state_a = obj_a.state
        state_b = obj_b.state
        
        # Relational synthesis: average + difference interaction
        if projection is not None:
            concat = torch.cat([state_a, state_b], dim=-1)
            combined_state = projection(concat)
        else:
            combined_state = (state_a + state_b) / 2.0
            
        rel_name = f"{rel_type}({obj_a.name}, {obj_b.name})"
        super().__init__(combined_state, name=rel_name)
        self.source = obj_a
        self.target = obj_b
        self.rel_type = rel_type


class CognitiveRelationLayer(nn.Module):
    """
    Differentiable Neural Synthesizer for the Axiom of Recursion: R(A, B) -> O'.
    Maps two cognitive states into a newly emergent cognitive object representation.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.synth = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, state_a: torch.Tensor, state_b: torch.Tensor) -> torch.Tensor:
        """
        Synthesizes relation R(A, B) into a new state O'.
        state_a, state_b: [B, D] or [B, L, D]
        returns: O' [B, D] or [B, L, D]
        """
        concat = torch.cat([state_a, state_b], dim=-1)
        rel_state = self.synth(concat)
        return self.norm((state_a + state_b) * 0.5 + rel_state)


class CognitivePalette(nn.Module):
    """
    Module implementing the 8 Cognitive Primitives of the Mind Alphabet (PyTorch vector operations):
    1. O - Obiekt/Stan (State definition)
    2. D - Delta (Difference vector)
    3. E - Equivalence/Symmetry (Similarity measure)
    4. M - Metric (Euclidean distance measure with safe epsilon)
    5. V - Vector (Direction/Gradient of change)
    6. T - Transformation (State transition in time tau)
    7. C - Causality (Causal link strength)
    8. G - Goal Gravity (Distance minimization / pull to goal G)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.causal_weight = nn.Linear(d_model * 2, 1)
        self.relation_synth = CognitiveRelationLayer(d_model)

    def difference_D(self, state_a: torch.Tensor, state_b: torch.Tensor) -> torch.Tensor:
        """D (Delta): Computes vector difference state_b - state_a."""
        return state_b - state_a

    def metric_M(self, state_a: torch.Tensor, state_b: torch.Tensor) -> torch.Tensor:
        """M (Metric): Computes numerically safe Euclidean distance between states."""
        diff = state_b - state_a
        return torch.sqrt(torch.sum(diff ** 2, dim=-1, keepdim=True) + 1e-8)

    def equivalence_E(self, state_a: torch.Tensor, state_b: torch.Tensor) -> torch.Tensor:
        """E (Equivalence/Symmetry): Exponential similarity in (0, 1]."""
        dist = self.metric_M(state_a, state_b)
        return torch.exp(-torch.clamp(torch.abs(self.gamma), min=0.01, max=10.0) * dist)

    def direction_V(self, state_a: torch.Tensor, state_b: torch.Tensor) -> torch.Tensor:
        """V (Vector): Safe directional unit vector from state_a to state_b."""
        diff = self.difference_D(state_a, state_b)
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1, keepdim=True) + 1e-8)
        return diff / dist

    def transform_T(self, state: torch.Tensor, action_vector: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """T (Transformation): State transition over time step tau."""
        return state + action_vector * tau

    def causality_C(self, state_a: torch.Tensor, state_b: torch.Tensor) -> torch.Tensor:
        """C (Causality): Evaluates causal link score between states in [0, 1]."""
        concat = torch.cat([state_a, state_b], dim=-1)
        return torch.sigmoid(self.causal_weight(concat))

    def goal_gravity_G(self, current_state: torch.Tensor, goal_state: torch.Tensor) -> torch.Tensor:
        """G (Goal Gravity): Computes attraction vector pulling towards goal."""
        return self.direction_V(current_state, goal_state) * self.metric_M(current_state, goal_state)

    def synthesize_relation_O_prime(self, state_a: torch.Tensor, state_b: torch.Tensor) -> torch.Tensor:
        """Axiom of Recursion R(A, B) -> O'."""
        return self.relation_synth(state_a, state_b)

    def forward(self, state_a: torch.Tensor, state_b: torch.Tensor, goal_state: torch.Tensor = None) -> dict:
        """
        Executes complete Cognitive Palette evaluation on a pair of states.
        Returns dictionary of cognitive primitive representations.
        """
        diff = self.difference_D(state_a, state_b)
        dist = self.metric_M(state_a, state_b)
        sim = self.equivalence_E(state_a, state_b)
        vec = self.direction_V(state_a, state_b)
        causal = self.causality_C(state_a, state_b)
        o_prime = self.synthesize_relation_O_prime(state_a, state_b)
        
        results = {
            "D_diff": diff,
            "M_dist": dist,
            "E_similarity": sim,
            "V_direction": vec,
            "C_causality": causal,
            "O_prime": o_prime
        }
        
        if goal_state is not None:
            results["G_gravity"] = self.goal_gravity_G(state_b, goal_state)
            
        return results
