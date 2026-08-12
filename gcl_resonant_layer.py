"""
GCL & Resonant Core Integration Module (PyTorch).
Fuses Global Context Layer (Schema / Global Picture + AdaptiveGate + SchemaReflector)
with Pisano Clock (pi(n)=24) and Resonant Micro-Networks Kn (6, 12, 18, 24) with T <= 7 convergence.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_pisano_sequence(n, length=24):
    """Generates Fibonacci sequence modulo n for a given length."""
    seq = []
    a, b = 0, 1
    for _ in range(length):
        seq.append(a)
        a, b = b, (a + b) % n
    return seq


class PisanoClock(nn.Module):
    """Fibonacci Pisano Clock Generator mod 9 (pi(n)=24)."""
    def __init__(self, d_model, resonant_sizes=[6, 12, 18, 24]):
        super().__init__()
        self.d_model = d_model
        self.resonant_sizes = resonant_sizes
        
        pisano_tensors = []
        for n in resonant_sizes:
            seq = get_pisano_sequence(n, length=24)
            seq_norm = [2.0 * (val / max(1, n - 1)) - 1.0 for val in seq]
            pisano_tensors.append(torch.tensor(seq_norm, dtype=torch.float32))
            
        self.register_buffer("pisano_phases", torch.stack(pisano_tensors))
        self.phase_proj = nn.Linear(len(resonant_sizes), d_model)

    def forward(self, batch_size, seq_len):
        device = self.pisano_phases.device
        positions = torch.arange(seq_len, device=device) % 24
        clock_phases = self.pisano_phases[:, positions]
        clock_flat = clock_phases.permute(1, 0)
        clock_emb = self.phase_proj(clock_flat)
        return clock_emb.unsqueeze(0).expand(batch_size, -1, -1)


class AdaptiveGate(nn.Module):
    """Adaptive Gate mechanism from GCL (0 to 1 scaling)."""
    def __init__(self, d_model):
        super().__init__()
        self.gate_linear = nn.Linear(d_model * 2, 1)

    def forward(self, x, schema_expanded):
        concat = torch.cat([x, schema_expanded], dim=-1)
        return torch.sigmoid(self.gate_linear(concat))


class SchemaReflector(nn.Module):
    """GCL Schema Reflector - transforms token details under global schema influence."""
    def __init__(self, d_model, schema_dim):
        super().__init__()
        self.proj_schema = nn.Linear(schema_dim, d_model)
        self.reflect = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x, schema):
        # schema: [B, schema_dim] -> [B, 1, d_model] -> [B, L, d_model]
        schema_emb = self.proj_schema(schema).unsqueeze(1).expand(-1, x.shape[1], -1)
        reflected = self.reflect(x + schema_emb)
        return reflected, schema_emb


class GCLResonantLayer(nn.Module):
    """
    Fused GCL & Resonant Core Layer.
    Pipeline:
      1. Global Schema Pooling (Global Context)
      2. Schema Reflector & Adaptive Gate
      3. Pisano Clock Injection (pi=24)
      4. Parallel Resonant Micro-Networks Kn (T <= 7 convergence loop)
    """
    def __init__(self, d_model, schema_dim=128, max_iter=7, tolerance=1e-4):
        super().__init__()
        self.d_model = d_model
        self.schema_dim = schema_dim
        self.max_iter = max_iter
        self.tolerance = tolerance
        
        # GCL components
        self.schema_pool = nn.Linear(d_model, schema_dim)
        self.reflector = SchemaReflector(d_model, schema_dim)
        self.gate = AdaptiveGate(d_model)
        
        # Pisano Clock
        self.clock = PisanoClock(d_model)
        
        # Resonant Kn Micro-Networks (6, 12, 18, 24)
        self.sizes = [6, 12, 18, 24]
        self.input_projs = nn.ModuleList([nn.Linear(d_model, n) for n in self.sizes])
        self.output_projs = nn.ModuleList([nn.Linear(n, d_model) for n in self.sizes])
        
        self.comm_weight = nn.Parameter(torch.tensor(0.3))
        self.norm = nn.LayerNorm(d_model)
        self.fusion = nn.Linear(d_model * 4, d_model)

    def forward(self, x, return_details=False):
        B, L, D = x.shape
        
        # Step 1: GCL Global Schema (Las -> Drzewa)
        schema = torch.mean(self.schema_pool(x), dim=1)  # [B, schema_dim]
        reflected_x, schema_expanded = self.reflector(x, schema)
        gate_val = self.gate(x, schema_expanded)
        
        # Enhanced GCL Representation
        gcl_x = x + gate_val * reflected_x
        
        # Step 2: Pisano Clock Injection
        clock_emb = self.clock(B, L)
        state = self.norm(gcl_x + clock_emb)
        
        prev_fused = None
        iters_taken = 0
        all_phase_states = []
        
        # Step 3: Resonant Micro-Network Convergence Loop (T <= 7)
        for t in range(self.max_iter):
            iters_taken += 1
            net_outputs = []
            current_phases = []
            
            for i, n in enumerate(self.sizes):
                nodes = F.gelu(self.input_projs[i](state))
                out = self.output_projs[i](nodes)
                net_outputs.append(out)
                current_phases.append(nodes)
                
            all_phase_states.append(current_phases)
            
            concat_out = torch.cat(net_outputs, dim=-1)
            fused = self.fusion(concat_out)
            
            w = torch.clamp(self.comm_weight, 0.0, 0.9)
            state = state * (1.0 - w) + fused * w
            
            if prev_fused is not None:
                diff = torch.mean(torch.abs(fused - prev_fused)).item()
                if diff < self.tolerance:
                    break
                    
            prev_fused = fused
            
        output = x + state
        
        if return_details:
            return output, iters_taken, all_phase_states, schema, gate_val
        return output
