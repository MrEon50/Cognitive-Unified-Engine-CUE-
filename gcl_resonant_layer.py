"""
GCL (Global Context Layer) v3.0 & Resonant Core Integration Module (PyTorch).
Fuses:
1. Attentive Salience Pooling (Multi-Head K-slot context weighting, avoiding uniform semantic dilution)
2. Multi-Aspect Slot Schema (Semantics, Structure, Goal, Harmony)
3. FiLM (Feature-wise Linear Modulation: (1 + tanh(gamma)) * x + beta) & Schema Reflector
4. Adaptive Gate with negative-bias warmup (start sigmoid(-2) ~ 0.11)
5. SchemaMemory (Working memory with EMA update for autoregressive LLM generation)
6. Pisano Clock (pi(n)=24) & Complete Graph Kn Micro-Networks (6, 12, 18, 24) with T <= 7 convergence.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_pisano_sequence(n: int, length: int = 24) -> list:
    """Generates Fibonacci sequence modulo n for a given length."""
    seq = []
    a, b = 0, 1
    for _ in range(length):
        seq.append(a)
        a, b = b, (a + b) % n
    return seq


class PisanoClock(nn.Module):
    """
    Fibonacci Pisano Clock Generator mod 9 and resonant sizes (pi(n)=24).
    Provides natural cyclic time substrate without artificial RoPE / Positional Embeddings.
    """
    def __init__(self, d_model: int, resonant_sizes: list = None):
        super().__init__()
        if resonant_sizes is None:
            resonant_sizes = [6, 12, 18, 24]
        self.d_model = d_model
        self.resonant_sizes = resonant_sizes
        
        pisano_tensors = []
        for n in resonant_sizes:
            seq = get_pisano_sequence(n, length=24)
            # Normalize to [-1.0, 1.0] range
            seq_norm = [2.0 * (val / max(1, n - 1)) - 1.0 for val in seq]
            pisano_tensors.append(torch.tensor(seq_norm, dtype=torch.float32))
            
        self.register_buffer("pisano_phases", torch.stack(pisano_tensors))  # [num_sizes, 24]
        self.phase_proj = nn.Linear(len(resonant_sizes), d_model)

    def forward(self, batch_size: int, seq_len: int, device: torch.device = None) -> torch.Tensor:
        if device is None:
            device = self.pisano_phases.device
        positions = torch.arange(seq_len, device=device) % 24
        clock_phases = self.pisano_phases.to(device)[:, positions]  # [num_sizes, seq_len]
        clock_flat = clock_phases.permute(1, 0)                     # [seq_len, num_sizes]
        clock_emb = self.phase_proj(clock_flat)                     # [seq_len, d_model]
        return clock_emb.unsqueeze(0).expand(batch_size, -1, -1)     # [B, L, D]


class ResonantMicroNet(nn.Module):
    """
    Represents a single resonant micro-network as a Complete Graph K_n.
    Off-diagonal elements of raw_adj represent learnable inter-neuron synaptic coupling.
    """
    def __init__(self, n_nodes: int, d_model: int):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_model = d_model
        
        # Complete graph Kn adjacency weights (learnable synaptic coupling)
        self.raw_adj = nn.Parameter(torch.randn(n_nodes, n_nodes) / math.sqrt(n_nodes))
        
        # Node projections
        self.input_proj = nn.Linear(d_model, n_nodes)
        self.output_proj = nn.Linear(n_nodes, d_model)
        self.act = nn.GELU()

    def get_adjacency(self) -> torch.Tensor:
        """Returns zero-diagonal symmetric adjacency matrix for Kn."""
        adj = (self.raw_adj + self.raw_adj.T) / 2.0
        mask = torch.eye(self.n_nodes, device=adj.device)
        return adj * (1.0 - mask)  # Zero out self-loops

    def forward(self, x: torch.Tensor):
        """
        x: [batch_size, seq_len, d_model]
        returns:
            nodes_activated: [batch_size, seq_len, n_nodes]
            out: [batch_size, seq_len, d_model]
        """
        nodes = self.input_proj(x)
        adj = self.get_adjacency()
        nodes_updated = torch.matmul(nodes, adj)
        nodes_activated = self.act(nodes_updated)
        out = self.output_proj(nodes_activated)
        return nodes_activated, out


class AttentiveSaliencePooler(nn.Module):
    """
    GCL Attentive Salience Pooler (Multi-Slot Extraction):
    Extracts K-aspect slot representations using learned query-based attention weights.
    Prioritizes semantically salient tokens over auxiliary/stop tokens, preventing dilution.
    """
    def __init__(self, d_model: int, schema_dim: int, num_slots: int = 4):
        super().__init__()
        self.d_model = d_model
        self.schema_dim = schema_dim
        self.num_slots = num_slots
        self.slot_dim = max(16, schema_dim // num_slots)
        
        # Salience queries for K distinct aspects (Semantics, Structure, Goal, Harmony)
        self.query_proj = nn.Linear(d_model, num_slots)
        self.value_proj = nn.Linear(d_model, self.slot_dim * num_slots)
        self.out_proj = nn.Linear(self.slot_dim * num_slots, schema_dim)
        self.norm = nn.LayerNorm(schema_dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None, causal: bool = False):
        """
        x: [B, L, D]
        attention_mask: [B, L] optional mask
        returns:
            schema: [B, schema_dim] (or [B, L, schema_dim] if causal)
            salience_weights: [B, L, num_slots]
        """
        B, L, D = x.shape
        logits = self.query_proj(x) / math.sqrt(D)  # [B, L, K]
        
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask_exp = attention_mask.unsqueeze(-1)  # [B, L, 1]
            else:
                mask_exp = attention_mask
            logits = logits.masked_fill(mask_exp == 0, -1e9)
            
        values = self.value_proj(x).view(B, L, self.num_slots, self.slot_dim)  # [B, L, K, S]
        
        if causal:
            # Causal salience pooling: position i aggregates values from positions 0..i
            causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()  # [L, L]
            logits_k = logits.permute(0, 2, 1).unsqueeze(2).expand(-1, -1, L, -1)          # [B, K, L_q, L_k]
            logits_k = logits_k.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
            attn_weights = F.softmax(logits_k, dim=-1)                                      # [B, K, L, L]
            values_k = values.permute(0, 2, 1, 3)                                           # [B, K, L, S]
            causal_slots = torch.matmul(attn_weights, values_k)                             # [B, K, L, S]
            causal_flat = causal_slots.permute(0, 2, 1, 3).reshape(B, L, self.num_slots * self.slot_dim)
            schema = self.norm(self.out_proj(causal_flat))                                  # [B, L, schema_dim]
            salience_weights = F.softmax(logits, dim=1)
            return schema, salience_weights
        else:
            # Global attentive pooling over L
            salience_weights = F.softmax(logits, dim=1)                                     # [B, L, K]
            slots = torch.einsum('blk,blks->bks', salience_weights, values)                 # [B, K, S]
            slots_flat = slots.reshape(B, self.num_slots * self.slot_dim)
            schema = self.norm(self.out_proj(slots_flat))                                  # [B, schema_dim]
            return schema, salience_weights


class FiLMModulator(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) and Schema Reflector:
    Modulates token representations: y = (1.0 + tanh(gamma(S))) * x + beta(S)
    Enables selective enhancement of contextual features and attenuation of irrelevant noise.
    """
    def __init__(self, d_model: int, schema_dim: int):
        super().__init__()
        self.d_model = d_model
        self.schema_dim = schema_dim
        
        self.film_gen = nn.Sequential(
            nn.Linear(schema_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 2)
        )
        
        self.refine = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, schema: torch.Tensor):
        """
        x: [B, L, d_model]
        schema: [B, schema_dim] or [B, L, schema_dim]
        returns:
            refined_x: [B, L, d_model]
            schema_emb: [B, L, d_model]
        """
        if schema.dim() == 2:
            film_params = self.film_gen(schema).unsqueeze(1)  # [B, 1, 2 * d_model]
        else:
            film_params = self.film_gen(schema)               # [B, L, 2 * d_model]
            
        gamma, beta = film_params.chunk(2, dim=-1)
        # Bounded scale modulation: 1.0 + tanh(gamma) in (0, 2)
        scale = 1.0 + torch.tanh(gamma)
        modulated = scale * x + beta
        refined = self.refine(self.norm(modulated))
        schema_emb = beta.expand_as(x) if schema.dim() == 2 else beta
        return refined, schema_emb


class AdaptiveGate(nn.Module):
    """
    Adaptive Gate mechanism with negative-bias warm-up initialization.
    Starts with gate ≈ 0.119 (sigmoid(-2.0)) to preserve pristine base representations
    and gradually opens up to integrate schema as training matures.
    """
    def __init__(self, d_model: int, init_bias: float = -2.0):
        super().__init__()
        self.gate_linear = nn.Linear(d_model * 2, 1)
        nn.init.xavier_uniform_(self.gate_linear.weight, gain=0.1)
        nn.init.constant_(self.gate_linear.bias, init_bias)

    def forward(self, x: torch.Tensor, schema_expanded: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([x, schema_expanded], dim=-1)
        return torch.sigmoid(self.gate_linear(concat))


class SchemaMemory(nn.Module):
    """
    GCL Working Memory - maintains and updates running multi-aspect global schema
    during autoregressive generation (token-by-token) with learnable EMA.
    """
    def __init__(self, schema_dim: int, ema_decay: float = 0.95):
        super().__init__()
        self.schema_dim = schema_dim
        self.ema_decay = ema_decay
        self.register_buffer("cached_schema", torch.zeros(1, schema_dim))
        self.register_buffer("is_initialized", torch.tensor(False))

    def reset(self):
        """Resets cached schema for a new sequence/conversation."""
        self.cached_schema.zero_()
        self.is_initialized.fill_(False)

    def update(self, new_schema: torch.Tensor) -> torch.Tensor:
        """
        Incorporate new schema projection into memory via exponential moving average.
        new_schema: [B, schema_dim]
        returns: updated schema [B, schema_dim]
        """
        if not self.is_initialized:
            self.cached_schema = new_schema.detach().mean(dim=0, keepdim=True)
            self.is_initialized.fill_(True)
            return new_schema
        
        alpha = self.ema_decay
        updated = alpha * self.cached_schema.expand_as(new_schema) + (1.0 - alpha) * new_schema
        self.cached_schema = updated.detach().mean(dim=0, keepdim=True)
        return updated


class GCLResonantLayer(nn.Module):
    """
    Fused GCL v3.0 & Resonant Core Layer.
    Pipeline:
      1. Attentive Salience Pooling (Multi-Head K-Slot Global Context extraction)
      2. FiLM Modulation & Schema Reflector (Scale & Shift + Nonlinear Refinement)
      3. Adaptive Gate with negative-bias warm-up
      4. Pisano Clock Injection (pi=24)
      5. Parallel Complete Graph Kn Micro-Networks (T <= 7 dynamic convergence loop)
    """
    def __init__(self, d_model: int, schema_dim: int = 128, num_slots: int = 4, max_iter: int = 7, tolerance: float = 1e-3, causal: bool = False):
        super().__init__()
        self.d_model = d_model
        self.schema_dim = schema_dim
        self.num_slots = num_slots
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.causal = causal
        
        # GCL v3.0 components
        self.salience_pooler = AttentiveSaliencePooler(d_model, schema_dim, num_slots=num_slots)
        self.film_modulator = FiLMModulator(d_model, schema_dim)
        self.gate = AdaptiveGate(d_model, init_bias=-2.0)
        self.memory = SchemaMemory(schema_dim)
        
        # Pisano Clock
        self.clock = PisanoClock(d_model)
        
        # Resonant Kn Micro-Networks (6, 12, 18, 24) with complete graph Kn topology
        self.sizes = [6, 12, 18, 24]
        self.micro_nets = nn.ModuleList([
            ResonantMicroNet(n, d_model) for n in self.sizes
        ])
        
        self.comm_weight = nn.Parameter(torch.tensor(0.3))
        self.norm = nn.LayerNorm(d_model)
        self.fusion = nn.Linear(d_model * 4, d_model)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None, return_details: bool = False, use_memory: bool = False):
        """
        x: [B, L, D]
        attention_mask: [B, L] optional binary or float mask
        return_details: if True, returns output and details dictionary
        use_memory: if True, uses SchemaMemory for incremental generation
        """
        B, L, D = x.shape
        
        # Step 1: GCL Attentive Salience Multi-Slot Pooling
        schema, salience_weights = self.salience_pooler(x, attention_mask=attention_mask, causal=self.causal)
        
        if use_memory and not self.causal:
            schema = self.memory.update(schema)
            
        # Step 2: FiLM Modulation & Reflector
        reflected_x, schema_expanded = self.film_modulator(x, schema)
        gate_val = self.gate(x, schema_expanded)
        
        # Enhanced GCL Representation with Residual
        gcl_x = x + gate_val * reflected_x
        
        # Step 3: Pisano Clock Injection
        clock_emb = self.clock(B, L, device=x.device)
        state = self.norm(gcl_x + clock_emb)
        
        prev_fused = None
        iters_taken = 0
        all_phase_states = []
        diff_val = 0.0
        
        # Step 4: Resonant Micro-Network Convergence Loop (Complete Graph Kn, T <= 7)
        for t in range(self.max_iter):
            iters_taken += 1
            net_outputs = []
            current_phases = []
            
            for micro_net in self.micro_nets:
                nodes_act, out = micro_net(state)
                net_outputs.append(out)
                current_phases.append(nodes_act)
                
            all_phase_states.append(current_phases)
            
            concat_out = torch.cat(net_outputs, dim=-1)
            fused = self.fusion(concat_out)
            
            w = torch.clamp(self.comm_weight, 0.0, 0.9)
            state = state * (1.0 - w) + fused * w
            
            if prev_fused is not None:
                # Relative delta norm for dynamic convergence
                delta = torch.norm(fused - prev_fused, dim=-1).mean()
                base_norm = torch.norm(fused, dim=-1).mean() + 1e-6
                diff_val = (delta / base_norm).item()
                if diff_val < self.tolerance:
                    break
                    
            prev_fused = fused
            
        output = x + state
        
        if return_details:
            details = {
                "iters": iters_taken,
                "phases": all_phase_states,
                "schema": schema,
                "salience_weights": salience_weights,
                "gate": gate_val,
                "diff": diff_val
            }
            return output, details
            
        return output
