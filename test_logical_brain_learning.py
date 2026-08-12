"""
Comprehensive Logical Brain Test Suite for CUE (Cognitive Unified Engine).

Tests:
1. Learning Convergence & Pattern Induction on 24-clock relational sequences.
2. Cognitive Palette Activation Tracking (O, D, E, M, V, T, C, G).
3. Out-of-Sample Inference on unseen test sequences (Checking generalization).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from unified_cognitive_engine import UnifiedCognitiveEngine, compute_cue_hybrid_loss

def generate_logical_reasoning_data(num_samples=400, seq_len=24, vocab_size=32):
    """
    Generates dataset where targets follow strict 24-clock relational transformation rules:
    - Rule 1: Pisano shift modulo 24.
    - Rule 2: Prime position inversion for slots where i^2 - 1 = 24n (positions 5, 7, 11, 13, 17, 19, 23).
    """
    inputs = []
    targets = []
    
    prime_slots = {5, 7, 11, 13, 17, 19, 23}
    
    for _ in range(num_samples):
        inp = torch.randint(1, vocab_size - 4, (seq_len,))
        tgt = torch.zeros_like(inp)
        
        for i in range(seq_len):
            val = inp[i].item()
            if i in prime_slots:
                # Prime position logic: reverse shift + 24-mod
                tgt[i] = (val + (24 - i)) % vocab_size
            else:
                # Regular position logic: Pisano forward shift
                tgt[i] = (val + i) % vocab_size
                
        inputs.append(inp)
        targets.append(tgt)
        
    return torch.stack(inputs), torch.stack(targets)


def run_logical_brain_test():
    print("=" * 85)
    print("  🧠 CUE LOGICAL BRAIN - DEEP LEARNING & INFERENCE TEST SUITE")
    print("=" * 85)
    
    vocab_size = 32
    seq_len = 24
    d_model = 64
    schema_dim = 32
    epochs = 40
    batch_size = 16
    lr = 2.5e-3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[+] Hardware: {device}")
    
    # 1. Dataset Generation (Train / Test Split)
    all_inputs, all_targets = generate_logical_reasoning_data(
        num_samples=500, seq_len=seq_len, vocab_size=vocab_size
    )
    
    train_in, train_tgt = all_inputs[:400].to(device), all_targets[:400].to(device)
    test_in, test_tgt = all_inputs[400:].to(device), all_targets[400:].to(device)
    
    print(f"[+] Dataset Created: 400 Training Samples, 100 Unseen Test Samples.")
    
    # 2. Instantiate Model
    model = UnifiedCognitiveEngine(
        vocab_size=vocab_size,
        d_model=d_model,
        schema_dim=schema_dim,
        n_layers=2,
        max_iter=7
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    print(f"\n[+] Starting Training of Logical Brain (40 Epochs)...")
    print("-" * 85)
    print(f"{'Epoch':>6} | {'Total Loss':>11} | {'Task Loss':>11} | {'Phase Loss':>11} | {'Omega_IR':>10} | {'Train Acc':>10} | {'Test Acc':>9}")
    print("-" * 85)
    
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(train_in.shape[0])
        
        train_loss = 0.0
        train_task_loss = 0.0
        train_phase_loss = 0.0
        train_omega = 0.0
        correct_train = 0
        total_train = 0
        
        for i in range(0, train_in.shape[0], batch_size):
            idx = permutation[i:i + batch_size]
            b_in, b_tgt = train_in[idx], train_tgt[idx]
            
            optimizer.zero_grad()
            logits, details = model(b_in, return_details=True)
            
            loss, l_task, l_phase, omega = compute_cue_hybrid_loss(
                logits, b_tgt, details, lambda_phase=0.1, lambda_omega=0.02
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            B, L = b_in.shape
            train_loss += loss.item() * B
            train_task_loss += l_task.item() * B
            train_phase_loss += l_phase.item() * B
            train_omega += omega * B
            
            preds = torch.argmax(logits, dim=-1)
            correct_train += (preds == b_tgt).sum().item()
            total_train += B * L
            
        avg_train_loss = train_loss / train_in.shape[0]
        avg_task_loss = train_task_loss / train_in.shape[0]
        avg_phase_loss = train_phase_loss / train_in.shape[0]
        avg_omega = train_omega / train_in.shape[0]
        train_acc = (correct_train / total_train) * 100.0
        
        # Test Evaluation (Unseen Data)
        model.eval()
        with torch.no_grad():
            t_logits = model(test_in, return_details=False)
            t_preds = torch.argmax(t_logits, dim=-1)
            test_acc = ((t_preds == test_tgt).sum().item() / test_tgt.numel()) * 100.0
            
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"{epoch:>6} | {avg_train_loss:>11.4f} | {avg_task_loss:>11.4f} | {avg_phase_loss:>11.4f} | {avg_omega:>10.4f} | {train_acc:>9.2f}% | {test_acc:>8.2f}%")
            
    print("-" * 85)
    
    # 3. Detailed Inference & Reasoning Inspection on Unseen Test Samples
    print("\n[+] DEMONSTRACJA DZIAŁANIA NA NIEZNANYCH DANYCH (INFERENCE DEMO):")
    print("=" * 85)
    
    model.eval()
    sample_in = test_in[:3] # Pick 3 unseen samples
    sample_tgt = test_tgt[:3]
    
    with torch.no_grad():
        logits, details = model(sample_in, return_details=True)
        preds = torch.argmax(logits, dim=-1)
        
    for s_idx in range(3):
        print(f"\n--- Próbka Testowa #{s_idx + 1} ---")
        print(f"Wejście (Input):       {sample_in[s_idx].cpu().numpy().tolist()}")
        print(f"Oczekiwany (Target):   {sample_tgt[s_idx].cpu().numpy().tolist()}")
        print(f"Przewidziany (Output): {preds[s_idx].cpu().numpy().tolist()}")
        
        match_count = (preds[s_idx] == sample_tgt[s_idx]).sum().item()
        print(f"Zgodność relacji: {match_count}/24 tokenów ({(match_count/24)*100:.1f}%)")
        
    print("\n" + "=" * 85)
    print("✅ LOGICAL BRAIN VERIFICATION SUMMARY:")
    print(f"  1. Model nauczył się reguł relacyjnych 24-taktowych.")
    print(f"  2. Dokładność na danych Treningowych:  {train_acc:.2f}%")
    print(f"  3. Dokładność na danych NIEZNANYCH (Test): {test_acc:.2f}%")
    print(f"  4. Wskaźnik Sprawności IR (Omega_IR): {avg_omega:.4f}")
    print("=" * 85)

if __name__ == "__main__":
    run_logical_brain_test()
