r"""
Comprehensive Unit & Integration Test Suite for CUE (Cognitive Unified Engine).
Validates:
1. Prime Sieve 24 ($p^2 - 1 = 24n$) and active prime slots.
2. Cognitive Palette & Axiom of Recursion ($R(A, B) \rightarrow O'$).
3. HOX & IR Validator and Differentiable $\Omega_{\text{IR}}$ Calculator.
4. GCL & Complete Graph $K_n$ Resonant Layer ($T \le 7$, Causal safety, Masking).
5. Full End-to-End Gradient Flow across all CUE modules.
6. CUE Transformer & LLM Integration.
"""

import math
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cognitive_primitives import CognitivePalette, CognitiveObject, CognitiveRelation, CognitiveRelationLayer
from hox_ir_validator import PrimeSieve24, OmegaIRCalculator, IRValidator
from gcl_resonant_layer import PisanoClock, ResonantMicroNet, SchemaMemory, GCLResonantLayer
from unified_cognitive_engine import UnifiedCognitiveEngine, compute_cue_hybrid_loss
from example_llm_integration import CUETransformerBlock, CUELanguageModel


class TestPrimeSieve24(unittest.TestCase):
    def test_prime_theorem(self):
        primes = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for p in primes:
            is_div, n_val = PrimeSieve24.verify_prime_theorem(p)
            self.assertTrue(is_div, f"Failed for prime p={p}")
            self.assertEqual(p * p - 1, 24 * n_val)
            
    def test_prime_slots(self):
        prime_slots = [1, 5, 7, 11, 13, 17, 19, 23]
        for slot in prime_slots:
            self.assertTrue(PrimeSieve24.is_valid_prime_slot(slot))
        mask = PrimeSieve24.get_prime_slot_mask(48)
        self.assertEqual(mask.shape, (48,))
        self.assertEqual(mask[5].item(), 1.0)
        self.assertEqual(mask[6].item(), 0.0)


class TestCognitivePrimitives(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.palette = CognitivePalette(self.d_model)

    def test_primitives_numerical_stability(self):
        state_a = torch.zeros(2, self.d_model)
        state_b = torch.zeros(2, self.d_model)  # Identical states (distance 0)
        
        results = self.palette(state_a, state_b)
        self.assertFalse(torch.isnan(results["M_dist"]).any(), "M_dist has NaN for zero distance")
        self.assertFalse(torch.isnan(results["V_direction"]).any(), "V_direction has NaN for zero distance")
        self.assertFalse(torch.isnan(results["E_similarity"]).any(), "E_similarity has NaN")
        self.assertAlmostEqual(results["E_similarity"].mean().item(), 1.0, places=3)

    def test_recursion_axiom(self):
        synth = CognitiveRelationLayer(self.d_model)
        state_a = torch.randn(2, self.d_model)
        state_b = torch.randn(2, self.d_model)
        
        o_prime = synth(state_a, state_b)
        self.assertEqual(o_prime.shape, (2, self.d_model))
        self.assertFalse(torch.isnan(o_prime).any())


class TestHOXIRValidator(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.validator = IRValidator(self.d_model)
        self.omega_calc = OmegaIRCalculator()

    def test_ir_validator_outputs(self):
        x = torch.randn(2, 24, self.d_model)
        eicr_scores, harmony_score, is_valid = self.validator(x)
        
        self.assertEqual(eicr_scores.shape, (2, 24, 4))
        self.assertEqual(harmony_score.shape, (2, 24, 1))
        self.assertTrue((harmony_score >= 0.0).all() and (harmony_score <= 1.0).all())

    def test_omega_ir_differentiability(self):
        phi_iit = torch.tensor([[[0.85]]], requires_grad=True)
        mass = torch.tensor([[[1.0]]])
        energy = torch.tensor([[[1.0]]])
        entropy_h = torch.tensor([[[2.0]]])
        
        omega = self.omega_calc(phi_iit, mass, energy, entropy_h)
        omega.backward()
        self.assertIsNotNone(phi_iit.grad)
        self.assertGreater(phi_iit.grad.item(), 0.0)


class TestGCLResonantLayer(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.layer = GCLResonantLayer(self.d_model, schema_dim=32, max_iter=7)

    def test_micro_net_kn_adjacency(self):
        for micro_net in self.layer.micro_nets:
            adj = micro_net.get_adjacency()
            # Check zero diagonal
            diag = torch.diagonal(adj)
            self.assertTrue(torch.allclose(diag, torch.zeros_like(diag)))
            # Check symmetry
            self.assertTrue(torch.allclose(adj, adj.T))

    def test_convergence_limit(self):
        x = torch.randn(2, 24, self.d_model)
        out, details = self.layer(x, return_details=True)
        self.assertEqual(out.shape, (2, 24, self.d_model))
        self.assertLessEqual(details["iters"], 7)
        self.assertEqual(len(details["phases"]), details["iters"])

    def test_causal_no_future_leak(self):
        causal_layer = GCLResonantLayer(self.d_model, schema_dim=32, max_iter=3, causal=True)
        causal_layer.eval()
        
        seq_len = 10
        x1 = torch.randn(1, seq_len, self.d_model)
        x2 = x1.clone()
        # Alter future tokens (from position 5 onwards)
        x2[:, 5:, :] = torch.randn(1, seq_len - 5, self.d_model)
        
        with torch.no_grad():
            out1 = causal_layer(x1)
            out2 = causal_layer(x2)
            
        # Positions 0..4 must be identical
        diff_past = torch.max(torch.abs(out1[:, :5, :] - out2[:, :5, :])).item()
        self.assertAlmostEqual(diff_past, 0.0, places=5)

    def test_schema_memory(self):
        memory = SchemaMemory(schema_dim=32, ema_decay=0.9)
        s1 = torch.ones(1, 32)
        out1 = memory.update(s1)
        self.assertTrue(torch.allclose(out1, s1))
        
        s2 = torch.zeros(1, 32)
        out2 = memory.update(s2)
        # EMA: 0.9 * 1.0 + 0.1 * 0.0 = 0.9
        self.assertAlmostEqual(out2.mean().item(), 0.9, places=3)
        
        memory.reset()
        self.assertFalse(memory.is_initialized.item())


class TestUnifiedCognitiveEngine(unittest.TestCase):
    def test_full_gradient_flow(self):
        vocab_size = 32
        d_model = 64
        model = UnifiedCognitiveEngine(vocab_size=vocab_size, d_model=d_model, schema_dim=32, n_layers=2)
        
        inputs = torch.randint(0, vocab_size, (2, 24))
        targets = torch.randint(0, vocab_size, (2, 24))
        
        logits, details = model(inputs, return_details=True)
        loss, l_task, l_phase, omega = compute_cue_hybrid_loss(logits, targets, details)
        
        loss.backward()
        
        # Verify gradients exist across all key submodules
        self.assertIsNotNone(model.ir_validator.harmony_classifier.weight.grad, "harmony_classifier grad missing")
        self.assertIsNotNone(model.palette.causal_weight.weight.grad, "palette causal_weight grad missing")
        self.assertIsNotNone(model.palette.relation_synth.synth[0].weight.grad, "palette relation_synth grad missing")
        self.assertIsNotNone(model.layers[0].micro_nets[0].raw_adj.grad, "micro_net raw_adj grad missing")
        self.assertIsNotNone(model.layers[0].schema_pool.weight.grad, "schema_pool grad missing")
        self.assertIsNotNone(model.layers[0].fusion.weight.grad, "fusion grad missing")
        self.assertIsNotNone(model.lm_head.weight.grad, "lm_head grad missing")


class TestLLMIntegration(unittest.TestCase):
    def test_transformer_block_and_mini_llm(self):
        model = CUELanguageModel(vocab_size=100, d_model=64, n_layers=2)
        inputs = torch.randint(0, 100, (2, 16))
        
        logits, details = model(inputs, return_cue_details=True)
        self.assertEqual(logits.shape, (2, 16, 100))
        self.assertEqual(len(details), 2)
        self.assertIn("iters", details[0])
        self.assertIn("gate", details[0])


if __name__ == "__main__":
    unittest.main()
