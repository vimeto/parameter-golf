# Parameter Golf: Comprehensive Research Analysis

**Updated: 2026-03-24**
**Competition:** [OpenAI Parameter Golf](https://github.com/openai/parameter-golf)
**Objective:** Train the best language model (lowest bits-per-byte) in a 16MB artifact, training in ≤10 min on 8xH100 SXM.

## Competition Landscape (as of 2026-03-24)

### Merged Leaderboard

| Rank | bpb | Author | Key Techniques |
|------|-----|--------|----------------|
| 1 | **1.1228** | signalrush | 11L EMA + GPTQ-lite + warmdown3500 + QAT@0.15 |
| 2 | 1.1248 | jfprincz | 11L Partial RoPE + LN Scale + EMA + XSA4 |
| 3 | 1.1271 | jfprincz | 11L XSA4 + EMA + Int6 MLP3x + WD=0.04 |
| 4 | 1.1307 | unnir | 11L Efficient Partial XSA |
| 5 | 1.1428 | thwu1 | 10L Int5-MLP + BigramHash10240 + SWA |

### Open PR Frontier

| bpb | TTT? | Key Innovation |
|-----|------|---------------|
| **0.6430** | 8-epoch | PROTEUS: rank-16 LoRA TTT, per-block bias, T=0.98 |
| 0.7853 | 5-epoch | PROTEUS v8: cosine LR, score every epoch |
| 1.0523 | 3-pass | Multi-pass streaming score-first TTT |
| **1.1171** | **No** | **Full GPTQ + LeakyReLU(0.5)² + Parallel Muon** |
| 1.1175 | No | Value Residual Learning + LeakyReLU² + Full GPTQ |
| 1.1181 | No | SwiGLU (Star-ReLU, hidden=1792) + VE128 |

### Converged Architecture Stack (All Top Entries)

- 11 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- 3x MLP expansion (1536 hidden), relu² activation
- U-Net skip connections (5 encoder + 6 decoder)
- XSA on last 4 layers, Partial RoPE (16/64 dims), LN Scale (1/sqrt(layer+1))
- SmearGate + BigramHash (2048 buckets) + Shared Value Embedding
- Muon (lr=0.025, WD=0.04) + AdamW (embeddings)
- EMA (0.997) + Tight SWA + Late QAT (threshold 0.15)
- GPTQ-lite (5-percentile clip search) + Int6/Int8 mixed + zstd-22
- Sliding window eval (stride=64)

## Immediate Improvement Opportunities

### Tier 1: Proven, Ready to Deploy

| Technique | Impact | Effort | Evidence |
|-----------|--------|--------|---------|
| **LeakyReLU(0.5)²** | -0.004 bpb | 1 line | Batch 39 confirmed, PRs #593/#569/#535 |
| **Full Hessian GPTQ** | -0.005 bpb | Scripts ready | PR #593 achieves 1.1171 no-TTT |
| **Combined** | **~-0.009** | Ready | **Target: 1.123 bpb** |

### Tier 2: From Literature, Worth Testing

| Technique | Impact | Source | Notes |
|-----------|--------|--------|-------|
| Multi-Token Prediction | -0.002 est. | NanoGPT speedrun | Free auxiliary signal |
| Hadamard pre-GPTQ (from QuIP#) | -0.001 to -0.003 | arXiv:2402.04396 | Random ortho rotation before quantization |
| Paired Head Attention | -0.001 est. | NanoGPT speedrun | Doubles context for same compute |
| Cautious Weight Decay | Unknown | NanoGPT speedrun | Gated decay |
| YaRN Window Warmup | Unknown | NanoGPT speedrun | Gradual context expansion |
| 2-bit QAT (ParetoQ recipe) | Large if works | arXiv:2502.02631 | 3x params in 16MB |

### Tier 3: TTT (Legal Eval-Time Adaptation)

| Method | bpb | Eval Time | Approach |
|--------|-----|-----------|----------|
| Document-isolated sliding eval | ~-0.011 | ~70s | Just reset state between docs |
| Score-first SGD TTT (3 epochs) | ~-0.02 | ~300s | Train on scored chunks |
| PROTEUS LoRA TTT (8 epochs) | ~-0.50 | Full 10min | Rank-16 LoRA, per-block bias |

**Key insight**: Most "TTT" gain is from document isolation + striding, not gradient steps.

## NanoGPT Speedrun Analysis

The modded-nanogpt speedrun (3.28 val loss in <90s on 8xH100) demonstrates:

### Training Recipe Innovations
- **NorMuon optimizer**: Polar Express orthogonalization, fused momentum+ortho kernel
- **Multi-stage schedule**: Dynamic batch (131K→393K), seq len (896→2048), window sizes
- **Multi-token prediction**: Predict next 1-3 tokens, weights decay during training
- **Adam on odd steps only**: Halves embedding/gate communication

### Architecture Innovations
- **Paired Head Attention**: Adjacent heads share keys, doubles context window
- **Value Embeddings** at 5 specific layers with gated injection
- **Simplified Hyperconnections**: Cache layer 7 output for layers 8-10
- **Sparse Attention Gates**: Sigmoid on first 12 dims of activation
- **Skip from layer 3 + backout from layer 7**

### Systems Innovations
- **FP8 matmul** for LM head (forward + backward)
- **Parameter banks**: Merged QKVO/MLP into 3D tensors for efficient sharding
- **Sparse gradient communication** for bigram embeddings
- **Triton kernels**: Fused ReLU², softcapped cross-entropy, Polar Express

## Quantization Research

### ParetoQ (arXiv:2502.02631) — Key Finding
2-bit, 3-bit, and ternary are all roughly Pareto-equivalent and **beat 4-bit**. Below 3 bits, training from scratch works better than post-training quantization. A 2-bit QAT model with 64M params in 16MB could potentially beat our 25M INT6 model.

### Full GPTQ (vs GPTQ-lite)
- Collect Hessian H=X^TX per layer via calibration batches
- Column-by-column quantization with Cholesky error compensation
- Worth -0.005 bpb over GPTQ-lite (5-percentile search)
- Zero training cost, applied at export

### Hadamard Incoherence (from QuIP#)
Apply random orthogonal (Hadamard) rotation before quantization to make weights incoherent. Cheap O(n log n) preprocessing that improves any quantization scheme.

## Alternative Architecture Assessment

| Architecture | Pros | Cons | Verdict |
|-------------|------|------|---------|
| **GLA (via FLA)** | ROCm Triton kernels, throughput gain | Quality loss at short context | Hybrid worth trying |
| **RWKV-7** | Strong scaling, FLA support | Untested at tiny scale | Possible exploration |
| **Mamba** | Linear-time | torch.compile NaN, custom kernels | Not recommended |
| **xLSTM** | Good at 125M+ | ROCm untested | Lower priority |
| **Hyena/Based** | Subquadratic | 1024 context too short | Not recommended |

**Conclusion: Transformers dominate at this scale.** No alternative architecture has beaten a well-optimized 11L transformer in the competition.

## Path to Sub-1.0 bpb

| Phase | Target | Techniques |
|-------|--------|-----------|
| A: Match SOTA | 1.12 | Full GPTQ + LeakyReLU² |
| B: Beat SOTA | 1.11 | + Hadamard pre-GPTQ + MTP |
| C: Light TTT | 1.09 | + Score-first TTT (3 epochs) |
| D: Heavy TTT | 0.8-0.9 | + PROTEUS 5-8 epoch LoRA TTT |
| E: Moonshot | <0.8 | + 2-bit QAT (3x params) + max TTT |

## References

- [NanoGPT Speedrun](https://github.com/kellerjordan/modded-nanogpt) — Training recipe innovations
- [XSA Paper (arXiv:2603.09078)](https://arxiv.org/abs/2603.09078) — Exclusive Self Attention
- [ParetoQ (arXiv:2502.02631)](https://arxiv.org/abs/2502.02631) — Pareto-optimal quantization
- [QuIP# (arXiv:2402.04396)](https://arxiv.org/abs/2402.04396) — Hadamard incoherence + lattice codebooks
- [Titans (arXiv:2501.00663)](https://arxiv.org/abs/2501.00663) — Memory-augmented test-time learning
- [RWKV-7 (arXiv:2503.14456)](https://arxiv.org/abs/2503.14456) — RNN with transformer quality
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) — Triton kernels for linear attention
- [Value Residual Learning](https://aclanthology.org/2025.acl-long.1375/) — ResFormer
- [PTQ1.61 (arXiv:2502.13179)](https://arxiv.org/abs/2502.13179) — Sub-2-bit post-training quantization
- [BitNet b1.58](https://arxiv.org/abs/2402.17764) — Native ternary training
- [Parameter Golf](https://github.com/openai/parameter-golf) — Competition repository
