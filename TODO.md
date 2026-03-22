# Parameter Golf: Implementation Plan

## Current Best: 1.1697 bpb, 15.7MB (INT5+zstd22 + late QAT on 11L MLP3x)

## The Core Insight

At 16MB, the winner will be **absurdly parameter-efficient** — not just "a good small model" but something that maximizes effective capacity per stored byte through structured transformations. Key directions from senior LLM researchers:

1. **Extreme sparsity**: Fastfood layers, structured random matrices
2. **Recurrence pushed to the max**: shared weights + per-loop LoRA/signals + aux loss
3. **Random projections via seed storage**: regenerate large matrices from a stored seed instead of storing weights
4. **Butterfly matrices / FFT-based ops**: O(n log n) mixing instead of O(n²) dense layers
5. **Exploiting ops like FFT**: frequency-domain processing for cheap global mixing

The theme: **structure beats parameters**. A 16MB model with structured ops can have the effective capacity of a much larger model.

### The Winning Architecture (Hypothesis)
The winner likely combines ALL of these:
- **Deep recurrence** (shared weights + LoRA differentiation) for effective depth 20-40+
- **Sparse attention + MLP** (Fastfood/butterfly/FFT) for O(d) params per layer
- **Aggressive quantization** (ternary/INT4) on the few stored params
- **MoE routing** for conditional computation (many experts, few active per token)
- **Longer sequence length** enabled by the param savings

This combination gives a model that is effectively "much larger" (100M+ effective params) while storing only a few MB of unique weights. Each technique multiplies the others' benefit.

### Deep Mixture-of-Experts (MoE) [HIGH PRIORITY]
MoE is a natural fit for 16MB: store many small experts, route tokens to a few at a time.
- Expert size can be tiny (Fastfood or butterfly-based)
- Router is very cheap (~dim*num_experts params)
- With recurrence: same experts used at different loops = massive effective capacity
- Top-2 routing with load balancing aux loss
- Could combine with ternary quantization for even more experts

## Phase 0: Infrastructure [DONE]

- [x] SLURM training scripts + LUMI SSH bridge
- [x] ROCm compatibility (MI250X + torch.compile, 189ms/step 8-GPU)
- [x] Dataset setup, experiment tracking, Claude Code skills
- [x] Per-experiment scripts workflow (specs/batchN/runM.py)
- [x] 8 experiments per hour throughput

## Phase 1: Baseline Optimization [DONE — 1.1697 bpb]

- [x] Architecture: 11L MLP3x dim512 seq2048 (best depth/width trade-off)
- [x] Optimizer: LR=0.02, momentum=0.99, warmdown=3000
- [x] Compression: INT5+zstd22 (13.5-15.7MB, fits 16MB limit)
- [x] Late QAT at 50%: -0.017 bpb improvement (biggest single technique)
- [x] Validated: 1.1697 bpb, 15.7MB artifact

Key lessons:
- INT5+zstd22 is the compression sweet spot (INT6 doesn't fit, INT4 too lossy)
- Big batch helps in 1-GPU exploration but hurts at 8-GPU scale
- Late QAT is the single biggest post-compression win

## Phase 2: Structured Efficiency [IN PROGRESS]

### 2A: Depth Recurrence (Batches 14-17)
Status: gap narrowed from +0.069 → +0.028 with LoRA+aux loss+curriculum, but still behind standard 11L.

- [x] Plain recurrence (batch 14): +0.026-0.069 worse, norm swap too weak
- [x] Per-loop LoRA rank 32-64 (batch 15): +0.048, LoRA helps but not enough alone
- [x] Aux loss + curriculum (batch 16): +0.028, significant improvement
- [x] Wider recurrent models (batch 17): r3x3 dim640 MLP4x = 1.305 exploration
- [ ] Try LoRA on ALL linear layers (not just Q/V) — more per-loop capacity
- [ ] Try higher LoRA rank (128+) on wider models — with recurrence headroom we can afford it
- [ ] Validate best recurrent config at 8-GPU scale (job 16932792 running)

### 2B: Fastfood / Structured Random Layers [NOT STARTED — HIGH PRIORITY]
The MLP is the biggest parameter consumer. Replace dense MLP with structured alternatives:

- [ ] **Fastfood Transform**: Replace MLP weight matrices with `S·H·G·Π·H·B` where:
  - S = diagonal scaling (d params)
  - H = Hadamard matrix (free, no params)
  - G = diagonal Gaussian (d params from seed!)
  - Π = random permutation (from seed!)
  - B = diagonal binary ±1 (from seed!)
  - Total: ~2d learnable params instead of d² — massive savings
- [ ] **Butterfly matrices**: O(n log n) parameterization for linear transforms
  - log₂(d) layers of sparse butterfly factors
  - ~d·log(d) params instead of d² for the weight matrix
- [ ] **Monarch matrices**: product of two block-diagonal matrices
  - sqrt(n) blocks of sqrt(n)×sqrt(n) each
  - Captures both local and global patterns

### 2C: Seed-Based Random Projections [NOT STARTED — HIGH PRIORITY]
Store a seed (8 bytes) instead of a full random matrix (~2MB):

- [ ] **Random feature attention**: replace Q·K^T with random Fourier features
  - Store the projection matrix seed, regenerate at eval time
  - Linear attention approximation: O(n·d) instead of O(n²)
- [ ] **Frozen random MLP layers**: some MLP layers use seed-generated weights (frozen)
  - Only store the scaling/bias (d params per frozen layer)
  - "Extreme Lottery Ticket" — random projections + learned routing
- [ ] **Hash-based embeddings**: reduce embedding table via hashing with stored seed

### 2D: FFT-Based Mixing [NOT STARTED — MEDIUM PRIORITY]
Replace attention or MLP with frequency-domain operations:

- [ ] **FNet-style FFT mixing**: replace attention with 2D FFT (no learned params!)
  - Alternating FFT and MLP layers
  - Or use FFT for some heads, attention for others (hybrid)
- [ ] **Spectral MLP**: apply learned transforms in frequency domain
  - FFT → pointwise multiply by learned spectrum → inverse FFT
  - d params instead of d² for the mixing matrix
- [ ] **Convolutional mixing**: short convolutions (kernel=3-7) for local patterns
  - Very few params, complements global attention

### 2E: Advanced Quantization [PARTIALLY DONE]
- [x] INT5+zstd22 baseline (15.7MB for 11L)
- [ ] **Ternary QAT** (BitNet b1.58): {-1, 0, +1} weights → ~2 bits per weight
  - Could fit 40-50M params in ~10MB (vs 26M in 15.7MB with INT5)
  - Requires careful STE training + lambda warmup
- [ ] **Mixed ternary/INT8**: ternary for large matrices, INT8 for embeddings
- [ ] Custom 2-bit packing + zstd for ternary weights

## Phase 3: Test-Time Domination [NOT STARTED]

Exploit the separate 10-minute eval budget (currently unused!):

- [ ] TTT: Fine-tune on validation data during eval (LoRA or full)
- [ ] More recurrence loops at eval time (free effective depth!)
- [ ] Sliding window eval for longer effective context
- [ ] NTK-aware RoPE scaling for context extension
- [ ] Meta-learning: train initialization optimized for gradient descent at eval

## Phase 4: Moonshots

- [ ] Deep Equilibrium Model (DEQ): fixed-point iteration
- [ ] Mixture of Recursions: per-token dynamic depth
- [ ] Knowledge distillation from larger model
- [ ] LAWA weight averaging
- [ ] Byte-level tokenizer (vocab=256, minimal embedding overhead)
- [ ] SP-4096 tokenizer with factored embeddings

## Throughput Note

We run **8 experiments per hour** on LUMI (8x 1-GPU packed). Both exploration and validation nodes should be running at all times. At this throughput, we can test 50-100 ideas per day.
