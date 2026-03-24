# Parameter Golf: Implementation Plan

## Current State
- **Record-lane best**: 1.1324 bpb (val19, specs/batch33/run_new_winner.py, wd=4000)
- **Merged upstream SOTA**: 1.1233 bpb (PR #414)
- **Best open PR (no TTT)**: 1.1171 bpb (PR #593: Full GPTQ + LeakyReLU² + Parallel Muon)
- **Best open PR (with TTT)**: 0.6430 bpb (PR #596: PROTEUS 8-epoch LoRA TTT)

## Immediate: Close the Gap to 1.11 (No-TTT)

Three techniques get us from 1.1233 to ~1.11 with NO TTT. All proven in open PRs:

### 1. LeakyReLU(0.5)² [TRIVIAL — DO FIRST]
- One line: `F.leaky_relu(x, 0.5).square()` instead of `relu(x).square()`
- Impact: **-0.003 bpb** (consistent across PRs #593, #569, #518, #535)
- Zero params, zero cost

### 2. Full Hessian GPTQ [BIGGEST NO-TTT WIN]
- Replace GPTQ-lite (5-percentile clip search) with full Hessian-aware GPTQ
- Collect `H = X^T X` per layer via ~256 calibration batches after training
- Column reordering by Hessian diagonal, block-wise Cholesky error compensation
- Impact: **-0.005 bpb** (PRs #593, #569, #535)
- Zero training cost — only changes export/quantization
- Reference: PR #593 achieves 1.1171 with this + LeakyReLU²

### 3. Value Residual Learning (VRL) [SMALL BUT PROVEN]
- From ResFormer (arXiv:2410.17897)
- Cache V output from layer 0, blend into all subsequent layers via learned sigmoid gates
- Only 22 extra params
- Impact: **-0.002 bpb** (PR #569 ablation)

### Combined: 1.1233 - 0.003 - 0.005 - 0.002 = ~1.113 bpb (no TTT)

## Next: Legal TTT to Break 1.10

### 4. Score-First TTT [LEGAL EVAL-TIME — HIGH PRIORITY]
- Per-document LoRA (rank-8) on Q, V, LM-head
- Score each chunk FIRST, then train on it (no leakage)
- Reset between documents, batch 64 docs/GPU
- Impact: **-0.01 to -0.02 bpb** on top of no-TTT score
- Budget: 1-3 epochs, fits within 10-min eval
- PRs #549, #529, #557

### 5. Multi-Epoch Cosine TTT [HEAVY BUT MASSIVE]
- 5-20 epochs with cosine LR decay, per-layer LR groups
- LoRA rank-16 on LM-head, per-block bias tuning
- Impact: **1.11 → 0.78-1.05 bpb** depending on epochs
- PRs #596, #568, #573, #518

### Combined: ~1.113 - 0.02 = ~1.09 bpb (with light TTT)
### With heavy TTT: potentially < 0.80 bpb

## Other Proven Techniques from Open PRs

| Technique | Impact | Effort | Source |
|-----------|--------|--------|--------|
| XSA on ALL 11 layers | -0.001 | Trivial | #576, #587 |
| SwiGLU (hidden=1792) | -0.002 | Medium | #505 |
| Train Larger Quantize Harder (33.6M, INT5 GPTQ) | -0.003 | Medium | #576 |
| Parameter Banking + Parallel Muon | -0.002 | Complex | #593 |
| TrigramHash + ValueResidual + GradQuant | -0.003 | Medium | #486 |
| Shared Sparse Sidecar (late layers) | -0.002 | Medium | #555 |
| Late Soft-Round QAT | -0.001 | Small | #589 |
| QAT threshold 0.5 (earlier, more QAT steps) | -0.001 | Trivial | #535, #578 |

## Record Lane vs Non-Record Lane

### Record Lane (MAX_WALLCLOCK <= 3000s MI250)
- Current best: 1.1324 bpb
- Target: 1.11 (no TTT), 1.09 (light TTT)
- All submissions must produce <16MB artifacts

### Non-Record Lane (research, > 3000s)
- Current best: 1.1164 bpb (6000s)
- Useful for: finding architecture ceilings, developing TTT methods

## Explicitly Paused
- Full-model recurrence (DDP crashes, gap too large)
- Ternary/BitNet (gap too large with naive approach)
- Fastfood / block-diagonal MLP (compile/quality issues)
- Old INT5-base ablations (superseded)
- Calling non-record runs "better than leaderboard"

## Competitive Landscape (Open PRs, as of 2026-03-24)

### Tier 1: Heavy TTT (< 1.10 bpb)
- 0.64 (PROTEUS 8ep), 0.79 (PROTEUS 5ep), 0.95 (PROTEUS 3ep), 1.05 (multi-pass streaming)

### Tier 2: Light TTT or No-TTT (1.10-1.12 bpb)
- 1.1160 (batched LoRA TTT), 1.1164 (train larger quant harder)
- **1.1171 (Full GPTQ + LeakyReLU² — NO TTT!)** ← our target
- 1.1175 (VRL + LeakyReLU² + Full GPTQ — NO TTT)
- 1.1181 (SwiGLU + VE128 — NO TTT)

### The path is clear: LeakyReLU² + Full GPTQ + VRL gets to 1.11 no-TTT. Then TTT to go lower.
