# Parameter Golf: Implementation Plan

## Two Lanes

### Record Lane (MAX_WALLCLOCK <= 3000s on MI250, ~1.05x H100 token budget)
- **Current best: 1.1324 bpb** (val19, specs/batch33/run_new_winner.py, wd=4000, 15.84MB)
- **Upstream SOTA: 1.1233 bpb** (PR #414, 8xH100 600s)
- **Gap: +0.009** — mostly from fewer steps (MI250 slower) and no FlashAttention 3

### Non-Record Lane (research, MAX_WALLCLOCK > 3000s)
- **Current best: 1.1164 bpb** (val24, 6000s, ~18000 steps)
- NOT leaderboard-valid. Useful for: finding the architecture ceiling, studying scaling curves, developing techniques to port back to record lane.

### Invalid Runs (do NOT cite as records)
- val13 (0.9815): TRAIN_ON_VAL=1 violates rules
- val17 TTT: broken (1.3185), standard eval fine

## Immediate Priorities

### 1. Clean Bookkeeping
- [x] Fix state.json to separate record/non-record lanes
- [ ] Backfill validations 18-24 into validation_log.md
- [ ] Graduate proven features from new SOTA into root train_gpt.py
- [ ] Ensure all three metrics (standard, sliding, TTT) are clearly labeled

### 2. Record-Lane Improvement
The record-lane best (1.1324) uses the 2026-03-22 SOTA port. To improve:
- [ ] Fix TTT on the new SOTA backbone (broken in val17 — biggest potential win)
- [ ] Try legal causal eval-time adaptation (document-isolated sliding + TTT)
- [ ] Small training/export refinements (GPTQ-lite clip search tuning, better EMA settings)
- [ ] Keep MAX_WALLCLOCK <= 3000s for all record-lane validations

### 3. Legal Eval-Time Adaptation [HIGHEST UPSIDE]
The invalid val-only run (0.9815) proves massive gains exist in adaptation. Legal versions:
- [ ] TTT LoRA: fix the bug, then sweep rank/LR/target-layers on new SOTA backbone
- [ ] Titans-style memory / fast-weight mechanism at eval time
- [ ] Document-isolated sliding eval with causal adaptation
- [ ] Stay within 10-minute eval budget

### 4. H100 Confirmation
- [ ] Run record-lane candidate on H100 only when clearly below 1.1228 on MI250 (record lane)
- [ ] Or when legal eval-time method shows >0.005 nat margin

## What's Proven on MI250

| Technique | Best bpb | Lane | Script |
|-----------|----------|------|--------|
| Old winner port (10L) | 1.1441 | Record | specs/batch28/run_winner_rocm.py |
| New SOTA port (11L, wd4000) | 1.1324 | Record | specs/batch33/run_new_winner.py |
| Extended wallclock (6000s) | 1.1164 | Non-record | specs/batch33/run_new_winner.py |

## Scaling Curve (non-record)

| Wallclock | Steps | Sliding bpb | Delta |
|-----------|-------|-------------|-------|
| 2700s | ~8200 | 1.1324 | baseline |
| 3600s | ~10800 | 1.1265 | -0.006 |
| 4500s | ~13800 | 1.1215 | -0.005 |
| 6000s | ~18000 | 1.1164 | -0.005 |

~0.005 bpb per +1500s. Curve still improving but will plateau.

## Longer-Term Directions

### Ambitious: Legal Causal Adaptation / Memory
- Titans-style memory / fast-weight eval-time mechanism
- Hybrid dense-trunk + recurrent-tail model
- This is the only credible path to sub-1.0 within rules

### Architecture
- Keep current 11-layer upstream stack
- Next gains come from eval/export discipline, not architecture resets
- Hybrid recurrence only after legal adaptation path is explored

### Low-Bit Quantization
- Third priority unless genuinely better recipe (ParetoQ, PTQ1.61)
- Naive ternary/INT4 already proven insufficient

## Explicitly Paused
- Full-model recurrence (DDP crashes, gap too large)
- Ternary/BitNet (gap too large)
- Fastfood / block-diagonal MLP (compile/quality issues)
- Old INT5-base ablations (superseded by new SOTA)
- Calling non-record runs "better than leaderboard"
