# Parameter Golf: Implementation Plan

## Current Best: 1.1441 bpb, 15.77MB (winner port validated on MI250)
## Legal Best with TTT: ~1.13 bpb (estimated, val17 running)

## Guiding Rule

**Stop spending exploration budget on ideas that already lost clearly or cannot be ranked reliably on 1-GPU packed runs.** Every batch must produce actionable signal within the packed-run time budget.

## Immediate Priorities

### 1. Fix Exploration Eval Protocol [CRITICAL]
Winner-family scripts are too slow for full sliding eval on 1 GPU (969K windows). Need a fast proxy:
- Post-quant serialization + artifact bytes (already works)
- Fixed sliding-window subset on a small validation slice (NOT full eval)
- No TTT in packed exploration (too slow, save for 8-GPU validation)
- Explicit `proxy_bpb` logging so `packed-results` can parse modern scripts
- Use relative rankings, not absolute numbers

### 2. Validate Winner Family on 8 GPUs [IN PROGRESS]
- [x] Val 16: exact winner port → **1.1441 bpb** (matches H100 leaderboard)
- [ ] Val 17: winner + TTT LoRA (submitted, job 16955596)
- [ ] Val 18: winner + XSA (after batch 31 shows best XSA depth)

### 3. Eval-Time Adaptation (Legal TTT) [HIGHEST UPSIDE]
The invalid val-only results proved the remaining gain lives in adaptation/memorization. TTT is the legal version:
- [ ] Document-isolated sliding eval on winner backbone
- [ ] TTT sweep on winner backbone (rank, target layers, chunk size, eval context, batch size)
- [ ] Tune under the 10-minute eval budget on 8-GPU
- [ ] TTT validation-only mode (skip exploration, go straight to 8-GPU)

### 4. Winner-Family Exploration [NEXT BATCH]
All 8 runs inside winner family, using fast proxy eval:
- Control (exact winner port)
- XSA last 2/4/5 layers
- No SWA (isolate SWA contribution on MI250)
- Smaller bigram sidecar (BIGRAM_VOCAB_SIZE=8192)
- Smaller bigram projection (BIGRAM_DIM=96)
- 11L funded by byte cuts (reduced sidecar + slightly smaller MLP)

Answers: does XSA help on the right backbone? Is SWA helping on MI250? Should bytes go to bigram sidecar or extra layer?

## Experiment Bookkeeping

Separate three metrics in all logs:
1. **standard post-quant bpb** — the quick roundtrip metric (always available)
2. **sliding post-quant bpb** — stride=64, full val set (slow, 8-GPU only)
3. **TTT bpb** — test-time LoRA fine-tuning (8-GPU only, within 10-min eval budget)

## What's Working (Proven on MI250)

| Technique | Source | Validated bpb | Status |
|-----------|--------|--------------|--------|
| 10L MLP3x + full winner stack | batch 28 / val 16 | **1.1441** | CURRENT BEST |
| 11L MLP3x + INT5 + late QAT + TTT | batch 9 / val 5 | 1.1543 | Previous best |
| Sliding window eval (stride=64) | winner script | Built-in | Needs 8-GPU |
| Mixed INT5/INT6 quantization | winner script | Part of 1.1441 | Proven |
| BigramHash 10240 | winner script | Part of 1.1441 | Proven |
| SWA (start_frac=0.35) | winner script | Part of 1.1441 | Needs isolation |
| XSA (last 4 layers) | PR #349 | 1.1399 on H100 | Testing |

## Explicitly Paused

Do not spend exploration budget on these until winner-family path is stable:
- Full-model recurrence (DDP crashes, +0.028 gap even in exploration)
- Ternary/BitNet QAT (+0.066 gap, naive approach failed)
- Fastfood / block-diagonal / butterfly MLP replacements (torch.compile incompatible or too lossy)
- Old INT5-base ablations (superseded by winner family)
- "Kitchen sink" technique mixing (batch 24 showed bolting-on doesn't work)

## Longer-Term Directions (Reopen After Winner Path Stable)

### Hybrid Recurrence
Strong dense trunk + shared recurrent tail with loop-specific deltas on ALL linear layers (not just Q/V). Not full-model recurrence.

### Memory-Augmented Models
Shared core + fast weights / online memory / legal test-time state. More plausible than plain weight sharing.

### Dynamic-Depth Recurrence
Let only hard tokens take extra loops. Better compute budget usage than looping every token equally.

### Compression-Aware Architecture Search
Keep reallocating bytes between layers, MLP width, bigram sidecar, FP16-sensitive tensors, and pruning. This is where most real gains are coming from.

### Better Low-Bit Quantization
Only with a substantially better training/export pipeline. Naive ternary is not worth more batches.

## Throughput Note

We run **8 experiments per hour** on LUMI (8x 1-GPU packed). Both exploration and validation nodes should run at all times. At this throughput, we can test 50-100 ideas per day. But only if the exploration protocol produces actionable signal — fix the eval proxy first.
