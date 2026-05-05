# Record candidate: PR #1797 + AWQ-lite top3 + LQER 60k on b180-tlr56 — val_bpb 1.06043 (seed=0)

Builds on the b180-tlr56 lineage (PR #1935) by stacking PR #1908's AWQ-lite mixed-precision GPTQ on top, plus a small LQER budget bump that uses cap margin freed by AWQ-lite's INT8 promotions.

Single seed (SEED=0): val_bpb 1.06043, val_loss 2.32062 nats. Beats PR #1855's 3-seed mean (1.06108 BPB, 2.32203 nats) by -0.00065 BPB / -0.00141 nats. Eval 599.3s, train 596.2s, both inside the 600s lane caps. Per-group lrzip artifact 15,947,372 bytes; total submission 15,982,182 (cap margin 17,818).

i'm running additional seeds at this configuration on RunPod 8xH100 right now and will append SEED=314 and SEED=1234 numbers as soon as they finish. some of those runs are exploring slightly different hparam neighborhoods (LQER budget, AWQ top_k) to map the local landscape — the headline single-seed value is from the configuration documented below.

This PR consolidates the test-plan items left open in PR #1935 (which promised SEED=0 / SEED=1234 multi-seed reference logs but couldn't complete them due to a pod crash on the original session). PR #1935 is being closed.

## Recipe (vs PR #1855)

| Lever | PR #1855 | This | Source |
|---|---|---|---|
| QK_GAIN_INIT | 5.25 | 6.0 | b180 lineage tuning |
| TTT_LORA_RANK | 80 | 56 | rank ablation in PR #1935 (inverted-U at 56) |
| TTT_MLP_LORA / K / O | 1 / 1 / 1 | 0 / 1 / 1 | drop M only |
| LQER_BUDGET_BYTES | 80000 | 60000 | uses cap margin freed by AWQ |
| AWQ_LITE_ENABLED | off | 1 | PR #1908 lineage |
| AWQ_LITE_GROUP_TOP_K | n/a | 3 | top-3 saliency 64-col groups → INT8 |
| AWQ_LITE_SKIP_EMBED | n/a | 1 | tok_emb stays at INT7 |
| COMPRESSOR | pergroup | pergroup_lrzip | helper port from PR #1855 lineage |

All other knobs identical to PR #1855.

## Result

| Seed | Steps | ms/step | Train ms | Pre-quant BPB | Quant BPB | Post-TTT BPB | TTT eval s | Artifact bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4831 | 123.4 | 596,153 | 1.06496396 | 1.07344835 | 1.06043138 | 599.3 | 15,947,372 |

Total submission 15,982,182 / 16,000,000 = 99.89%, cap margin 17,818 bytes.

## Why this should beat PR #1855's mean

AWQ-lite top_k=3 with skip_embed=1 promotes the 3 highest-saliency 64-column groups per non-embedding tensor to INT8 within the same Hessian-based GPTQ solve. Saliency is `act_rms × |w|`, computed during the same calibration pass as the GPTQ Hessians. Skip-embed keeps tok_emb at INT7 to avoid bloating the artifact. Adds ~30s of GPTQ time inside `GPTQ_RESERVE_SECONDS`; no eval-time cost.

LQER budget 80k → 60k: AWQ-lite already reduces post-quant residual error on the top-K groups, so LQER's asymmetric rank-r correction has a smaller residual to capture. 60k bytes covers 15 tensors (vs 9 at 40k).

`TORCHINDUCTOR_CACHE_DIR` on the persistent volume amortizes torch.compile across multi-seed runs (saves ~57s/run on TTT compile warmup, ~20s/run on phase-1 first-call cost in-timer). Doesn't change BPB.

## Compliance

Inherits from PR #1855 / PR #1797 lineage. AWQ-lite runs in the same calibration-data pass as GPTQ — training shards only — and writes its INT8-promoted column metadata at quantize time. No validation data is touched.

C1 causality, C2 normalization, C3 score-first TTT, C4 single L→R pass: all preserved. CaseOps byte sidecar accounting (PR #1729 / #1736) preserved; `ZERO_PUE_MARKERS=1` ships.

## Test plan

- [x] SEED=0 full retrain on H100
- [x] Per-group lrzip artifact roundtrip verified lossless
- [x] Total submission ≤ 16 MB
- [ ] **SEED=314** at this configuration: running on RunPod 8xH100, will append
- [ ] **SEED=1234** at this configuration: running on RunPod 8xH100, will append

## Hardware

8x H100 80GB HBM3 SXM (RunPod), `vimetoivonen/pgolf:b180-tlr64` image, torch 2.9.1+cu129, FA3 via `flash_attn_interface`.

## Lineage and credits

- PR #1855 (merged) — base recipe and 9-hparam stack
- PR #1797 — SparseAttnGate + LQER asymmetric rank-r
- PR #1935 (closed; this PR continues that work) — QK_GAIN=6.0 + TTT_LORA_RANK=56 + drop-M LoRA
- PR #1908 — AWQ-lite mixed-precision GPTQ
- PR #1729 / #1736 — CaseOps lossless tokenizer + byte sidecar

## Files

- `train_gpt.py` — full training/eval script, includes per-group lrzip serialize/deserialize ports from PR #1855 lineage helpers (PGRP magic + LRZI byte autodetect on read), AWQ-lite encoder/decoder paths
- `submission.json` — recipe + per-seed metadata
- `train_seed0.log` — full training + TTT eval log for SEED=0
- `lossless_caps.py`, `prepare_caseops_data.py`, `requirements.txt`, `tokenizers/`
