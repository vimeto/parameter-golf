# Parameter Golf - Claude Code Guidelines

## Commits

ALWAYS use 1-line commit messages. NO co-authorship lines. Example:
```
git commit -m "feat: add slurm training scripts for LUMI"
```

## Project

OpenAI Parameter Golf challenge: train the best language model (lowest bits-per-byte) that fits in a 16MB artifact and trains in 10 min on 8xH100 SXM.

- Repo: https://github.com/vimeto/parameter-golf (fork of openai/parameter-golf)
- Upstream: https://github.com/openai/parameter-golf
- Research: see `RESEARCH_ANALYSIS.md` in parent dir (`~/code/omat/parameter-golf/`)
- Plan: see `TODO.md` in parent dir

## Key Constraints

- Artifact (code + compressed model) <= 16,000,000 bytes
- Training <= 10 min on 8xH100 SXM
- Evaluation <= 10 min on 8xH100 SXM (separate budget, can train on val data)
- BPB metric on FineWeb validation set

## CRITICAL: Code Architecture

### Root `train_gpt.py` = Proven Shared Infrastructure
- **ONLY contains proven, graduated features** — things every run needs
- ROCm detection + torch.compile, data loading, DDP, base model (GPT/Block/Attention/MLP), Muon optimizer, eval pipeline (BPB, TTT LoRA), INT8 serialization, logging
- Features only get merged into root AFTER validation proves they help
- `records/our_submission/` is the frozen upstream-compatible snapshot — do NOT modify it

### Per-Experiment Scripts = Exploration
- Each exploration run gets its OWN `train_gpt.py` copy under `specs/batchN/`
- The current backbone is `specs/batch28/run_winner_rocm.py` (ported H100 leaderboard winner). **All new experiments should build from the winner family, not the stale root.**
- The spec file references each via `TRAIN_SCRIPT=specs/batchN/runM.py`
- Validation runs ALSO use `TRAIN_SCRIPT` to point to the winning experiment's script
- IF something promising hasn't worked after 2 iterations, stop and research SOTA before trying again. Use a subagent to read papers/repos (clone to /tmp).

### Exploration Eval Protocol
Winner-family scripts have slow sliding eval (969K windows on 1 GPU). Use a fast proxy:
- Post-quant serialization + artifact bytes (always report)
- Standard (non-sliding) roundtrip bpb as the fast proxy metric
- NO TTT in packed exploration (too slow — reserve for 8-GPU validation)
- NO full sliding eval in packed exploration (too slow — use for 8-GPU only)
- Use relative rankings between runs, not absolute bpb numbers
- Separate three metrics in logs: `standard_postquant_bpb`, `sliding_bpb`, `ttt_bpb`

### TTT (Test-Time Training) — Proven Findings
- **Only LoRA TTT works.** Full fine-tuning TTT (SGD or AdamW, any LR, any freeze config) causes catastrophic forgetting and diverges on both Full GPTQ and GPTQ-lite models.
- LoRA TTT (rank-8 on Q/V, frozen base weights) keeps avg_loss stable near baseline (~2.76 vs 2.73 roundtrip).
- Score-first protocol: score chunk in `inference_mode()`, then train. Chunk N scored by model adapted on chunks 0..N-1.
- **Full GPTQ >> GPTQ-lite** for quantization: ~0.12 BPB gap (1.61 vs 1.73 on 1-GPU). Always use Full GPTQ.
- **Next step:** LoRA TTT on Full GPTQ model (combining best quant + best TTT).

### Exploration GPTQ Note
- Full Hessian GPTQ is too slow for 1-GPU packed exploration (256-batch Hessian collection takes ~32 min, quantization another ~30 min). For packed exploration: use GPTQ-lite or reduce calibration batches to 64.

### Validation Priority
1. Don't wait for noisy 1-GPU results to block validation — submit promising configs immediately
2. Don't promote old-base runs before winner-family runs
3. LoRA TTT is the viable eval-time adaptation path — full fine-tuning TTT diverges
4. Val-only training during training phase is AGAINST THE RULES — only TTT during eval

### Graduating Features
When an experiment wins validation and is confirmed better than global best:
1. Merge the winning changes back into the winner backbone script
2. Commit with a clear message about what was proven
3. All future experiment copies inherit the improvement

## Exploration Batch Generation (4-Step Process)

### Step 1: Data-Driven Analysis
- Read `experiments/state.json`, exploration/validation logs, `TODO.md`
- Identify what worked, what failed, what's untested
- Determine the highest-priority research direction

### Step 2: Conceptual Run List
- Design 8 experiments: 1 control (reproduce best known), 6 variations, 1 wildcard
- Each run has a clear hypothesis and expected outcome
- Document the rationale before writing any code

### Step 3: Generate Scripts via Subagents
- Launch subagents in parallel (one per script or batched)
- Each subagent: copies root `train_gpt.py` → applies the specific modification
- Scripts saved to `specs/batchN/runM.py`

### Step 4: Validate and Submit
- `python3 -c "import ast; ast.parse(open('specs/batchN/runM.py').read())"` for each
- Create spec file `specs/batchN/spec.txt` referencing all 8 scripts
- `git add`, `git commit`, `scripts/lumi.sh sync`
- `scripts/lumi.sh submit scripts/slurm/lumi_packed.sh specs/batchN/spec.txt`

## Compute Resources

### LUMI (MI250X) — Full-Node Allocation
- GPUs: AMD MI250X (8 GCDs per node, 64GB HBM2e each)
- Partition: `standard-g` (up to 2 days), `dev-g` for quick tests (3h max)
- Account: `project_462001163`
- Code dir: `~/parameter-golf`
- Data: `/scratch/project_462001163/vtoivone/pgolf_cache/`
- Container: `lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif`
- **Allocation model:** Reserving ANY GPU reserves the full 8-GPU node. Always use packed 8x1-GPU runs for exploration.
- **Exploration:** `lumi_packed.sh` — 8 parallel 1-GPU experiments per node (50 min each)
- **Validation:** `lumi_train.sh` — single 8-GPU run per node (2700s wallclock)
- Submit via: `scripts/lumi.sh submit <script> [args]`

### Mahti (A100) — Individual GPU Jobs
- GPUs: NVIDIA A100 (4 per node, 40GB each)
- Partition: `gpusmall` (1-2 GPUs, less congested) or `gpumedium` (3-4 GPUs, very congested)
- Account: `project_2013932`
- Code dir: `/scratch/project_2013932/vtoivone/pgolf`
- Data: `./data/datasets/fineweb10B_sp1024` (default relative paths work)
- SLURM script (on Mahti): `mahti_1gpu.sh` — uses `module load pytorch/2.9`, no container
- **Allocation model:** Full nodes are very congested. Submit individual 1-GPU jobs via `gpusmall` partition — they queue independently and start faster. Good for TTT sweeps where many configs run in parallel across separate queue slots.
- **Submission pattern:** `ssh mahti "cd /scratch/project_2013932/vtoivone/pgolf && sbatch --export=ALL,TRAIN_SCRIPT=specs/...,RUN_ID=...,MAX_WALLCLOCK_SECONDS=... --time=HH:MM:SS mahti_1gpu.sh"`
- **Sync:** `ssh mahti "cd /scratch/project_2013932/vtoivone/pgolf && git pull"`
- **Validation:** Not ideal (max 4 GPUs, congested). Prefer LUMI for 8-GPU validation.

### Cluster Strategy
- **LUMI** = packed exploration batches (8 runs/node) + 8-GPU validation
- **Mahti** = individual exploration runs (1 GPU each, queue independently) — best for sweeps with many configs
- **Both clusters run simultaneously** for maximum throughput

## LUMI Training Guidelines

- **ALWAYS use torch.compile** — enabled via `shape_padding=False` + `fullgraph=False` in our ROCm fork. Gives ~2x speedup over eager mode.
- **Token-parity with 8xH100 requires ~41 min on 8xMI250** — the benchmark processes 6.8B tokens in 10 min on 8xH100 (11.4M tok/s). Our 8xMI250 with compile does 2.80M tok/s (187ms/step with 524K batch), so set `MAX_WALLCLOCK_SECONDS=2500` (~42 min) for comparable runs. We are 4.1x slower than 8xH100.
- For quick exploration sweeps, use `lumi_packed.sh` with 8x 1-GPU runs (50 min each, ~360M tokens per run — directional signal only).
- For proper validation runs, use `lumi_train.sh` with `TRAIN_SCRIPT=specs/batchN/runM.py NUM_GPUS=8`.
- `dev-g` has faster queue times than `standard-g` — use for quick tests.

## Autoresearch Loop

Two-tier continuous experimentation — **both nodes must be running at all times**:
- **Node A (exploration):** 8x 1-GPU packed runs, 50min train + 10min eval. Each run has its own train_gpt.py.
- **Node B (validation):** 1x 8-GPU full node, 45min train (2700s). Uses the winning experiment's script via TRAIN_SCRIPT.

State tracked in:
- `experiments/state.json` — current phase, best scores, running jobs
- `experiments/exploration_log.md` — all exploration batches with results
- `experiments/validation_log.md` — all validation runs with comparison to global best

See `.claude/skills/experiment/SKILL.md` for the full protocol.

## Local Testing (M2 Max, 64GB)

```bash
cd ~/code/omat/parameter-golf
source .venv/bin/activate
RUN_ID=test ITERATIONS=200 TRAIN_BATCH_TOKENS=32768 VAL_LOSS_EVERY=0 \
  VAL_BATCH_SIZE=8192 TRAIN_LOG_EVERY=20 MAX_WALLCLOCK_SECONDS=0 \
  python3 train_gpt_mlx.py
```

Local numbers are ~2-3x worse than H100 due to undertraining. Use for directional comparison only.

## File Structure

- `train_gpt.py` — Base training script (ROCm-compatible, proven features only)
- `train_gpt_mlx.py` — MLX variant for local testing
- `records/our_submission/` — Frozen upstream-compatible submission (do NOT modify)
- `specs/batchN/` — Per-batch experiment scripts and spec files
- `scripts/lumi.sh` — LUMI SSH bridge
- `scripts/slurm/` — SLURM job scripts
- `experiments/` — State, logs, results (gitignored, local tracking)

## Skills

- **lumi-submit** (`.claude/skills/lumi-submit/SKILL.md`) — Submit and manage jobs on LUMI supercomputer
- **experiment** (`.claude/skills/experiment/SKILL.md`) — Autoresearch experiment loop
- **local-test** (`.claude/skills/local-test/SKILL.md`) — Quick MLX smoke tests on M2 Max
