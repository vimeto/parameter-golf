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
- Copies are generated from root, then modified for the specific experiment
- The spec file references each via `TRAIN_SCRIPT=specs/batchN/runM.py`
- This allows 8 completely different code paths to run in parallel
- Validation runs ALSO use `TRAIN_SCRIPT` to point to the winning experiment's script

### Graduating Features
When an experiment wins validation and is confirmed better than global best:
1. Merge the winning changes back into root `train_gpt.py`
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

## LUMI Supercomputer

- GPUs: AMD MI250X (8 GCDs per node, 64GB HBM2e each)
- Partition: `standard-g` (up to 2 days), `dev-g` for quick tests (3h max)
- Account: `project_462001163`
- Code dir on LUMI: `~/parameter-golf`
- Data on scratch: `/scratch/project_462001163/vtoivone/pgolf_cache/`
- Container: `lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif`

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
