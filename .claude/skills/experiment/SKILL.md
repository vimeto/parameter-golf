# Autoresearch Experiment Skill

## What This Does
Continuous two-tier experiment loop: quick exploration sweeps on packed 1-GPU nodes, validated by full 8-GPU runs at token-parity with the competition benchmark.

## Two-Tier Architecture

### Node A — Exploration (quick screening)
- Platform: 8x 1-GPU MI250X packed into one node, 50 min training + 10 min eval
- Token budget: ~360M tokens per run (directional signal only)
- Submit: `scripts/lumi.sh submit scripts/slurm/lumi_packed.sh specs/<spec>.txt`
- Purpose: screen 8 ideas in parallel, rank by bpb, pick winner for validation

### Node B — Validation (real numbers)
- Platform: 1x 8-GPU MI250X full node, 45 min training (MAX_WALLCLOCK_SECONDS=2700)
- Token budget: ~6.8B tokens (token-parity with 8xH100 benchmark)
- Submit: `scripts/lumi.sh submit scripts/slurm/lumi_train.sh` (with NUM_GPUS=8)
- Purpose: confirm exploration winners at full scale, produce real bpb numbers

Both tiers run in parallel continuously. While validation runs, the next exploration batch is already submitted.

## The Main Loop

```
1. Read experiments/state.json for current state
2. Check running jobs: scripts/lumi.sh status
3. Process completed exploration batch:
   a. Run scripts/analyze_exploration.sh <job_id> for compact results
   b. Verify integrity of each run (check for required log lines)
   c. Rank runs, identify failures, write to experiments/exploration_log.md
   d. Pick best VALID run for promotion (or runner-up from prior batch if nothing good)
   e. Submit validation run on Node B
4. Process completed validation run:
   a. Parse results, compare to global best
   b. Write to experiments/validation_log.md
   c. If new best: update experiments/best/ and state.json
   d. If disappointing: note why, consider runner-ups
5. Plan next exploration batch:
   a. Check TODO.md for current phase priorities
   b. If code change needed: implement, local test, commit, sync
   c. Generate spec file with 8 experiments
   d. Submit exploration on Node A
6. Update experiments/state.json with new job IDs and counters
```

## Rules

- ONLY modify `records/our_submission/train_gpt.py` — NEVER modify evaluation logic or BPB calculation
- Follow research directions in `~/code/omat/parameter-golf/TODO.md`
- Always check **post-quantization** bpb, not pre-quant
- MI250X exploration results are directional — validation at full scale is authoritative
- Use 1-line commit messages, NO co-authorship

## Run Integrity Framework

Every feature flag needs a corresponding integrity check — a specific log line that proves the feature is active. If the required log line is MISSING for an enabled feature, the run is INVALID.

| Feature | Env var | Required log line |
|---------|---------|-------------------|
| Int6 QAT | USE_INT6_QAT=1 | `qat_mode:int6 fake_quant_active:True` |
| Late QAT | USE_LATE_QAT=1 | `late_qat_switchover_step:NNNN` |
| XSA | USE_XSA=1 | `xsa_layers:[N,N,N] projection_dim:NN` |
| BigramHash | USE_BIGRAM_HASH=1 | `bigram_hash_table_size:NNNN` |
| SmearGate | USE_SMEAR_GATE=1 | `smear_gate:enabled` |
| EMA | USE_EMA=1 | `ema_decay:0.NNN ema_active:True` |
| Sliding eval | USE_SLIDING_EVAL=1 | `sliding_window_stride:NN windows_evaluated:NNNN` |
| Partial RoPE | ROPE_PARTIAL_DIMS=N | `rope_dims:N/total` |
| LN Scale | USE_LN_SCALE=1 | `ln_scale:1/sqrt(layer+1)` |

### Integrity Verification Process
For each run in a completed batch:
1. Check which env vars were set in the spec
2. For each enabled feature, grep the run's log for the required log line
3. If ANY required log line is missing: mark run as `RERUN(batch_NNN) — integrity`
4. Include failed runs in the next exploration batch for re-testing

## Startup Crash Policy

- **1 run crashes at startup**: let other 7 continue, note failed run for next batch
- **2+ runs crash at startup**: cancel node (`scripts/lumi.sh cancel <job_id>`), fix bug, resubmit all 8
- **Run diverges mid-training**: let it run, data is still informative (tells us approach is unstable)

## Exploration Logging Format

Each batch entry in `experiments/exploration_log.md`:

```
## Batch N (job XXXXX) — Phase P: Description
Platform: 8x 1-GPU MI250X packed, 50 min training

| rank | run_id | bpb | artifact | integrity | verdict |
|------|--------|-----|----------|-----------|---------|
| 1 | ... | ... | ... | ✓/✗ reason | PROMOTED → val_NNN / runner-up / RERUN / valid but worse |
...

### Promotion Decision
Promoted: <run_id> (rank N)
- Why: <specific reasoning, comparison to runner-ups>
- Confidence: HIGH/MEDIUM/LOW — <why>
- Risk: <what could go wrong at full scale>

Runner-ups available:
- <run_id> (rank N): <why it's a reasonable alternative>

### Key Learnings
- <what we learned from this batch>
```

When a runner-up is promoted later, update the original batch entry:
`| 2 | some_run | 1.28 | 15MB | ✓ | runner-up → PROMOTED(batch_12) as runner-up from batch_8 |`

## Validation Logging Format

Each entry in `experiments/validation_log.md`:

```
## Validation N (job XXXXX) — <run_id> (from exploration batch M)
Config: <full env vars>
Result: **X.XXXX bpb** | artifact XX.XMB | N steps in Ns
Integrity: ✓/✗
vs global best: X.XXXX → X.XXXX = +/-0.XXXX [NEW BEST / worse]
vs exploration: X.XX (1-GPU) → X.XXXX (8-GPU) — exploration was directional ✓/overestimated/underestimated
```

## Phase Advancement

- Phase complete when: 3+ validation batches show <0.005 bpb improvement, OR all TODO items tried
- Current phase TODO items tracked in `~/code/omat/parameter-golf/TODO.md`
- Advance to next phase, carry forward best config as new baseline
- Update `experiments/state.json` with new phase number

## Uncertain Category

Some techniques (late QAT, progressive training, curriculum learning) may only show benefit at full token budget. These should be:
- Marked as "uncertain" in exploration results
- NOT auto-promoted to validation
- Flagged for human review
- Tested only in validation-length runs where the technique can be properly evaluated

Example: late QAT switches quantization on mid-training — testing this in a 50-min exploration run (where "late" might be only 5 min) gives misleading signal. Test late QAT only in validation runs.

## Recording Results

After each validation run, also append to `experiments/results.tsv`:
```
exp_id	date	description	val_bpb	artifact_bytes	notes
```

## Snapshotting Improved Runs

When a validation run produces a new global best with a valid artifact:
1. Create `experiments/runs/<exp_id>/`
2. Copy `train_gpt.py` and the compressed artifact into the snapshot dir
3. Update `experiments/results.tsv` and `experiments/state.json`
4. Update `experiments/best.txt` with the new best bpb
