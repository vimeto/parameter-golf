# Local Test Skill

## What This Does
Quick MLX smoke tests on M2 Max to validate changes before submitting to LUMI.

## Run a 200-step smoke test

```bash
cd ~/code/omat/parameter-golf
source .venv/bin/activate
RUN_ID=test ITERATIONS=200 TRAIN_BATCH_TOKENS=32768 VAL_LOSS_EVERY=0 \
  VAL_BATCH_SIZE=8192 TRAIN_LOG_EVERY=20 MAX_WALLCLOCK_SECONDS=0 \
  python3 train_gpt_mlx.py
```

## Verify artifact size

```bash
# After training, check compressed model + code fits under 16MB
wc -c records/our_submission/train_gpt.py  # code size
ls -la *.bin 2>/dev/null || ls -la *.pt 2>/dev/null  # model size
# Total (code + int8 zlib compressed model) must be < 16,000,000 bytes
```

## Compare val_bpb between runs

Look at the training loss curve in the output. Lower training loss at 200 steps generally indicates a better configuration, but this is directional only.

## Gotchas

- Local numbers are 2-3x worse than H100 due to undertraining at 200 steps — do NOT optimize for absolute local bpb
- Validation takes ~18min per pass, so always set `VAL_LOSS_EVERY=0` for smoke tests
- Use local runs to catch errors and verify the model trains, not to measure final quality
- If you need actual bpb numbers, submit to LUMI
