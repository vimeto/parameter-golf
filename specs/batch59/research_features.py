from __future__ import annotations

import importlib.util
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _parse_int_list(raw: str, default: list[int]) -> list[int]:
    items = [part.strip() for part in raw.split(",") if part.strip()]
    if not items:
        return default
    return [int(part) for part in items]


def _build_fixed_random(shape: tuple[int, ...], seed: int) -> Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    out = torch.randn(shape, generator=gen, dtype=torch.float32)
    if len(shape) == 2:
        out = out / math.sqrt(shape[1])
    return out


def _load_base_module(module_name: str | None = None):
    base_path = Path(__file__).resolve().parents[1] / "batch55" / "run_nomtp.py"
    module_name = module_name or f"pgolf_batch55_{os.getpid()}_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, base_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load base module from {base_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_and_patch_base(module_name: str | None = None):
    base = _load_base_module(module_name)
    apply_research_patch(base)
    return base


def apply_research_patch(base) -> None:
    idea = os.environ.get("RESEARCH_IDEA", "control").strip().lower()
    allowed = {"control", "hnet", "ssm", "ut", "megakernel", "random"}
    if idea not in allowed:
        raise ValueError(f"Unknown RESEARCH_IDEA={idea!r}; expected one of {sorted(allowed)}")

    def research_log(msg: str) -> None:
        if not base.dist.is_available() or not base.dist.is_initialized() or base.dist.get_rank() == 0:
            print(msg, flush=True)

    class ChunkLift(nn.Module):
        def __init__(self, dim: int, chunk_size: int):
            super().__init__()
            if chunk_size < 2:
                raise ValueError(f"HNET chunk_size must be >= 2, got {chunk_size}")
            self.chunk_size = chunk_size
            self.gate = nn.Parameter(torch.full((dim,), -2.0, dtype=torch.float32))
            self.scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

        def forward(self, x: Tensor) -> Tensor:
            bsz, seq_len, dim = x.shape
            pad = (-seq_len) % self.chunk_size
            if pad:
                x_pad = F.pad(x, (0, 0, 0, pad))
            else:
                x_pad = x
            coarse = x_pad.reshape(bsz, -1, self.chunk_size, dim).mean(dim=2)
            coarse_prev = torch.cat([torch.zeros_like(coarse[:, :1]), coarse[:, :-1]], dim=1)
            coarse = 0.5 * coarse + 0.5 * coarse_prev
            up = coarse.repeat_interleave(self.chunk_size, dim=1)[:, :seq_len]
            gate = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
            return up * gate * self.scale.to(dtype=x.dtype)

    class CumulativeMixer(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.gate = nn.Parameter(torch.full((dim,), -2.5, dtype=torch.float32))
            self.scale = nn.Parameter(torch.tensor(0.08, dtype=torch.float32))

        def forward(self, x: Tensor) -> Tensor:
            csum = torch.cumsum(x.float(), dim=1)
            denom = torch.arange(1, x.size(1) + 1, device=x.device, dtype=torch.float32)[None, :, None]
            mean = (csum / denom).to(dtype=x.dtype)
            mean_prev = torch.cat([torch.zeros_like(mean[:, :1]), mean[:, :-1]], dim=1)
            mixed = 0.5 * mean + 0.5 * mean_prev
            gate = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
            return mixed * gate * self.scale.to(dtype=x.dtype)

    class RandomFeatureResidual(nn.Module):
        def __init__(self, dim: int, feature_dim: int, seed: int):
            super().__init__()
            self.register_buffer("proj_in", _build_fixed_random((feature_dim, dim), seed), persistent=False)
            self.register_buffer("proj_out", _build_fixed_random((dim, feature_dim), seed + 1), persistent=False)
            self.feature_gate = nn.Parameter(torch.full((feature_dim,), 0.10, dtype=torch.float32))
            self.channel_gate = nn.Parameter(torch.full((dim,), -2.0, dtype=torch.float32))
            self.scale = nn.Parameter(torch.tensor(0.06, dtype=torch.float32))

        def forward(self, x: Tensor) -> Tensor:
            z = F.linear(x.float(), self.proj_in)
            z = torch.tanh(z * self.feature_gate.to(dtype=z.dtype)[None, None, :])
            out = F.linear(z, self.proj_out).to(dtype=x.dtype)
            gate = torch.sigmoid(self.channel_gate.to(dtype=x.dtype))[None, None, :]
            return out * gate * self.scale.to(dtype=x.dtype)

    class UTRefiner(nn.Module):
        def __init__(self, dim: int, num_steps: int):
            super().__init__()
            if num_steps <= 0:
                raise ValueError(f"UT num_steps must be > 0, got {num_steps}")
            self.step_bias = nn.ParameterList(
                [nn.Parameter(torch.zeros(dim, dtype=torch.float32)) for _ in range(num_steps)]
            )
            self.step_scale = nn.ParameterList(
                [nn.Parameter(torch.full((dim,), 0.10, dtype=torch.float32)) for _ in range(num_steps)]
            )
            self.threshold = nn.Parameter(torch.zeros(num_steps, dtype=torch.float32))
            self.temperature = nn.Parameter(torch.full((num_steps,), 4.0, dtype=torch.float32))

        def apply(self, x: Tensor, step_fn) -> Tensor:
            for idx, bias in enumerate(self.step_bias):
                x_in = x + bias.to(dtype=x.dtype)[None, None, :]
                candidate = step_fn(x_in)
                stat = x.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
                gate = torch.sigmoid(
                    self.temperature[idx].to(dtype=stat.dtype) * (stat - self.threshold[idx].to(dtype=stat.dtype))
                ).to(dtype=x.dtype)
                delta = candidate - x
                scale = torch.tanh(self.step_scale[idx].to(dtype=x.dtype))[None, None, :]
                x = x + gate * scale * delta
            return x

    def softcap_cross_entropy(logits_proj: Tensor, targets: Tensor, softcap: float) -> Tensor:
        logits = softcap * torch.tanh(logits_proj / softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    class ResearchGPT(base.GPT):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.research_idea = idea
            self.use_hnet = idea == "hnet"
            self.use_ssm = idea == "ssm"
            self.use_ut = idea == "ut"
            self.use_mega_loss = idea == "megakernel"
            self.use_random = idea == "random"
            self.hnet_mid_idx = None

            if self.use_hnet:
                chunk_size = int(os.environ.get("HNET_CHUNK_SIZE", "4"))
                self.blocks[0].research_hnet_pre = ChunkLift(self.tok_emb.embedding_dim, chunk_size)
                self.hnet_mid_idx = max(self.num_encoder_layers - 1, 0)
                self.blocks[self.hnet_mid_idx].research_hnet_mid = ChunkLift(self.tok_emb.embedding_dim, chunk_size)
                research_log(
                    f"research_feature:hnet chunk_size={chunk_size} pre_layer=0 mid_layer={self.hnet_mid_idx}"
                )

            if self.use_ssm:
                ssm_layers = _parse_int_list(
                    os.environ.get("SSM_LAYER_INDICES", f"{max(self.num_layers - 3, 0)},{max(self.num_layers - 2, 0)},{self.num_layers - 1}"),
                    list(range(max(self.num_layers - 3, 0), self.num_layers)),
                )
                for idx in ssm_layers:
                    if idx < 0 or idx >= self.num_layers:
                        raise ValueError(f"SSM layer index {idx} out of range for num_layers={self.num_layers}")
                    self.blocks[idx].research_ssm = CumulativeMixer(self.tok_emb.embedding_dim)
                research_log(f"research_feature:ssm cumulative_mixer_layers={ssm_layers}")

            if self.use_ut:
                ut_steps = int(os.environ.get("UT_EXTRA_STEPS", "2"))
                self.blocks[-1].research_ut = UTRefiner(self.tok_emb.embedding_dim, ut_steps)
                research_log(f"research_feature:ut extra_steps={ut_steps} shared_block={self.num_layers - 1}")

            if self.use_random:
                feature_dim = int(os.environ.get("RANDOM_FEATURE_DIM", "96"))
                random_layers = _parse_int_list(
                    os.environ.get("RANDOM_LAYER_INDICES", f"{max(self.num_layers - 3, 0)},{max(self.num_layers - 2, 0)},{self.num_layers - 1}"),
                    list(range(max(self.num_layers - 3, 0), self.num_layers)),
                )
                seed = int(os.environ.get("RANDOM_SEED", "1234"))
                for offset, idx in enumerate(random_layers):
                    if idx < 0 or idx >= self.num_layers:
                        raise ValueError(f"Random layer index {idx} out of range for num_layers={self.num_layers}")
                    self.blocks[idx].research_random = RandomFeatureResidual(
                        self.tok_emb.embedding_dim, feature_dim, seed + offset * 17
                    )
                research_log(
                    f"research_feature:random feature_dim={feature_dim} layers={random_layers} seed={seed}"
                )

            self._softcap_ce = softcap_cross_entropy
            if self.use_mega_loss and torch.cuda.is_available():
                self._softcap_ce = torch.compile(softcap_cross_entropy, mode="default", fullgraph=False)
                mega_mode = "compiled"
            elif self.use_mega_loss:
                mega_mode = "eager_cpu_fallback"
            else:
                mega_mode = "disabled"
            if self.use_mega_loss:
                research_log(f"research_feature:megakernel softcap_ce_mode={mega_mode}")
            if not any((self.use_hnet, self.use_ssm, self.use_ut, self.use_mega_loss, self.use_random)):
                research_log("research_feature:control wrapper_active=True")

        def _apply_research_residual(self, layer_idx: int, x: Tensor) -> Tensor:
            block = self.blocks[layer_idx]
            if hasattr(block, "research_ssm"):
                x = x + block.research_ssm(x)
            if hasattr(block, "research_random"):
                x = x + block.research_random(x)
            if self.use_hnet and self.hnet_mid_idx == layer_idx and hasattr(block, "research_hnet_mid"):
                x = x + block.research_hnet_mid(x)
            return x

        def _run_hidden(self, input_ids: Tensor) -> tuple[Tensor, Tensor, dict]:
            n = self.num_layers
            x = self.tok_emb(input_ids)
            if self.bigram is not None:
                x = x + self.bigram(input_ids)
            x = F.rms_norm(x, (x.size(-1),))
            if self.use_hnet and hasattr(self.blocks[0], "research_hnet_pre"):
                x = x + self.blocks[0].research_hnet_pre(x)
            x = self.smear(x)
            x0 = x
            skips: list[Tensor] = []
            ve_cache: dict = {}
            for i in range(self.num_encoder_layers):
                ve = self._get_ve(i, input_ids, ve_cache)
                x = self.blocks[i](
                    x,
                    x0,
                    self.qo_bank[i],
                    self.kv_bank[i],
                    self.kv_bank[n + i],
                    self.qo_bank[n + i],
                    self.mlp_up_bank[i],
                    self.mlp_down_bank[i],
                    v_embed=ve,
                )
                x = self._apply_research_residual(i, x)
                skips.append(x)
            for i in range(self.num_decoder_layers):
                bi = self.num_encoder_layers + i
                if skips:
                    x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
                ve = self._get_ve(bi, input_ids, ve_cache)
                x = self.blocks[bi](
                    x,
                    x0,
                    self.qo_bank[bi],
                    self.kv_bank[bi],
                    self.kv_bank[n + bi],
                    self.qo_bank[n + bi],
                    self.mlp_up_bank[bi],
                    self.mlp_down_bank[bi],
                    v_embed=ve,
                )
                x = self._apply_research_residual(bi, x)
            if self.use_ut and hasattr(self.blocks[-1], "research_ut"):
                bi = self.num_layers - 1

                def step_fn(x_in: Tensor) -> Tensor:
                    ve = self._get_ve(bi, input_ids, ve_cache)
                    return self.blocks[bi](
                        x_in,
                        x0,
                        self.qo_bank[bi],
                        self.kv_bank[bi],
                        self.kv_bank[n + bi],
                        self.qo_bank[n + bi],
                        self.mlp_up_bank[bi],
                        self.mlp_down_bank[bi],
                        v_embed=ve,
                    )

                x = self.blocks[-1].research_ut.apply(x, step_fn)
            return self.final_norm(x), x0, ve_cache

        def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
            x, _, _ = self._run_hidden(input_ids)
            x_flat = x.reshape(-1, x.size(-1))
            targets = target_ids.reshape(-1)
            if self.tie_embeddings:
                logits_proj = F.linear(x_flat, self.tok_emb.weight)
            else:
                if self.lm_head is None:
                    raise RuntimeError("lm_head is required when tie_embeddings=False")
                logits_proj = self.lm_head(x_flat)
            if self.use_mega_loss:
                main_loss = self._softcap_ce(logits_proj, targets, self.logit_softcap)
            else:
                logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
                main_loss = F.cross_entropy(logits.float(), targets, reduction="mean")
            if self.training and self.mtp_num_heads > 0 and self.mtp_loss_weight > 0.0:
                _, seqlen, dim = x.shape
                mtp_loss_sum = x.new_zeros(())
                mtp_loss_count = 0
                for k, mtp_head in enumerate(self.mtp_heads):
                    valid_t = seqlen - (k + 1)
                    if valid_t <= 0:
                        continue
                    mtp_hidden = x[:, :valid_t, :].reshape(-1, dim)
                    mtp_targets = target_ids[:, k + 1 :].reshape(-1)
                    mtp_logits_proj = mtp_head(mtp_hidden)
                    if self.use_mega_loss:
                        mtp_loss_sum = mtp_loss_sum + self._softcap_ce(
                            mtp_logits_proj, mtp_targets, self.logit_softcap
                        )
                    else:
                        mtp_logits = self.logit_softcap * torch.tanh(mtp_logits_proj / self.logit_softcap)
                        mtp_loss_sum = mtp_loss_sum + F.cross_entropy(
                            mtp_logits.float(), mtp_targets, reduction="mean"
                        )
                    mtp_loss_count += 1
                if mtp_loss_count > 0:
                    main_loss = main_loss + self.mtp_loss_weight * (mtp_loss_sum / mtp_loss_count)
            if not torch.isfinite(main_loss):
                raise RuntimeError(f"Non-finite loss in research idea {self.research_idea}")
            return main_loss

        def forward_logits(self, input_ids: Tensor) -> Tensor:
            x, _, _ = self._run_hidden(input_ids)
            x_flat = x.reshape(-1, x.size(-1))
            if self.tie_embeddings:
                logits_proj = F.linear(x_flat, self.tok_emb.weight)
            else:
                if self.lm_head is None:
                    raise RuntimeError("lm_head is required when tie_embeddings=False")
                logits_proj = self.lm_head(x_flat)
            logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
            return logits.reshape(input_ids.size(0), input_ids.size(1), -1)

    base.GPT = ResearchGPT
