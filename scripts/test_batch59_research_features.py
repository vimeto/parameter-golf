from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "specs" / "batch59" / "run_research_families.py"


def load_module():
    spec = importlib.util.spec_from_file_location("batch59_research", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def base_kwargs(family: str) -> dict:
    return dict(
        vocab_size=64,
        num_layers=4,
        model_dim=32,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2.0,
        tie_embeddings=True,
        tied_embed_init_std=0.01,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.0,
        train_seq_len=16,
        mtp_num_heads=0,
        mtp_loss_weight=0.0,
        bigram_vocab_size=0,
        bigram_dim=16,
        xsa_last_n=0,
        rope_dims=8,
        ln_scale=False,
        dtg=False,
        ve_enabled=False,
        ve_dim=8,
        ve_layers="",
        research_family=family,
        hnet_chunk_size=4,
        ut_steps=2,
        random_feature_seed=12345,
    )


def smoke_family(module, family: str) -> None:
    model = module.GPT(**base_kwargs(family)).float()
    x = torch.randint(0, 64, (2, 16), dtype=torch.long)
    y = torch.randint(0, 64, (2, 16), dtype=torch.long)

    if family == "megakernel":
        module.SOFTCAP_CE_FN = torch.compile(module.softcap_cross_entropy, mode="default", fullgraph=False)
    else:
        module.SOFTCAP_CE_FN = module.softcap_cross_entropy

    loss = model(x, y)
    logits = model.forward_logits(x)
    assert torch.isfinite(loss), family
    assert logits.shape == (2, 16, 64), (family, logits.shape)

    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    unbanked = module._unbank_state_dict(sd, model.num_layers)
    quant_result, quant_meta = module.mixed_quantize_int6(unbanked, {"mlp", "attn"}, hessians=None)
    deq_unbanked = module.dequantize_mixed_int6(quant_result, quant_meta, unbanked)
    deq_state = module._rebank_state_dict(deq_unbanked, model.num_layers, sd)

    eval_model = module.GPT(**base_kwargs(family)).float()
    eval_model.load_state_dict(deq_state, strict=True)
    eval_logits = eval_model.forward_logits(x)
    assert eval_logits.shape == (2, 16, 64), (family, eval_logits.shape)
    assert torch.isfinite(eval_logits).all(), family
    print(f"{family}: loss={float(loss):.6f} eval_logits_ok")


def assert_prefix_causal(module, family: str) -> None:
    model = module.GPT(**base_kwargs(family)).float().eval()
    prefix = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    alt = prefix.clone()
    alt[:, 4:] = torch.tensor([[31, 32, 33, 34]], dtype=torch.long)
    with torch.no_grad():
        logits_a = model.forward_logits(prefix)
        logits_b = model.forward_logits(alt)
    assert torch.allclose(logits_a[:, :4], logits_b[:, :4], atol=1e-5, rtol=1e-5), family
    print(f"{family}: prefix_causal_ok")


def main() -> None:
    module = load_module()
    for family in ("control", "hnet", "ssm", "ut", "megakernel", "random"):
        smoke_family(module, family)
        assert_prefix_causal(module, family)


if __name__ == "__main__":
    main()
