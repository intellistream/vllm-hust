from __future__ import annotations

import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

ROOT = Path(".cache/ascend_sparse_experiments")
MODEL_DIR = ROOT / "tiny_llama_model"
TEAL_ROOT = ROOT / "teal_tiny_s0.50"
TEAL_HIST = TEAL_ROOT / "official_histograms"
TEAL_THRESH = TEAL_ROOT / "thresholds"
LAROSA_ROOT = ROOT / "larosa_tiny_identity"

VOCAB_TOKENS = [
    "<unk>",
    "<s>",
    "</s>",
    "<pad>",
    "TEAL",
    "La",
    "RoSA",
    "applies",
    "magnitude",
    "based",
    "activation",
    "sparsity",
    "to",
    "transformer",
    "hidden",
    "states",
    "without",
    "retraining",
    "Perplexity",
    "alignment",
    "is",
    "checked",
    "by",
    "scoring",
    "same",
    "target",
    "tokens",
    "under",
    "dense",
    "and",
    "sparse",
    "execution",
    "paths",
    "The",
    "vLLM",
    "path",
    "should",
    "match",
    "HF",
    "reference",
    "once",
    "threshold",
    "mapping",
    "prefill",
    "semantics",
    "are",
    "the",
    "same",
    ".",
    ",",
    "-",
]

SUPPORTED_PROJECTIONS = (
    "self_attn.qkv",
    "self_attn.o",
    "mlp.gate_up",
    "mlp.down",
)


def save_tiny_model() -> LlamaConfig:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    unique_tokens = list(dict.fromkeys(VOCAB_TOKENS))
    vocab = {token: idx for idx, token in enumerate(unique_tokens)}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    fast.save_pretrained(MODEL_DIR)

    torch.manual_seed(1234)
    config = LlamaConfig(
        vocab_size=len(vocab),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        bos_token_id=vocab["<s>"],
        eos_token_id=vocab["</s>"],
        pad_token_id=vocab["<pad>"],
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config)
    model.save_pretrained(MODEL_DIR, safe_serialization=True)
    return config


def save_teal_artifacts(config: LlamaConfig) -> None:
    centers = torch.linspace(0.0, 1.0, 101, dtype=torch.float32)
    counts = torch.ones(101, dtype=torch.float32)
    threshold = torch.tensor(0.75, dtype=torch.float32)

    for layer_idx in range(config.num_hidden_layers):
        for part in ("mlp", "self_attn"):
            path = TEAL_HIST / f"layer-{layer_idx}" / part
            path.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "h1_centers": centers,
                    "h1": counts,
                    "h2_centers": centers,
                    "h2": counts,
                },
                path / "histograms.pt",
            )

        TEAL_THRESH.mkdir(parents=True, exist_ok=True)
        for proj in SUPPORTED_PROJECTIONS:
            torch.save(
                threshold,
                TEAL_THRESH / f"layers.{layer_idx}.{proj}.threshold.pt",
            )

    (TEAL_ROOT / "activation_sparsity_config.json").write_text(
        json.dumps(
            {
                "enable": True,
                "method": "teal",
                "uniform_sparsity": 0.5,
                "calibration_path": str(TEAL_THRESH.resolve()),
                "decode_only": False,
                "apply_all_tokens": False,
                "prefill_sparsify": "half",
                "strict_unsupported_check": True,
                "use_sparse_gemv": False,
            },
            indent=2,
        )
        + "\n"
    )


def save_larosa_artifacts(config: LlamaConfig) -> None:
    for layer_idx in range(config.num_hidden_layers):
        path = LAROSA_ROOT / "histograms" / f"layer-{layer_idx}" / "self_attn"
        path.mkdir(parents=True, exist_ok=True)
        torch.save(torch.eye(config.hidden_size, dtype=torch.float32), path / "D.pt")


def main() -> None:
    config = save_tiny_model()
    save_teal_artifacts(config)
    save_larosa_artifacts(config)
    print(f"MODEL_DIR={MODEL_DIR.resolve()}")
    print(f"TEAL_ROOT={TEAL_ROOT.resolve()}")
    print(f"LAROSA_ROOT={LAROSA_ROOT.resolve()}")


if __name__ == "__main__":
    main()
