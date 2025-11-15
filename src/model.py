# src/model.py
"""Model & tokenizer utilities (4-bit QLoRA ready)."""

from __future__ import annotations

from typing import Tuple

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

CACHE_DIR = ".cache/"


def build_model_and_tokenizer(cfg) -> Tuple[torch.nn.Module, AutoTokenizer]:
    model_name: str = cfg.model.name

    # tokenizer --------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    # quantisation / dtype ---------------------------------------------------
    quant = str(cfg.model.get("quantization", "none"))
    kwargs = {"cache_dir": CACHE_DIR, "device_map": "auto"}
    if quant == "4bit":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg.model.precision == "bf16" else torch.float16,
        )
        kwargs["quantization_config"] = bnb_cfg
    else:
        dtype = torch.bfloat16 if cfg.model.precision == "bf16" else torch.float16
        kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    # LoRA -------------------------------------------------------------------
    if "lora" in cfg.model:
        l_cfg = cfg.model.lora
        lora_cfg = LoraConfig(
            r=l_cfg.r,
            lora_alpha=l_cfg.alpha,
            target_modules=list(l_cfg.target_modules),
            lora_dropout=l_cfg.dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    return model, tokenizer