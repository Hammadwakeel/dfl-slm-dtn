"""One-shot GPU inference worker for the reply-style LoRA adapter."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from master.reply_service import (
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    ReplyInputError,
    SYSTEM_PROMPT,
    build_user_prompt,
)


def generate(message: str, context: dict[str, Any]) -> str:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = Path(os.environ["DFL_REPLY_BASE_MODEL"])
    adapter = Path(os.environ["DFL_REPLY_ADAPTER"])
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The reply model requires CUDA with BF16 support")
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model),
        local_files_only=True,
    )
    if tokenizer.eos_token_id is None:
        raise RuntimeError("The reply model tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(
        model,
        str(adapter),
        is_trainable=False,
    )
    model.eval()
    model.config.use_cache = True
    model.generation_config.temperature = None
    model.generation_config.top_p = None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(message, context)},
    ]
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if token_ids.shape[1] > MAX_INPUT_TOKENS:
        raise ReplyInputError(
            f"request exceeds the {MAX_INPUT_TOKENS}-token fine-tuned model input limit"
        )
    token_ids = token_ids.to(model.device)
    attention_mask = torch.ones_like(token_ids)
    with torch.inference_mode():
        output = model.generate(
            token_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            synced_gpus=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output[0, token_ids.shape[1] :],
        skip_special_tokens=True,
    ).strip()


def main() -> None:
    raw_payload = sys.stdin.read(16_385)
    if len(raw_payload) > 16_384:
        raise ReplyInputError("request body exceeds the allowed size")
    payload = json.loads(raw_payload)
    if not isinstance(payload, dict):
        raise ReplyInputError("request must be a JSON object")
    message = payload.get("message")
    context = payload.get("context", {})
    if not isinstance(context, dict):
        raise ReplyInputError("context must be a JSON object")
    try:
        reply = generate(message, context)
    except ReplyInputError as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False))
        return
    print(json.dumps({"reply": reply}, ensure_ascii=False))


if __name__ == "__main__":
    main()
