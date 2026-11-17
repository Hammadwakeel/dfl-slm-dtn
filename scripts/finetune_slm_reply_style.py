#!/usr/bin/env python3
"""LoRA fine-tune Llama 3.2 1B on natural-language DFL recommendations."""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "You are a concise, helpful decentralized federated learning assistant. "
    "Use only the supplied DFL state. Recommend a next module when the evidence "
    "supports it; otherwise explain what information is missing. Reply in natural "
    "language with one short reason and one actionable next step. Never output JSON, "
    "an action object, or claim that you executed an action."
)
MODULES = frozenset(
    {
        "Adaptive Aggregation",
        "Continue Current Training",
        "Dynamic Topology Optimization",
        "Hyperparameter Optimization",
        "Insufficient Evidence",
        "NAS",
        "Node Selection",
        "Topology Optimization",
    }
)


def natural_language_target(row: dict[str, Any]) -> dict[str, str]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("each example must contain system, user, and assistant messages")
    if [message.get("role") for message in messages if isinstance(message, dict)] != [
        "system",
        "user",
        "assistant",
    ]:
        raise ValueError("conversation roles must be system, user, assistant")

    user_message = messages[1].get("content")
    if not isinstance(user_message, str) or not user_message.strip():
        raise ValueError("user message must be non-empty text")
    try:
        answer = json.loads(messages[2].get("content", ""))
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("assistant message must contain the expected JSON labels") from error
    if not isinstance(answer, dict):
        raise ValueError("assistant label must be a JSON object")

    recommendations = answer.get("recommended_modules")
    if not isinstance(recommendations, list) or not recommendations:
        raise ValueError("assistant label must include a recommended module")
    recommendation = recommendations[0]
    if not isinstance(recommendation, dict):
        raise ValueError("recommended module label is malformed")
    module = recommendation.get("module")
    reason = answer.get("reason")
    next_action = answer.get("next_action")
    if not isinstance(module, str) or module not in MODULES:
        raise ValueError("recommended module label is not recognized")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("assistant label must include a reason")
    if not isinstance(next_action, str) or not next_action.strip():
        raise ValueError("assistant label must include a next step")

    opening = (
        "There is not enough information to recommend a module yet."
        if module == "Insufficient Evidence"
        else f"I recommend {module}."
    )
    target = f"{opening} {reason.strip()} Next, {next_action.strip()}"
    return {"user": user_message.strip(), "target": target}


def read_split(path: Path) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("row must be a JSON object")
            examples.append(natural_language_target(row))
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"{path.name}:{line_number}: {error}") from error
    if not examples:
        raise ValueError(f"{path} contains no examples")
    return examples


def _perplexity(metrics: dict[str, Any], loss_key: str) -> float:
    loss = metrics.get(loss_key)
    if not isinstance(loss, (int, float)) or not math.isfinite(float(loss)):
        raise ValueError(f"Evaluation did not return a finite {loss_key}")
    return math.exp(float(loss))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_length < 64 or not 1 <= args.epochs <= 10 or not 1 <= args.rank <= 32:
        raise ValueError("max-length, epochs, or LoRA rank is outside the supported range")
    if not args.base_model.is_dir():
        raise FileNotFoundError(f"Local base-model snapshot was not found: {args.base_model}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "status": "running",
        "task": "natural-language reply style only; structured action handling unchanged",
        "base_model": str(args.base_model),
        "max_length": args.max_length,
        "rank": args.rank,
        "epochs": args.epochs,
        "local_files_only": True,
    }
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA and BF16 support are required on the Jetson trainer")
        report["gpu"] = torch.cuda.get_device_name(0)
        report["cuda_memory_before_mib"] = round(
            torch.cuda.mem_get_info()[0] / 1024**2,
            1,
        )

        raw_splits = {
            split: read_split(args.dataset_dir / f"dfl_{split}.jsonl")
            for split in ("train", "validation", "test")
        }
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.base_model),
            local_files_only=True,
        )
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                raise RuntimeError("Tokenizer has no padding or EOS token")
            tokenizer.pad_token = tokenizer.eos_token

        tokenized: dict[str, Dataset] = {}
        token_lengths: dict[str, dict[str, int | float]] = {}
        for split, examples in raw_splits.items():
            records = []
            lengths = []
            for example in examples:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["user"]},
                    {"role": "assistant", "content": example["target"]},
                ]
                token_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                )
                prefix_ids = tokenizer.apply_chat_template(
                    messages[:-1],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                if token_ids[: len(prefix_ids)] != prefix_ids:
                    raise RuntimeError(f"{split} assistant prefix does not match the chat template")
                if len(token_ids) > args.max_length:
                    raise ValueError(
                        f"{split} example needs {len(token_ids)} tokens, "
                        f"above max_length={args.max_length}"
                    )
                records.append(
                    {
                        "input_ids": token_ids,
                        "attention_mask": [1] * len(token_ids),
                        "labels": [-100] * len(prefix_ids) + token_ids[len(prefix_ids) :],
                    }
                )
                lengths.append(len(token_ids))
            tokenized[split] = Dataset.from_list(records)
            token_lengths[split] = {
                "rows": len(lengths),
                "min": min(lengths),
                "mean": round(sum(lengths) / len(lengths), 2),
                "max": max(lengths),
            }
        report["split_rows"] = {name: len(data) for name, data in tokenized.items()}
        report["token_lengths"] = token_lengths
        if args.preflight_only:
            report["status"] = "preflight_passed"
            (args.output_dir / "training_report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            print("PREFLIGHT_REPORT=" + json.dumps(report, allow_nan=False), flush=True)
            return

        model = AutoModelForCausalLM.from_pretrained(
            str(args.base_model),
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        target_modules = sorted(
            {
                name.rsplit(".", 1)[-1]
                for name, module in model.named_modules()
                if isinstance(module, torch.nn.Linear) and not name.endswith("lm_head")
            }
        )
        expected_targets = [
            "down_proj",
            "gate_proj",
            "k_proj",
            "o_proj",
            "q_proj",
            "up_proj",
            "v_proj",
        ]
        if target_modules != expected_targets:
            raise RuntimeError(f"Unexpected LoRA target modules: {target_modules}")
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.rank,
                lora_alpha=args.rank * 2,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            ),
        )
        model.config.use_cache = False
        model.config.pad_token_id = tokenizer.pad_token_id
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
        report["target_modules"] = target_modules
        report["trainable_parameters"] = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )

        args_trainer = TrainingArguments(
            output_dir=str(args.output_dir / "checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            optim="adamw_torch",
            bf16=True,
            fp16=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_strategy="steps",
            logging_steps=10,
            report_to="none",
            dataloader_pin_memory=False,
            disable_tqdm=True,
            seed=42,
        )
        trainer = Trainer(
            model=model,
            args=args_trainer,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            data_collator=DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                padding=True,
                label_pad_token_id=-100,
                pad_to_multiple_of=8,
            ),
        )
        report["base_validation_metrics"] = trainer.evaluate(
            tokenized["validation"],
            metric_key_prefix="base_validation",
        )
        report["base_test_metrics"] = trainer.evaluate(
            tokenized["test"],
            metric_key_prefix="base_test",
        )
        report["base_perplexity_metrics"] = {
            "validation": _perplexity(
                report["base_validation_metrics"],
                "base_validation_loss",
            ),
            "test": _perplexity(report["base_test_metrics"], "base_test_loss"),
        }

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        train_output = trainer.train()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        adapter_dir = args.output_dir / "adapter"
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        report["validation_metrics"] = trainer.evaluate(
            tokenized["validation"],
            metric_key_prefix="validation",
        )
        report["test_metrics"] = trainer.evaluate(
            tokenized["test"],
            metric_key_prefix="test",
        )
        report["perplexity_metrics"] = {
            "validation": _perplexity(
                report["validation_metrics"],
                "validation_loss",
            ),
            "test": _perplexity(report["test_metrics"], "test_loss"),
        }
        report.update(
            status="training_completed_probe_pending",
            train_seconds=round(elapsed, 2),
            steps=int(trainer.state.global_step),
            seconds_per_step=round(elapsed / max(1, int(trainer.state.global_step)), 3),
            train_loss=float(train_output.training_loss),
            train_metrics=train_output.metrics,
            peak_torch_cuda_memory_allocated_mib=round(
                torch.cuda.max_memory_allocated() / 1024**2,
                1,
            ),
            adapter_dir=str(adapter_dir),
            adapter_size_bytes=sum(
                path.stat().st_size for path in adapter_dir.rglob("*") if path.is_file()
            ),
            structured_action_handling="unchanged",
        )
        (args.output_dir / "training_report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        probe_examples = raw_splits["test"][:12]
        generated_json = 0
        generated_empty = 0
        generated_lengths = []
        model.eval()
        for example in probe_examples:
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["user"]},
                ],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)
            attention_mask = torch.ones_like(prompt)
            with torch.inference_mode():
                generated = model.generate(
                    prompt,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    do_sample=False,
                    synced_gpus=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(
                generated[0, prompt.shape[1] :],
                skip_special_tokens=True,
            ).strip()
            generated_empty += int(not text)
            generated_lengths.append(len(text))
            try:
                json.loads(text)
            except json.JSONDecodeError:
                pass
            else:
                generated_json += 1

        report.update(
            status="completed",
            generation_probe={
                "rows": len(probe_examples),
                "empty_responses": generated_empty,
                "json_responses": generated_json,
                "mean_characters": round(sum(generated_lengths) / len(generated_lengths), 1),
            },
        )
    except BaseException as error:
        report.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error)[:2000],
        )
        traceback.print_exc()
        (args.output_dir / "training_report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        raise

    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("TRAINING_REPORT=" + json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
