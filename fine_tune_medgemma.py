import unsloth          # must be first — patches transformers/trl at import time
import argparse
import os
import sys
import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel
from typing import Any


# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="MedGemma 4B LoRA fine-tuning")
    parser.add_argument("--model_name",     type=str,   default="unsloth/medgemma-4b-it-bnb-4bit",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--train_file",     type=str,   default="data/train.jsonl",
                        help="Path to training JSONL file")
    parser.add_argument("--eval_file",      type=str,   default="data/eval.jsonl",
                        help="Path to evaluation JSONL file")
    parser.add_argument("--output_dir",     type=str,   default="medgemma-4b-it-sft-lora",
                        help="Output directory for checkpoints and final model")
    parser.add_argument("--epochs",         type=int,   default=5,
                        help="Number of training epochs")
    parser.add_argument("--lr",             type=float, default=2e-4,
                        help="Peak learning rate")
    parser.add_argument("--max_seq_length", type=int,   default=2048,
                        help="Max token sequence length")
    parser.add_argument("--lora_r",         type=int,   default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha",     type=int,   default=32,
                        help="LoRA alpha (scale = alpha / r)")
    parser.add_argument("--lora_dropout",   type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--weight_decay",   type=float, default=0.0,
                        help="AdamW weight decay (0 = no regularisation)")
    parser.add_argument("--grad_accum",     type=int,   default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--batch_size",     type=int,   default=2,
                        help="Per-device train batch size")
    parser.add_argument("--warmup_ratio",   type=float, default=0.03,
                        help="Warmup ratio for LR scheduler")
    return parser.parse_args()


# ── 1. Load model ─────────────────────────────────────────────────────────────
def load_model(args):
    print(f"[1/6] Loading model: {args.model_name}", flush=True)
    model, processor = FastVisionModel.from_pretrained(
        model_name=args.model_name,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        # Avoid the very long Flex Attention/Triton autotune pause seen on
        # this cluster before the first training step.
        attn_implementation="sdpa",
    )
    # Right-padding avoids issues during batched training
    processor.tokenizer.padding_side = "right"
    print("      Model loaded.", flush=True)
    return model, processor


# ── 2. LoRA adapters ──────────────────────────────────────────────────────────
def apply_lora(model, args):
    print(f"[2/6] Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout})", flush=True)
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision=False,           # text-only task — no images in dataset
        finetune_language=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    print("      LoRA adapters attached.", flush=True)
    return model


# ── 3. Dataset ────────────────────────────────────────────────────────────────
def load_data(args):
    print(f"[3/6] Loading dataset:", flush=True)
    print(f"      train -> {args.train_file}", flush=True)
    print(f"      eval  -> {args.eval_file}", flush=True)

    if not os.path.isfile(args.train_file):
        sys.exit(f"ERROR: training file not found: {args.train_file}")
    if not os.path.isfile(args.eval_file):
        sys.exit(f"ERROR: eval file not found: {args.eval_file}")

    train_data = load_dataset("json", data_files=args.train_file)["train"]
    eval_data  = load_dataset("json", data_files=args.eval_file)["train"]
    print(f"      Loaded {len(train_data)} train / {len(eval_data)} eval samples.", flush=True)
    return train_data, eval_data


# ── 3.5. Dataset filter ──────────────────────────────────────────────────────
def filter_dataset(data, processor, max_seq_length: int, split_name: str = "dataset"):
    """
    Remove samples that would produce all-masked labels (and thus NaN loss).
    This happens when:
      1. The output field is empty.
      2. The prompt alone is >= max_seq_length (after truncation, zero output tokens remain).
    """
    def is_valid(example):
        if not example.get("output", "").strip():
            return False
        messages_prompt = [{"role": "user", "content": example["instruction"]}]
        messages_full   = [
            {"role": "user",      "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
        prompt_text = processor.apply_chat_template(
            messages_prompt, add_generation_prompt=True, tokenize=False
        ).strip()
        full_text = processor.apply_chat_template(
            messages_full, add_generation_prompt=False, tokenize=False
        ).strip()
        prompt_len = len(processor.tokenizer.encode(prompt_text, add_special_tokens=False))
        full_len   = len(processor.tokenizer.encode(full_text,   add_special_tokens=False))
        # At least 1 output token must survive after truncation
        return full_len > prompt_len and prompt_len < max_seq_length

    before   = len(data)
    filtered = data.filter(is_valid)
    after    = len(filtered)
    if before != after:
        print(f"[INFO] {split_name}: removed {before - after} invalid sample(s) "
              f"({before} → {after})", flush=True)
    return filtered


# Key fix: locate the <start_of_turn>model boundary and mask all tokens before
# it with -100, so the model is only penalised for the JSON output — not the prompt.

RESPONSE_TEMPLATE = "<start_of_turn>model\n"


def build_collate_fn(processor, max_seq_length: int = 1024):
    """
    Robust label masking: instead of searching for response-template token IDs
    (which can silently fail and produce all-NaN eval loss), we tokenize the
    prompt-only portion and use its length directly to mask labels.
    """
    def collate_fn(examples: list[dict[str, Any]]):
        texts = []
        prompt_lengths = []

        for example in examples:
            # Full conversation: user prompt + assistant output
            messages_full = [
                {"role": "user",      "content": example["instruction"]},
                {"role": "assistant", "content": example["output"]},
            ]
            # Prompt only (with generation prompt so the template boundary is correct)
            messages_prompt = [
                {"role": "user", "content": example["instruction"]},
            ]

            full_text = processor.apply_chat_template(
                messages_full, add_generation_prompt=False, tokenize=False
            ).strip()
            prompt_text = processor.apply_chat_template(
                messages_prompt, add_generation_prompt=True, tokenize=False
            ).strip()

            texts.append(full_text)
            # Measure the prompt length in tokens (no special tokens — they're
            # already embedded in the chat template string)
            prompt_ids = processor.tokenizer.encode(
                prompt_text, add_special_tokens=False
            )
            prompt_lengths.append(len(prompt_ids))

        batch = processor(
            text=texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_seq_length
        )
        labels = batch["input_ids"].clone()

        for i, prompt_len in enumerate(prompt_lengths):
            # Mask the prompt portion — model only learns to generate the output
            labels[i, :prompt_len] = -100

        # Mask padding tokens
        if processor.tokenizer.pad_token_id is not None:
            labels[labels == processor.tokenizer.pad_token_id] = -100

        # Sanity check: warn if the entire sequence was masked (means output is empty)
        all_masked = (labels != -100).sum(dim=-1) == 0
        if all_masked.any():
            print(f"[WARNING] {all_masked.sum().item()} example(s) have all labels masked. "
                  "Check that 'output' field is non-empty.", flush=True)

        batch["labels"] = labels
        return batch

    return collate_fn


# ── 5. Training config ────────────────────────────────────────────────────────
def build_training_args(args):
    print(f"[4/6] Building SFTConfig:", flush=True)
    print(f"      epochs={args.epochs}  lr={args.lr}  max_seq_length={args.max_seq_length}", flush=True)
    print(f"      output_dir={args.output_dir}", flush=True)
    print("      attention=sdpa (Flex Attention disabled)", flush=True)

    return SFTConfig(
        output_dir=args.output_dir,

        # Training duration
        num_train_epochs=args.epochs,

        # Batch / memory
        # max_seq_length removed from SFTConfig in TRL 0.13 — now set in collate_fn
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Optimizer
        optim="paged_adamw_8bit",
        weight_decay=args.weight_decay,

        # Learning rate schedule
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,

        # Precision
        bf16=True,

        # Regularisation
        max_grad_norm=0.3,

        # Logging / evaluation
        logging_steps=5,          # print train loss every 5 steps
        eval_strategy="steps",
        eval_steps=25,            # print eval loss every 25 steps
        save_strategy="steps",    # must match eval_strategy for load_best_model_at_end
        save_steps=25,
        load_best_model_at_end=True,   # keep checkpoint with lowest eval_loss
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,       # keep only best + last checkpoint to save disk

        # Misc
        push_to_hub=False,
        report_to=["wandb", "tensorboard"],
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        label_names=["labels"],
    )


# ── 6. Quick inference test ───────────────────────────────────────────────────
def run_inference_test(model, processor):
    print("\n[6/6] Running quick inference test ...", flush=True)
    FastVisionModel.for_inference(model)

    prompt_text = (
        "Given the description below, produce the protocol metadata "
        "(labware, pipettes, modules, reagents, categories) as JSON.\n"
        "This protocol automates GNA Octea prep. Using the P300 Multi-Channel Pipette (GEN2) "
        "and the P300 Single-Channel Pipette (GEN2), all the reagents necessary for the "
        "sample prep are transferred to the samples and the samples are incubated on the "
        "Temperature Module and Magnetic Module. Once all sample prep has been completed, "
        "the samples are transferred to a custom plate containing the labware needed to test "
        "the samples on the GNA analyzer.\n---"
    )

    messages = [{"role": "user", "content": prompt_text}]
    formatted = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(text=formatted, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=1.0,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    prompt_length = inputs["input_ids"].shape[-1]
    answer = processor.tokenizer.decode(
        output_tokens[0][prompt_length:], skip_special_tokens=True
    )
    print("\n=== Generated Response ===\n")
    print(answer)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print("=" * 60, flush=True)
    print("  MedGemma 4B Fine-Tuning — v2", flush=True)
    print("=" * 60, flush=True)
    print(f"  GPU available : {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"  GPU name      : {torch.cuda.get_device_name(0)}", flush=True)
        print(f"  VRAM total    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)
    print("=" * 60, flush=True)

    model, processor  = load_model(args)
    model             = apply_lora(model, args)
    train_data, eval_data = load_data(args)

    # Filter out samples that would produce NaN loss
    train_data = filter_dataset(train_data, processor, args.max_seq_length, "train")
    eval_data  = filter_dataset(eval_data,  processor, args.max_seq_length, "eval")

    training_args     = build_training_args(args)
    collate_fn        = build_collate_fn(processor, args.max_seq_length)

    print("[5/6] Starting training ...", flush=True)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=processor,
        data_collator=collate_fn,
    )
    trainer.train()

    # Save model + processor
    save_path = args.output_dir
    trainer.save_model(save_path)
    processor.save_pretrained(save_path)
    print(f"\n[OK] Model saved to: {save_path}", flush=True)

    run_inference_test(model, processor)
    print("\n[OK] Done.", flush=True)


if __name__ == "__main__":
    main()
