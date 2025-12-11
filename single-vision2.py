#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal vision-language model fine-tuning with Unsloth + DDP
Similar to train-ddp.py but for vision models with essential patch
————————————————————————————————————————————————————————————
Usage (example, 2x GPUs):
  accelerate launch --num_processes=2 single-vision2.py
  # or
  torchrun --nproc_per_node=2 single-vision2.py
"""

import os
import torch
from datasets import load_dataset
from transformers import TrainerCallback

# --------------------------- DDP bootstrap ---------------------------
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
IS_DIST = WORLD_SIZE > 1

if IS_DIST:
    if not torch.distributed.is_initialized():
        torch.cuda.set_device(LOCAL_RANK)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    else:
        torch.cuda.set_device(LOCAL_RANK)

# Disable Unsloth compilation to avoid hanging
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

# =====================================================================
# ESSENTIAL PATCH: Fix Unsloth gradient checkpointing for DDP
# =====================================================================
# Vision models trigger reentrant backward issues that language models don't.
# This minimal patch intercepts the problematic nested backward call.
# =====================================================================

def patch_unsloth_gradient_checkpointing():
    """Minimal patch to fix DDP reentrant backward issue."""
    try:
        import unsloth_zoo.gradient_checkpointing as unsloth_gc
        if hasattr(unsloth_gc, 'Unsloth_Gradient_Checkpointer'):
            CheckpointerClass = unsloth_gc.Unsloth_Gradient_Checkpointer
            original_backward = CheckpointerClass.backward
            
            @staticmethod
            def patched_backward(ctx, *args):
                """Skip nested backward call in DDP mode."""
                if WORLD_SIZE > 1 and torch.distributed.is_initialized():
                    # DDP mode: patch the nested backward call
                    original_torch_backward = torch.autograd.backward
                    def patched_torch_backward(*args, **kwargs):
                        return  # Skip nested backward - gradients flow naturally
                    torch.autograd.backward = patched_torch_backward
                    result = original_backward(ctx, *args)
                    torch.autograd.backward = original_torch_backward
                    return result
                else:
                    # Non-DDP: use original
                    return original_backward(ctx, *args)
            
            CheckpointerClass.backward = patched_backward
            if RANK == 0:
                print("[Patch] ✅ Fixed Unsloth gradient checkpointing for DDP")
    except Exception as e:
        if RANK == 0:
            print(f"[Patch] ⚠️ Could not patch: {e}")

# Patch before importing unsloth
try:
    patch_unsloth_gradient_checkpointing()
except:
    pass

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

# Patch again after import
try:
    patch_unsloth_gradient_checkpointing()
except:
    pass

# --------------------------- User config -----------------------------
MODEL_PATH = "unsloth/Qwen3-VL-4B-Instruct"
LORA_RANK = 16
MAX_LEN = 512
LR = 2e-4
NUM_STEPS = 10
PER_DEVICE_BS = 2
GRAD_ACCUM = 2

# ---------------------- 1) Load model/tokenizer ----------------------
device_map = {'': torch.cuda.current_device()} if torch.cuda.is_available() else None

model, tokenizer = FastVisionModel.from_pretrained(
    MODEL_PATH,
    load_in_4bit=True,
    # load_in_8bit=False,
    use_gradient_checkpointing="unsloth",
    device_map=device_map,
    # load_in_fp8 = True, # Float8 RL / GRPO!
    # full_finetuning = True,
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    r=LORA_RANK,
    lora_alpha=LORA_RANK,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    random_state=3407,
)

FastVisionModel.for_training(model)

# -------------------- 2) Prepare dataset ----------------------
dataset = load_dataset("unsloth/LaTeX_OCR", split="train")

instruction = "Write the LaTeX representation for this image."

def convert_to_conversation(sample):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": sample["image"]}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["text"]}]
            }
        ]
    }

converted_dataset = [convert_to_conversation(sample) for sample in dataset]

# -------------------- 3) Optional safety callback --------------------
class GradientGuard(TrainerCallback):
    """Skip a step if grad_norm explodes."""
    def on_step_end(self, args, state, control, **kwargs):
        gn = kwargs.get("logs", {}).get("grad_norm")
        if gn is None and state.log_history:
            gn = state.log_history[-1].get("grad_norm")
        if gn is not None and gn > 100:
            if RANK == 0:
                print(f"[Guard] grad_norm={gn:.1f} -> skip step {state.global_step}")
            control.should_skip_next_step = True

# --------------------- 4) Trainer & training --------------------
from trl import SFTTrainer, SFTConfig # import order matters, otherwise it will cause EOS error

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,  # Use processing_class instead of tokenizer
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=converted_dataset,
    args=SFTConfig(
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=5,
        max_steps=NUM_STEPS,
        learning_rate=LR,
        logging_steps=1,
        optim="adamw_8bit",
        # optim="muon",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=24,
        max_length=MAX_LEN,
        gradient_checkpointing=False,  # Using Unsloth's checkpointing instead
        ddp_find_unused_parameters=False,  # Important for LoRA + checkpointing
        # eos_token=tokenizer.eos_token,
        packing = True,
        # unsloth_tiled_mlp = True, => long context training
        dataloader_num_workers = 24,  # More workers for image processing
        dataloader_pin_memory = True,  # Faster CPU→GPU transfer
        dataloader_prefetch_factor = 24,  # Prefetch more batches
    ),
    callbacks=[GradientGuard()],
)

if RANK == 0:
    print("Starting training...")
train_stats = trainer.train()

# --------------------- 5) Save model ------------------------
if RANK == 0:
    save_dir = "lora_model_vision"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"✅ Training complete. Saved to: {save_dir}")

# ----------------------- Graceful DDP cleanup ------------------------
if IS_DIST and torch.distributed.is_initialized():
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()

