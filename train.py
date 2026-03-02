#!/usr/bin/env python3
"""
keen-echo — Digital Twin Training Script
Fine-tunes Qwen3-8B on your personal writing style using LoRA via Unsloth.

Run on RunPod with:
- GPU: RTX 4090, A10, or A100 (24GB+ VRAM)
- Template: RunPod Pytorch 2.1+ or Unsloth template
- Time: ~20-40 minutes
- VRAM: ~5GB minimum with QLoRA (4-bit)

Usage:
    pip install unsloth
    pip install --upgrade transformers trl datasets
    python train.py

Notes:
    - Uses Qwen3-8B with Unsloth Dynamic 4-bit quantization
    - Thinking mode is DISABLED (enable_thinking=False) for persona fine-tuning
    - See: https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune
"""

import os
import json
import torch
from datetime import datetime

# ============================================================
# CONFIGURATION — Edit this section to customize for your persona
# ============================================================

CONFIG = {
    # Model - Qwen3-8B with Unsloth Dynamic 4-bit (better accuracy than standard bnb-4bit)
    # Available options:
    #   - "unsloth/Qwen3-8B-unsloth-bnb-4bit"  (recommended, Dynamic 4-bit)
    #   - "unsloth/Qwen3-8B-bnb-4bit"          (standard 4-bit)
    #   - "unsloth/Qwen3-4B-unsloth-bnb-4bit"  (smaller, faster)
    #   - "unsloth/Qwen3-14B-unsloth-bnb-4bit" (larger, more capable)
    # Note: There is NO Qwen3-7B - sizes are 0.6B, 1.7B, 4B, 8B, 14B, 32B
    "model_name": "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    "max_seq_length": 2048,
    "load_in_4bit": True,
    # LoRA - these settings are validated for Qwen3
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    # Training
    "num_epochs": 3,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 10,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "seed": 42,
    # Data — point this to your JSONL training pairs
    "dataset_path": "augmented_pairs.jsonl",
    # Output — model artifacts will be saved here
    "output_dir": "./my-model",
    "hub_model_id": None,  # Set to "your-username/your-model" to push to HF
}

# ============================================================
# SYSTEM PROMPT — Define your persona here
# ============================================================
# This prompt is baked into every training example and used at inference time.
# Customize it to match the writing style you're fine-tuning on.

SYSTEM_PROMPT = """You are a digital twin. You write in a unique personal style.

Customize this prompt to describe your target persona's writing patterns:
- tone and formality level
- punctuation and capitalization habits
- common expressions, filler words, slang
- emoji usage
- topics and themes they gravitate toward

Respond authentically in that persona's voice."""

# ============================================================
# MAIN TRAINING SCRIPT
# ============================================================


def main():
    print("=" * 60)
    print("keen-echo — Digital Twin Training")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {CONFIG['model_name']}")
    print(f"Dataset: {CONFIG['dataset_path']}")
    print()

    # --------------------------------------------------------
    # Step 1: Install dependencies (if needed)
    # --------------------------------------------------------
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
    except ImportError:
        print("Installing Unsloth...")
        os.system("pip install unsloth")
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template

    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # --------------------------------------------------------
    # Step 2: Load Model
    # --------------------------------------------------------
    print("Loading model...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CONFIG["model_name"],
        max_seq_length=CONFIG["max_seq_length"],
        load_in_4bit=CONFIG["load_in_4bit"],
        dtype=None,  # Auto-detect
    )

    print(f"Model loaded: {CONFIG['model_name']}")

    # --------------------------------------------------------
    # Step 3: Add LoRA Adapters
    # --------------------------------------------------------
    print("Adding LoRA adapters...")

    model = FastLanguageModel.get_peft_model(
        model,
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=CONFIG["seed"],
    )

    print(f"LoRA adapters added (r={CONFIG['lora_r']}, alpha={CONFIG['lora_alpha']})")

    # --------------------------------------------------------
    # Step 4: Load and Format Dataset
    # --------------------------------------------------------
    print("Loading dataset...")

    # Load JSONL
    data = []
    with open(CONFIG["dataset_path"], "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print(f"Loaded {len(data)} training pairs")

    # Format for chat template
    # IMPORTANT: enable_thinking=False disables Qwen3's reasoning mode
    # This is critical for persona fine-tuning - we want direct responses, not <think> blocks
    def format_conversation(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,  # Disable thinking mode for persona (no <think> blocks)
        )

        return {"text": text}

    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_conversation, remove_columns=dataset.column_names)

    print(f"Dataset formatted with chat template")
    print(f"  Sample:\n{dataset[0]['text'][:500]}...")

    # --------------------------------------------------------
    # Step 5: Training
    # --------------------------------------------------------
    print("\nStarting training...")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=CONFIG["max_seq_length"],
        dataset_num_proc=2,
        packing=True,  # Pack short sequences together
        args=TrainingArguments(
            output_dir=CONFIG["output_dir"],
            num_train_epochs=CONFIG["num_epochs"],
            per_device_train_batch_size=CONFIG["batch_size"],
            gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
            learning_rate=CONFIG["learning_rate"],
            warmup_steps=CONFIG["warmup_steps"],
            weight_decay=CONFIG["weight_decay"],
            max_grad_norm=CONFIG["max_grad_norm"],
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            seed=CONFIG["seed"],
            report_to="none",
        ),
    )

    # Train!
    trainer_stats = trainer.train()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
    print(f"Final loss: {trainer_stats.metrics['train_loss']:.4f}")

    # --------------------------------------------------------
    # Step 6: Save Model
    # --------------------------------------------------------
    print("\nSaving model...")

    # Save LoRA adapters
    model.save_pretrained(CONFIG["output_dir"])
    tokenizer.save_pretrained(CONFIG["output_dir"])
    print(f"LoRA adapters saved to {CONFIG['output_dir']}")

    # Save merged model (for vLLM deployment)
    merged_dir = f"{CONFIG['output_dir']}-merged"
    print(f"\nMerging and saving full model to {merged_dir}...")

    model.save_pretrained_merged(
        merged_dir,
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"Merged model saved to {merged_dir}")

    # --------------------------------------------------------
    # Step 7: Export to GGUF (for Ollama)
    # --------------------------------------------------------
    print("\nExporting to GGUF (for Ollama)...")

    gguf_dir = f"{CONFIG['output_dir']}-gguf"

    model.save_pretrained_gguf(
        gguf_dir,
        tokenizer,
        quantization_method="q4_k_m",  # Good balance of size/quality
    )
    print(f"GGUF exported to {gguf_dir}")

    # --------------------------------------------------------
    # Step 8: Push to HuggingFace (optional)
    # --------------------------------------------------------
    if CONFIG["hub_model_id"]:
        print(f"\nPushing to HuggingFace Hub: {CONFIG['hub_model_id']}...")

        model.push_to_hub_merged(
            CONFIG["hub_model_id"],
            tokenizer,
            save_method="merged_16bit",
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"Model pushed to https://huggingface.co/{CONFIG['hub_model_id']}")

    # --------------------------------------------------------
    # Step 9: Test the model
    # --------------------------------------------------------
    print("\nTesting the model...")

    FastLanguageModel.for_inference(model)

    test_prompts = [
        "What's your philosophy on learning?",
        "thoughts on balance in life?",
        "React to learning something mind-blowing",
    ]

    for prompt in test_prompts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,  # Disable thinking mode for inference
            return_tensors="pt",
        ).to("cuda")

        # Qwen3 recommended params for non-thinking mode: temp=0.7, top_p=0.8, top_k=20
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.8,  # Qwen3 non-thinking recommendation
            top_k=20,  # Qwen3 non-thinking recommendation
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        response = tokenizer.decode(
            outputs[0][inputs.shape[1] :], skip_special_tokens=True
        )

        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
Files created:
  {CONFIG["output_dir"]}/          - LoRA adapters
  {merged_dir}/                    - Merged model (for vLLM/Cloud Run)
  {gguf_dir}/                      - GGUF file (for Ollama)

Next steps:
  1. Upload {merged_dir}/ to Google Cloud Storage
  2. Deploy to Cloud Run with vLLM (see docs/02-gcp-deployment.md)
  3. Or download the GGUF and run locally with Ollama (see docs/03-local-ollama.md)

Finished at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
""")


if __name__ == "__main__":
    main()
