#!/usr/bin/env python3
"""
Standalone GGUF export helper.

Usage:
    python export_gguf.py <merged_model_dir> [output_dir]

Example:
    python export_gguf.py ./my-model-merged ./my-model-gguf
"""

import sys

model_dir = sys.argv[1] if len(sys.argv) > 1 else "./my-model-merged"
output_dir = sys.argv[2] if len(sys.argv) > 2 else "./my-model-gguf"

from unsloth import FastLanguageModel

print(f"Loading model from {model_dir}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_dir,
    max_seq_length=2048,
    load_in_4bit=False,
)

print(f"Exporting to GGUF at {output_dir}...")
model.save_pretrained_gguf(
    output_dir,
    tokenizer,
    quantization_method="q4_k_m",
)
print("Done!")
