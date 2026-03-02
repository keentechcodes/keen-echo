import json
import re
import sys


def style_cleaner(text):
    """Apply persona-style text cleaning to training data outputs.

    This example lowercases text, removes trailing periods, and lowercases
    standalone "I" -- customize for your target persona's writing style.
    """
    # 1. Lowercase start of text
    if text and text[0].isupper():
        text = text[0].lower() + text[1:]

    # 2. Lowercase "I" when it's a standalone word
    text = re.sub(r"\bI\b", "i", text)

    # 3. Remove periods at the end of lines/paragraphs
    # Split by newlines, clean each line, join back
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line.endswith("."):
            line = line[:-1]
        if line:
            # Lowercase start of new lines too
            if line[0].isupper() and len(line) > 1:
                line = line[0].lower() + line[1:]
            cleaned_lines.append(line)
        else:
            cleaned_lines.append("")  # Preserve empty lines for spacing

    text = "\n".join(cleaned_lines)

    return text


# ============================================================
# CONFIGURATION — Edit these paths for your setup
# ============================================================

input_path = sys.argv[1] if len(sys.argv) > 1 else "augmented_pairs.jsonl"
output_path = sys.argv[2] if len(sys.argv) > 2 else "refined_pairs.jsonl"

processed_count = 0
original_output = ""
new_output = ""

with (
    open(input_path, "r", encoding="utf-8") as infile,
    open(output_path, "w", encoding="utf-8") as outfile,
):
    for line in infile:
        data = json.loads(line)
        original_output = data["output"]

        # Apply cleaning
        new_output = style_cleaner(original_output)

        data["output"] = new_output
        outfile.write(json.dumps(data) + "\n")
        processed_count += 1

print(f"Processed {processed_count} pairs.")
print(f"Input:  {input_path}")
print(f"Output: {output_path}")

# Show a before/after sample
if processed_count > 0:
    print("\n--- Sample Change ---")
    print("Before:")
    print(original_output[:100] + "...")
    print("\nAfter:")
    print(new_output[:100] + "...")
