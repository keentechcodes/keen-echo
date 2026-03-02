import json
import sys
from collections import Counter


def validate_dataset(file_path):
    print(f"Validating {file_path}...")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    valid_json_count = 0
    errors = []
    instruction_lengths = []
    output_lengths = []

    # Check for duplicates
    seen_inputs = set()
    duplicates = 0

    for i, line in enumerate(lines):
        line_num = i + 1
        try:
            data = json.loads(line)

            # Check required fields
            if "instruction" not in data or "output" not in data:
                errors.append(
                    f"Line {line_num}: Missing 'instruction' or 'output' field"
                )
                continue

            # Check content types
            if not isinstance(data["instruction"], str) or not isinstance(
                data["output"], str
            ):
                errors.append(f"Line {line_num}: Content is not string")
                continue

            # Check empties
            if not data["instruction"].strip():
                errors.append(f"Line {line_num}: Empty instruction")
            if not data["output"].strip():
                errors.append(f"Line {line_num}: Empty output")

            # Check duplicates (exact instruction match)
            if data["instruction"] in seen_inputs:
                duplicates += 1
            seen_inputs.add(data["instruction"])

            # Basic length stats (chars)
            instruction_lengths.append(len(data["instruction"]))
            output_lengths.append(len(data["output"]))

            valid_json_count += 1

        except json.JSONDecodeError:
            errors.append(f"Line {line_num}: Invalid JSON")

    print(f"\n--- Validation Results ---")
    print(f"Total lines: {len(lines)}")
    print(f"Valid pairs: {valid_json_count}")
    print(f"Duplicates: {duplicates} (same instruction)")

    if errors:
        print(f"\nErrors found ({len(errors)}):")
        for e in errors[:10]:  # Show first 10
            print(f"  - {e}")
        if len(errors) > 10:
            print(f"  ...and {len(errors) - 10} more.")
    else:
        print("\nNo structural errors found. ✅")

    if valid_json_count > 0:
        avg_inst = sum(instruction_lengths) / len(instruction_lengths)
        avg_out = sum(output_lengths) / len(output_lengths)
        print(f"\nStats:")
        print(f"  Avg Instruction Length: {avg_inst:.1f} chars")
        print(f"  Avg Output Length: {avg_out:.1f} chars")
        print(f"  Max Output Length: {max(output_lengths)} chars")


if __name__ == "__main__":
    path = "augmented_pairs.jsonl"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    validate_dataset(path)
