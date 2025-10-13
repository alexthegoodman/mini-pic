import json
from pathlib import Path

json_dir = Path("../../diffusiondb/unzipped-json-augmented/")
json_files = sorted(json_dir.glob("*.json"))

print(f"Checking {len(json_files)} JSON files...")

for i, json_file in enumerate(json_files):
    try:
        print(f"Checking {json_file.name}... ", end="")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"OK ({len(data)} items)")
    except json.JSONDecodeError as e:
        print(f"\n❌ JSON decode error in {json_file.name}:")
        print(f"   {e}")
        print(f"   Position: line {e.lineno}, col {e.colno}")
    except UnicodeDecodeError as e:
        print(f"\n❌ Unicode decode error in {json_file.name}:")
        print(f"   {e}")
    except Exception as e:
        print(f"\n❌ Error in {json_file.name}:")
        print(f"   {type(e).__name__}: {e}")

print("\nDone!")
