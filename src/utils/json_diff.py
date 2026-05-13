import json
import sys
import math

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def is_nan(x):
    return isinstance(x, float) and math.isnan(x)

def diff(a, b, path=""):
    """
    Recursively compare JSON-like structures.
    Yields tuples: (path, value_in_a, value_in_b)
    """
    if type(a) != type(b):
        yield (path, a, b)
        return

    if isinstance(a, dict):
        keys = set(a.keys()) | set(b.keys())
        for k in sorted(keys):
            new_path = f"{path}.{k}" if path else k
            if k not in a:
                yield (new_path, None, b[k])
            elif k not in b:
                yield (new_path, a[k], None)
            else:
                yield from diff(a[k], b[k], new_path)

    elif isinstance(a, list):
        max_len = max(len(a), len(b))
        for i in range(max_len):
            new_path = f"{path}[{i}]"
            if i >= len(a):
                yield (new_path, None, b[i])
            elif i >= len(b):
                yield (new_path, a[i], None)
            else:
                yield from diff(a[i], b[i], new_path)

    else:
        if not (a == b or (is_nan(a) and is_nan(b))):
            yield (path, a, b)

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} file1.json file2.json")
        sys.exit(1)

    a = load_json(sys.argv[1])
    b = load_json(sys.argv[2])

    differences = list(diff(a, b))

    if not differences:
        print("No differences found.")
    else:
        for path, va, vb in differences:
            print(f"{path}:")
            print(f"  - file1: {va}")
            print(f"  - file2: {vb}")

if __name__ == "__main__":
    main()

