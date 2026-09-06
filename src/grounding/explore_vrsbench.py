import json
import os

DATA_DIR = "data/vrsbench"

print("Files in data/vrsbench:")
for f in os.listdir(DATA_DIR):
    print(" -", f)

with open(f"{DATA_DIR}/VRSBench_EVAL_referring.json") as f:
    referring = json.load(f)

print("\nType:", type(referring))
if isinstance(referring, list):
    print("Number of entries:", len(referring))
    print("First entry:", referring[0])
else:
    print("First entry:", list(referring.items())[0])