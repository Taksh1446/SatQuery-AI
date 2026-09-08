from datasets import load_dataset

ds = load_dataset("mespinosami/sen12mscr", split="train", streaming=True)

for i, example in enumerate(ds):
    print("Keys:", example.keys())
    print("S1 type:", type(example["s1"]))
    print("S2 type:", type(example["s2"]))
    print("Text prompt:", example.get("text_prompt"))
    if i >= 0:
        break