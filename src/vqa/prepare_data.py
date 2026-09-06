import json
import os

DATA_DIR = "data/rsvqa_lr"
OUTPUT_FILE = "data/rsvqa_lr/clean_train.json"

def load(split):
    with open(f"{DATA_DIR}/LR_split_{split}_questions.json") as f:
        questions = json.load(f)["questions"]
    with open(f"{DATA_DIR}/LR_split_{split}_answers.json") as f:
        answers = json.load(f)["answers"]
    return questions, answers

def build_clean_list(split):
    questions, answers = load(split)
    answer_by_id = {a["id"]: a["answer"] for a in answers if "answer" in a and "id" in a}
    clean = []
    for q in questions:
        if "img_id" not in q or "answers_ids" not in q:
            continue
        ans_id = q["answers_ids"][0]
        if ans_id not in answer_by_id:
            continue
        clean.append({
            "image_id": q["img_id"],
            "question": q["question"],
            "answer": answer_by_id[ans_id]
        })
    return clean

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        clean = build_clean_list(split)
        print(f"Built {len(clean)} clean {split} examples")
        out_path = f"data/rsvqa_lr/clean_{split}.json"
        with open(out_path, "w") as f:
            json.dump(clean, f)
        print(f"Saved to {out_path}")