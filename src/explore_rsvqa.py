import json

DATA_DIR = "data/rsvqa_lr"

# Load a small piece of each file to see their structure
with open(f"{DATA_DIR}/LR_split_train_questions.json") as f:
    questions = json.load(f)

with open(f"{DATA_DIR}/LR_split_train_answers.json") as f:
    answers = json.load(f)

with open(f"{DATA_DIR}/LR_split_train_images.json") as f:
    images = json.load(f)

print("Type of questions data:", type(questions))
print("First question entry:", questions["questions"][0] if "questions" in questions else list(questions.items())[0])
print()
print("Type of answers data:", type(answers))
print("First answer entry:", answers["answers"][0] if "answers" in answers else list(answers.items())[0])
print()
print("Type of images data:", type(images))
print("First image entry:", images["images"][0] if "images" in images else list(images.items())[0])

import matplotlib.pyplot as plt
from PIL import Image

# Find image with id 0, get its filename
img_entry = images["images"][0]
img_id = img_entry["id"]

# Find a question for this image
question_entry = [q for q in questions["questions"] if q.get("img_id") == img_id][0]

# Find the matching answer
answer_id = question_entry["answers_ids"][0]
answer_entry = [a for a in answers["answers"] if a["id"] == answer_id][0]

print("\n--- Linked example ---")
print("Question:", question_entry["question"])
print("Answer:", answer_entry["answer"])

# Open and show the actual image (image files are usually named by their id, e.g. 0.tif)
img_path = f"{DATA_DIR}/Images_LR/Images_LR/{img_id}.tif"
img = Image.open(img_path)
plt.imshow(img)
plt.title(f"Q: {question_entry['question']}  |  A: {answer_entry['answer']}")
plt.show()