from huggingface_hub import hf_hub_download

# Download just the CDVQA question/answer file (small, 29MB)
cdvqa_json_path = hf_hub_download(
    repo_id="jirvin16/TEOChatlas",
    filename="eval/CDVQA.json",
    repo_type="dataset",
    local_dir="data/cdvqa"
)
print("CDVQA JSON downloaded to:", cdvqa_json_path)

# Download the images archive (larger, 6.14GB)
images_tar_path = hf_hub_download(
    repo_id="jirvin16/TEOChatlas",
    filename="eval/External_images.tar.gz",
    repo_type="dataset",
    local_dir="data/cdvqa"
)
print("Images tar downloaded to:", images_tar_path)

import json

with open("data/cdvqa/eval/CDVQA.json") as f:
    data = json.load(f)

print("Type:", type(data))
print("Length:", len(data))
print("First entry:", data[0])