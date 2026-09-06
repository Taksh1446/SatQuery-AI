from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="xiang709/VRSBench",
    repo_type="dataset",
    local_dir="data/vrsbench"
)

print("Download complete")