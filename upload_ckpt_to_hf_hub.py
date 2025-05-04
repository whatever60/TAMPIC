from huggingface_hub import HfApi

# Initialize the API
api = HfApi()

# Define your variables
local_ckpt_path = "tb_logs/20250109_TAMPIC_16s_genus_pretrained-rgb_only-no_empty-weight_density-192/version_1/checkpoints/epoch=249-step=12500-val-easy_top1_acc=0.67-val-easy_top3_acc=0.88.ckpt"

repo_id = "wusuowei60/tampic"  # YOUR space on HuggingFace
path_in_repo = "20250109_TAMPIC_16s_genus_pretrained-rgb_only-no_empty-weight_density-192/version_1/checkpoints/epoch=249-step=12500-val-easy_top1_acc=0.67-val-easy_top3_acc=0.88.ckpt"

# Upload
api.upload_file(
    path_or_fileobj=local_ckpt_path,
    path_in_repo=path_in_repo,
    repo_id=repo_id,
    repo_type="model",  # 👈 important, because you're uploading a model checkpoint
)

print("Upload complete!")
