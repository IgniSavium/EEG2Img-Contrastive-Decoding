import os
os.environ['HF_HOME'] = '/path/to/huggingface_cache'

from huggingface_hub import snapshot_download

# 允许的文件/文件夹模式
allow_patterns = [
    "Preprocessed_data_250Hz/sub-08",
    "fintune_ckpts/sub-08",
    ".gitattributes",
    "README.md",
    "ViT-H-14_features_test.pt",
    "ViT-H-14_features_train.pt",
    "generated_imgs.tar.gz",
    "test_image_latent_512.pt",
    "train_image_latent_512.pt"
]

snapshot_download(
    repo_id="LidongYang/EEG_Image_decode",
    repo_type="dataset",
    local_dir="/path/to/EEG2Img-Contrastive-Decoding/Data",
    allow_patterns=allow_patterns
)

