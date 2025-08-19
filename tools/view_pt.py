import torch

file_path = "/path/to/EEG2Img-Contrastive-Decoding/Data/ViT-H-14_features_train.pt"
# file_path = "/path/to/EEG2Img-Contrastive-Decoding/Data/ViT-H-14_features_test.pt"
# file_path = "/path/to/EEG2Img-Contrastive-Decoding/Data/train_image_latent_512.pt"
# file_path = "/path/to/EEG2Img-Contrastive-Decoding/Data/test_image_latent_512.pt"+

loaded_data = torch.load(file_path, map_location=torch.device('cpu'))

print("文件内容类型:", type(loaded_data))

if isinstance(loaded_data, dict):
    print("字典键列表:", loaded_data.keys())
    for key, value in loaded_data.items():
        if isinstance(value, list):
            info = f"长度: {len(value)}"
        elif hasattr(value, "shape"):
            info = f"形状: {value.shape}"
        else:
            info = "无形状"
        print(f"键: {key}, 类型: {type(value)}, {info}")


elif hasattr(loaded_data, "state_dict"):
    print("这是一个模型对象")
    print("模型结构:")
    print(loaded_data)

elif isinstance(loaded_data, torch.Tensor):
    print("这是一个张量")
    print("张量形状:", loaded_data.shape)
    print("张量数据类型:", loaded_data.dtype)
    print("张量前几行数据:", loaded_data[:5])

else:
    print("未知类型的数据")