import numpy as np

file_path = "/path/to/EEG2Img-Contrastive-Decoding/Data/Preprocessed_data_250Hz/sub-08/preprocessed_eeg_training.npy"
data = np.load(file_path, allow_pickle=True)

print("数据类型:", type(data))

if isinstance(data, dict):
    print("这是一个字典，包含以下键和值信息:")
    for key, value in data.items():
        print(f"键: {key}")
        print(f"  类型: {type(value)}")
        # 尝试打印 shape 属性，如果没有就跳过
        shape = getattr(value, 'shape', None)
        if shape is not None:
            print(f"  形状: {shape}")
        else:
            print("  形状: 无")
        print()  # 空行分隔
else:
    print("数组类型:", data.dtype)
    print("数组形状:", data.shape)
    print("数组维度:", data.ndim)
    print("数组大小:", data.size)
    if data.ndim <= 2:
        print(data[:5])
    else:
        print(data[0])
