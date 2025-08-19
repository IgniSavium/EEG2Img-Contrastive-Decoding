import numpy as np
import os
import torch

# 文件路径
file_path = "/path/to/EEG2Img-Contrastive-Decoding/Data/Preprocessed_data_250Hz/sub-08/preprocessed_eeg_test.npy"
save_dir = "/path/to/EEG2Img-Contrastive-Decoding/Data/Preprocessed_data_250Hz/sub-08/"
os.makedirs(save_dir, exist_ok=True)

# 读取数据
data = np.load(file_path, allow_pickle=True)
eeg = torch.from_numpy(data['preprocessed_eeg_data']).float().to("cuda")  # shape: (200, 80, 63, 250)
ch_names = data['ch_names']
times = data['times']

# 脑区索引
brain_areas = {
    'Frontal': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    'Temporal': [15, 16, 24, 25, 26, 34, 35, 36, 44, 45],
    'Central': [17, 18, 19, 20, 21, 22, 23, 27, 28, 29, 30, 31, 32, 33],
    'Parietal': [37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'Occipital': [60, 61, 62, 55, 56, 57, 58, 59],
}

# 获取所有通道的索引
all_channels = list(range(eeg.shape[2]))  # 假设第2维是通道数（63）

# 对每个脑区生成“扣掉该脑区的 EEG”
for area, idxs in brain_areas.items():
    eeg_kept = eeg.clone()

    # 找出非该脑区的所有通道
    other_channels = [ch for ch in all_channels if ch not in idxs]
    # 将非该脑区的通道数据置零
    eeg_kept[:, :, other_channels, :] = 0

    eeg_kept_np = eeg_kept.cpu().numpy()
    # 保存
    save_path = os.path.join(save_dir, f'preprocessed_eeg_test_only_{area}.npy')
    np.save(save_path, {'preprocessed_eeg_data': eeg_kept_np,
                        'ch_names': ch_names,
                        'times': times}, allow_pickle=True)
    print(f"Only {area} EEG saved at {save_path}")
