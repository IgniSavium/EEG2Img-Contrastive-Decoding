chan_order = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
              'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
              'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
              'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
              'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
              'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
              'O1', 'Oz', 'O2']

# 硬编码每个脑区的通道
frontal_ch = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
              'F7', 'F5', 'F3', 'F1', 'F2', 'F4', 'F6', 'F8']

temporal_ch = ['FT9', 'FT7', 'FT8', 'FT10', 'T7', 'T8', 'TP9', 'TP7', 'TP8', 'TP10']

central_ch = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
              'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6']

parietal_ch = ['CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
               'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',]

occipital_ch = ['O1', 'Oz', 'O2', 'PO7', 'PO3', 'POz', 'PO4', 'PO8']

# 生成索引
area_idx = {
    'Frontal': [chan_order.index(ch) for ch in frontal_ch],
    'Temporal': [chan_order.index(ch) for ch in temporal_ch],
    'Central': [chan_order.index(ch) for ch in central_ch],
    'Parietal': [chan_order.index(ch) for ch in parietal_ch],
    'Occipital': [chan_order.index(ch) for ch in occipital_ch]
}

# 打印结果
for area, idxs in area_idx.items():
    print(f"{area} ({len(idxs)} channels): {idxs}")


# chan_order = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
#               'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
#               'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
#               'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
#               'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
#               'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
#               'O1', 'Oz', 'O2']

# # 定义每个脑区关键字
# brain_areas = {
#     'Frontal': ['Fp', 'AF', 'F'],       # 额叶
#     'Temporal': ['FT', 'TP', 'T'],      # 颞叶
#     'Central': ['FC', 'CP', 'C'],       # 中央区
#     'Parietal': ['PO', 'P'],            # 顶叶
#     'Occipital': ['O']                  # 枕叶
# }

# # 输出每个脑区对应的索引
# area_idx = {area: [] for area in brain_areas}

# for idx, ch in enumerate(chan_order):
#     for area, keys in brain_areas.items():
#         if any(ch.startswith(k) for k in keys):
#             area_idx[area].append(idx)
#             break  # 保证不重复划入

# # 打印结果
# for area, idxs in area_idx.items():
#     print(f"{area}: {idxs}")
