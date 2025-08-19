
import torch
import torch.nn.functional as F
import numpy as np
import os
from datetime import datetime
from torchvision import transforms
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.models.feature_extraction import create_feature_extractor

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
local_rank = 0
print("device:",device)

# current_time = datetime.now().strftime("%m%d_%H%M%S")
prev_time = '0815_194901'  # Use a fixed time for reproducibility in this example

gen_path = f'/path/to/EEG2Img-Contrastive-Decoding/Evaluation/{prev_time}/generated_imgs_tensor/all_images.pt'
gt_path = f'/path/to/EEG2Img-Contrastive-Decoding/Evaluation/{prev_time}/test_images_tensor/all_images.pt'
all_gen_images = torch.load(f'{gen_path}')
all_gt_images = torch.load(f'{gt_path}')

# print(f"all_gen_images.shape: {all_gen_images.shape}") # torch.Size([200, 512, 512, 3])
# print(f"all_gt_images.shape: {all_gt_images.shape}") # torch.Size([200, 500, 500, 3])

# Load all tensors to GPU - this may cause OOM if the tensors are large
all_gt_images = all_gt_images.to(device)
all_gen_images = all_gen_images.to(device).to(all_gt_images.dtype)

# 在您原始代码加载后立即添加归一化步骤
all_gen_images = all_gen_images.float() / 255.0  # [0, 255] -> [0, 1]
all_gt_images = all_gt_images.float() / 255.0    # [0, 255] -> [0, 1]

# 确保值在[0,1]范围内（处理可能的溢出）
all_gen_images = torch.clamp(all_gen_images, 0.0, 1.0)
all_gt_images = torch.clamp(all_gt_images, 0.0, 1.0)

# ========================
# Pixel Correlation Metric
# ========================
def resize_images(images, size=(425, 425)):
    """
    Resize a batch of images to a given size using bilinear interpolation.
    
    Args:
        images (Tensor): shape [B, H, W, C]
        size (tuple): target size (H, W)
    
    Returns:
        Tensor: resized images with shape [B, size[0], size[1], C]
    """
    # permute to [B, C, H, W] for F.interpolate
    images = images.permute(0, 3, 1, 2)
    resized = F.interpolate(images, size=size, mode='bilinear', align_corners=False)
    # permute back to [B, H, W, C]
    resized = resized.permute(0, 2, 3, 1)
    return resized

all_gen_images_resized = resize_images(all_gen_images)
all_gt_images_resized = resize_images(all_gt_images)

def compute_pixel_correlation(gen_images, gt_images):
    """
    Compute per-image pixel correlation (Pearson) between generated and ground truth images.
    
    Args:
        gen_images (Tensor): shape [B, H, W, C]
        gt_images (Tensor): shape [B, H, W, C]
        
    Returns:
        float: mean pixel correlation over all images
    """
    B, H, W, C = gen_images.shape

    # Flatten spatial and channel dimensions: [B, H*W*C]
    gen_flat = gen_images.view(B, -1)
    gt_flat = gt_images.view(B, -1)

    # Normalize per image (zero mean, unit std)
    gen_mean = gen_flat.mean(dim=1, keepdim=True)
    gen_std = gen_flat.std(dim=1, keepdim=True) + 1e-8
    gt_mean = gt_flat.mean(dim=1, keepdim=True)
    gt_std = gt_flat.std(dim=1, keepdim=True) + 1e-8

    gen_norm = (gen_flat - gen_mean) / gen_std
    gt_norm = (gt_flat - gt_mean) / gt_std

    # Compute Pearson correlation per image
    correlation = (gen_norm * gt_norm).mean(dim=1)  # [B]
    return correlation.mean().item()

# print("Computing Pixel Correlation...")
pixel_corr = compute_pixel_correlation(all_gen_images_resized, all_gt_images_resized)
print(f"Average Pixel Correlation: {pixel_corr:.4f}")

 
# ==================================
# Structural Similarity Index (SSIM)
# ==================================

all_gen_images_resized = resize_images(all_gen_images) # torch.Size([200, 425, 425, 3])
all_gt_images_resized = resize_images(all_gt_images) # torch.Size([200, 425, 425, 3])

import kornia
from kornia.metrics import ssim


def compute_ssim(gen_images, gt_images, window_size=11):
    """
    Compute SSIM between generated and ground truth images.
    
    Args:
        gen_images (Tensor): shape [B, H, W, C], values in [0,1]
        gt_images (Tensor): shape [B, H, W, C], values in [0,1]
        window_size (int): SSIM window size
        
    Returns:
        float: mean SSIM over all images
    """
    # Convert to [B, C, H, W]
    gen_tensor = gen_images.permute(0, 3, 1, 2)  # [B, 3, 425, 425]
    gt_tensor = gt_images.permute(0, 3, 1, 2)    # [B, 3, 425, 425]

    # Compute SSIM map: [B, C, H, W]
    ssim_map = ssim(gen_tensor, gt_tensor, window_size=window_size)

    # Mean over spatial and channel dims
    ssim_score = ssim_map.mean(dim=[1, 2, 3])  # [B]
    return ssim_score.mean().item()

# print("Computing SSIM...")
ssim_value = compute_ssim(all_gen_images_resized, all_gt_images_resized)
print(f"Average SSIM: {ssim_value:.4f}")

# ==================================
# AlexNet Feature Similarity
# ==================================
import torchvision.transforms as transforms
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.models.feature_extraction import create_feature_extractor

# Load AlexNet and Feature Extractor
alex_weights = AlexNet_Weights.IMAGENET1K_V1
alex_model = create_feature_extractor(
    alexnet(weights=alex_weights),
    return_nodes={'features.4': 'early', 'features.11': 'mid'}
).to(device)
alex_model.eval().requires_grad_(False)
preprocess = alex_weights.transforms()  # ✅ 官方推荐的预处理

# Compute AlexNet Similarity
def compute_alexnet_similarity(gen_images, gt_images, model, preprocess, layer_key):
    """
    Compute cosine similarity between features from AlexNet at specified layer.
    
    Args:
        gen_images (Tensor): [B, H, W, C] in [0,1]
        gt_images (Tensor): [B, H, W, C] in [0,1]
        model: feature extractor model
        preprocess: torchvision transform
        layer_key: 'early' or 'mid'
        
    Returns:
        float: mean cosine similarity
    """
    B = gen_images.shape[0]
    similarities = []

    for i in range(B):
        gen_img = gen_images[i].permute(2, 0, 1)  # [C, H, W]
        gt_img = gt_images[i].permute(2, 0, 1)    # [C, H, W]

        # Preprocess
        gen_tensor = preprocess(gen_img).unsqueeze(0).to(device)  # [1, C, H, W]
        gt_tensor = preprocess(gt_img).unsqueeze(0).to(device)

        # Extract features
        with torch.no_grad():
            gen_feat = model(gen_tensor)[layer_key].flatten()
            gt_feat = model(gt_tensor)[layer_key].flatten()

        # Cosine similarity
        cos_sim = F.cosine_similarity(gen_feat, gt_feat, dim=0).item()
        similarities.append(cos_sim)

    return np.mean(similarities)

# Evaluate AlexNet(2) - early layer
# print("Computing AlexNet(2) similarity...")
alexnet2_sim = compute_alexnet_similarity(
    all_gen_images_resized, all_gt_images_resized,
    alex_model, preprocess, 'early'
)
print(f"Average AlexNet(2) Cosine Similarity: {alexnet2_sim:.4f}")

# Evaluate AlexNet(5) - mid layer
# print("Computing AlexNet(5) similarity...")
alexnet5_sim = compute_alexnet_similarity(
    all_gen_images_resized, all_gt_images_resized,
    alex_model, preprocess, 'mid'
)
print(f"Average AlexNet(5) Cosine Similarity: {alexnet5_sim:.4f}")

# ========================
# Inception Score (IS) 
# ========================
from torchvision import models, transforms
from scipy.stats import entropy
from scipy.linalg import sqrtm
from torchvision.models import inception_v3, Inception_V3_Weights

# ================== 加载 Inception v3 模型 ==================
weights = Inception_V3_Weights.IMAGENET1K_V1 
preprocess = weights.transforms()
inception_model = models.inception_v3(weights=weights, transform_input=False).to(device)
inception_model.eval()


# ================== 工具函数：计算 Inception Score ==================
def inception_score(imgs, batch_size=32, preprocess=None, splits=1):
    """
    Computes the inception score of the generated images imgs
    imgs: torch.tensor, shape: (N, 3, H, W), range: [0,1]
    """
    N = imgs.shape[0]

    def get_pred(x):
        with torch.no_grad():
            x = preprocess(x)
            pred = inception_model(x)
            return F.softmax(pred, dim=1).data.cpu().numpy()

    # imgs: [N, H, W, 3] -> [N, 3, H, W]
    if imgs.shape[-1] == 3:
        imgs = imgs.permute(0, 3, 1, 2)

    preds = np.zeros((N, 1000))
    n_batches = int(np.ceil(N / float(batch_size)))

    for i in range(n_batches):
        batch = imgs[i * batch_size:(i + 1) * batch_size].to(device)
        batch_preds = get_pred(batch)
        preds[i * batch_size:i * batch_size + batch_preds.shape[0]] = batch_preds

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

# ================== 调用函数计算 IS ==================
# print("Calculating Inception Score for generated images...")
is_mean, is_std = inception_score(all_gen_images, batch_size=32, preprocess=preprocess, splits=1)
print(f"Inception Score: Mean = {is_mean:.4f}, Std = {is_std:.4f}")

# ========================
# CLIP Similarity
# ========================
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
from torchvision import transforms

# CLIP 模型加载
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# 图像预处理函数（用于 CLIP）
def preprocess_for_clip(images):
    # 确保输入是 tensor
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    
    # 转换形状从 (N, H, W, C) 到 (N, C, H, W) 并归一化到 [0, 1]
    if images.ndim == 4 and images.shape[-1] == 3:
        images = images.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Expected images shape (N, H, W, 3), got {images.shape}")
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    return preprocess(images)


# CLIP Similarity 计算函数
def calculate_clip_score(gen_images, gt_images, model, batch_size=32):
    model.eval()
    total_scores = []

    with torch.no_grad():
        for i in range(0, gen_images.shape[0], batch_size):
            batch_gen = gen_images[i:i+batch_size]
            batch_gt = gt_images[i:i+batch_size]

            # 预处理
            batch_gen = preprocess_for_clip(batch_gen)
            batch_gt = preprocess_for_clip(batch_gt)

            # 提取图像特征
            gen_features = model.get_image_features(pixel_values=batch_gen)
            gt_features = model.get_image_features(pixel_values=batch_gt)

            # 归一化
            gen_features = gen_features / gen_features.norm(dim=-1, keepdim=True)
            gt_features = gt_features / gt_features.norm(dim=-1, keepdim=True)

            # 余弦相似度
            similarity = (gen_features * gt_features).sum(dim=-1)
            total_scores.append(similarity.cpu().numpy())

    # 合并所有 batch 的结果
    all_scores = np.concatenate(total_scores)
    mean_clip_score = np.mean(all_scores)
    return mean_clip_score, all_scores

# 执行 CLIP Score 计算
mean_score, all_scores = calculate_clip_score(all_gen_images, all_gt_images, clip_model)

print(f"Mean CLIP Similarity: {mean_score:.4f}")


# ================================
# Frechet Inception Distance (FID)
# ================================
# ================================
# Frechet Inception Distance (FID) - 修复版本
# ================================

import numpy as np
from scipy.linalg import sqrtm
from torchvision.models import inception_v3, Inception_V3_Weights
import torch
import torch.nn.functional as F

def calculate_fid(real_features, gen_features):
    """
    Compute FID score given real and generated image features from Inception-v3.
    
    Args:
        real_features (np.ndarray): Features of real images, shape [N, D]
        gen_features (np.ndarray): Features of generated images, shape [N, D]
        
    Returns:
        float: FID score
    """
    # 确保输入是 numpy 数组
    if torch.is_tensor(real_features):
        real_features = real_features.cpu().numpy()
    if torch.is_tensor(gen_features):
        gen_features = gen_features.cpu().numpy()
    
    # 计算均值和协方差
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    mu_gen = np.mean(gen_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False)
    
    # 计算均值差平方
    diff = mu_real - mu_gen
    diff_squared = np.sum(diff ** 2)
    
    # 处理协方差矩阵的数值稳定性
    eps = 1e-6
    sigma_real = sigma_real + eps * np.eye(sigma_real.shape[0])
    sigma_gen = sigma_gen + eps * np.eye(sigma_gen.shape[0])
    
    # 计算协方差矩阵的乘积平方根
    covmean = sqrtm(sigma_real.dot(sigma_gen))
    
    # 处理复数情况
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # 计算迹
    tr_covmean = np.trace(covmean)
    tr_sigma_real = np.trace(sigma_real)
    tr_sigma_gen = np.trace(sigma_gen)
    
    # 最终 FID 公式
    fid = diff_squared + tr_sigma_real + tr_sigma_gen - 2 * tr_covmean
    
    # 处理可能出现的负数情况（由于数值误差）
    if fid < 0:
        fid = 0.0
    
    return fid

# 加载 Inception v3 模型用于 FID 计算
inception_weights = Inception_V3_Weights.IMAGENET1K_V1
inception_model = inception_v3(weights=inception_weights, transform_input=False).to(device)
inception_model.eval()

# 修改模型以输出特征（移除最后的分类层）
# Inception-v3 的最后一个池化层输出是 2048 维
inception_model.fc = torch.nn.Identity()

preprocess_fid = inception_weights.transforms()

def get_inception_features(images, model, preprocess, batch_size=32):
    """
    Extract features from images using Inception v3 model.
    
    Args:
        images (Tensor): [N, H, W, C] in [0, 1]
        model: Inception v3 model without final fc layer
        preprocess: torchvision transform
        batch_size (int): batch size for inference
        
    Returns:
        np.ndarray: features array of shape [N, 2048]
    """
    if images.shape[-1] == 3:
        images = images.permute(0, 3, 1, 2)  # to [N, C, H, W]
    
    features = []
    model.eval()
    
    with torch.no_grad():
        for i in range(0, images.shape[0], batch_size):
            batch = images[i:i+batch_size].to(device)
            batch = preprocess(batch)
            feat = model(batch)
            features.append(feat.cpu().numpy())
    
    return np.concatenate(features, axis=0)

# 确保图像大小适合 Inception-v3 (299x299)
def resize_for_inception(images, size=(299, 299)):
    """
    Resize images to 299x299 for Inception-v3.
    
    Args:
        images (Tensor): shape [B, H, W, C]
        size (tuple): target size (H, W)
        
    Returns:
        Tensor: resized images with shape [B, size[0], size[1], C]
    """
    # permute to [B, C, H, W] for F.interpolate
    images = images.permute(0, 3, 1, 2)
    resized = F.interpolate(images, size=size, mode='bilinear', align_corners=False)
    # permute back to [B, H, W, C]
    resized = resized.permute(0, 2, 3, 1)
    return resized

# 调整图像大小以适应 Inception-v3
# print("Resizing images for FID calculation...")
all_gen_images_fid = resize_for_inception(all_gen_images)
all_gt_images_fid = resize_for_inception(all_gt_images)

# 提取真实图像和生成图像的特征
# print("Extracting features for FID calculation...")
real_features = get_inception_features(all_gt_images_fid, inception_model, preprocess_fid)
gen_features = get_inception_features(all_gen_images_fid, inception_model, preprocess_fid)

# print(f"Real features shape: {real_features.shape}")
# print(f"Generated features shape: {gen_features.shape}")

# 检查特征是否有异常值
# print(f"Real features stats - mean: {np.mean(real_features):.6f}, std: {np.std(real_features):.6f}")
# print(f"Gen features stats - mean: {np.mean(gen_features):.6f}, std: {np.std(gen_features):.6f}")

# 计算 FID 分数
try:
    fid_score = calculate_fid(real_features, gen_features)
    print(f"FID Score: {fid_score:.4f}")
except Exception as e:
    print(f"Error calculating FID: {e}")
    # 尝试更稳定的计算方法
    print("Trying alternative FID calculation...")
    
    # 简化版本的 FID 计算
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(gen_features, axis=0)
    
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(gen_features, rowvar=False)
    
    # 添加数值稳定性
    sigma_real += np.eye(sigma_real.shape[0]) * 1e-6
    sigma_gen += np.eye(sigma_gen.shape[0]) * 1e-6
    
    diff = mu_real - mu_gen
    fid = np.sum(diff**2) + np.trace(sigma_real + sigma_gen - 2 * sqrtm(sigma_real @ sigma_gen).real)
    print(f"Alternative FID Score: {fid:.4f}")


import pandas as pd

results = {
    "Metric": [
        "PixCorr",
        "SSIM",
        "FID Score",
        "AlexNet(2)",
        "AlexNet(5)",
        "IS(Mean)",
        "IS(Std)",
        "CLIPSim"
    ],
    "Value": [
        pixel_corr,
        ssim_value,
        fid_score,
        alexnet2_sim,
        alexnet5_sim,
        is_mean,
        is_std,
        mean_score
    ]
}

# 创建 Pandas DataFrame
df = pd.DataFrame(results)

# 保存为 CSV 文件
df.to_csv("evaluation_results.csv", index=False)

print("Results saved to 'evaluation_results.csv'")