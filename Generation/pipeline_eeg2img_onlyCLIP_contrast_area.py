import os
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import argparse
import re
from einops.layers.torch import Rearrange

from datetime import datetime
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor

# 从自定义库导入必要的类
from diffusion_prior import *
from custom_pipeline_low_level import *
from subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from subject_layers.Embed import DataEmbedding

BASE_DIR = "/path/to/EEG2Img-Contrastive-Decoding"


class Config:
    def __init__(self):
        self.task_name = 'classification'  # Example task name
        self.seq_len = 250                 # Sequence length
        self.pred_len = 250                # Prediction length
        self.output_attention = False      # Whether to output attention weights
        self.d_model = 250                 # Model dimension
        self.embed = 'timeF'               # Time encoding method
        self.freq = 'h'                    # Time frequency
        self.dropout = 0.25                # Dropout rate
        self.factor = 1                    # Attention scaling factor
        self.n_heads = 4                   # Number of attention heads
        self.e_layers = 1                  # Number of encoder layers
        self.d_ff = 256                    # Dimension of the feedforward network
        self.activation = 'gelu'           # Activation function
        self.enc_in = 63                   # Encoder input dimension (example value)

class iTransformer(nn.Module):
    def __init__(self, configs, joint_train=False,  num_subjects=10):
        super(iTransformer, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout, joint_train=False, num_subjects=num_subjects)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

    def forward(self, x_enc, x_mark_enc, subject_ids=None):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc, subject_ids)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = enc_out[:, :63, :]      
        # print("enc_out", enc_out.shape)
        return enc_out

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # Revised from ShallowNet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = x.unsqueeze(1)     
        # print("x", x.shape)   
        x = self.tsconv(x)
        # print("tsconv", x.shape)   
        x = self.projection(x)
        # print("projection", x.shape)  
        return x

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x

class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )

class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

class EEGEncoder(nn.Module):    
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=2, num_features=64, num_latents=1024, num_blocks=1):
        super(EEGEncoder, self).__init__()
        default_config = Config()
        self.encoder = iTransformer(default_config)   
        self.subject_wise_linear = nn.ModuleList([nn.Linear(default_config.d_model, sequence_length) for _ in range(num_subjects)])
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))    
         
    def forward(self, x, subject_ids):
        x = self.encoder(x, None, subject_ids)
        eeg_embedding = self.enc_eeg(x)
        
        out = self.proj_eeg(eeg_embedding)
        return out

class encoder_low_level(nn.Module):
    """EEG编码器，将EEG数据转换为VAE潜变量"""
    def __init__(self, num_channels=63, sequence_length=250):
        super().__init__()
        self.subject_wise_linear = nn.Linear(sequence_length, 128)
        
        # CNN上采样器
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(8064, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 4, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        """前向传播
        
        Args:
            x: EEG数据，形状为[batch_size, 63, 250]
            
        Returns:
            VAE潜变量，形状为[batch_size, 4, 64, 64]
        """
        x = self.subject_wise_linear(x)  # (batch, 63, 128)
        x = x.view(x.size(0), 8064, 1, 1)  # 重塑为(batch, 8064, 1, 1)
        return self.upsampler(x)  # (batch, 4, 64, 64)

def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None

def eeg_to_image_pipeline(original_eeg_data, without_eeg_data_dict, subject_id, output_dir_base=None, device=None, seed=42, contrastive_scales=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]):
    """
    EEG到图像的完整推理pipeline
    
    Args:
        eeg_test_data: EEG测试数据，形状为[num_samples, 63, 250]
        subject_id: 受试者ID，如'sub-08'
        output_dir: 输出目录，用于保存中间结果和最终图像
        device: 计算设备，如果为None则自动选择
        seed: 随机种子，用于可重复性
        
    Returns:
        generated_images: 生成的PIL图像列表
    """
    # 设置随机种子以确保结果可重复
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 创建输出目录
    if output_dir_base is None:
        current_time = datetime.now().strftime("%m%d_%H%M%S")
        output_dir_base = f"{BASE_DIR}/eeg_image_reconstruction_results/{subject_id}/{current_time}"
    os.makedirs(output_dir_base, exist_ok=True)
    print(f"Results will be saved to: {os.path.abspath(output_dir_base)}")
    
    # 转换输入数据为torch tensor
    num_samples = original_eeg_data.size(0)
    print(f"Processing {num_samples} EEG samples")
    
    # Step 1: 加载 EEGEncoder
    print("\n=== Step 1: Extracting EEG features using EEGEncoder ===")
    # 加载预训练的EEGEncoder
    eeg_encoder = EEGEncoder().to(device)
    eeg_encoder_path = f"{BASE_DIR}/models/contrast/ATMS/sub-08/08-15_18-21/50.pth"

    if not os.path.exists(eeg_encoder_path):
        raise FileNotFoundError(f"EEGEncoder model not found at {eeg_encoder_path}")
    
    eeg_encoder.load_state_dict(torch.load(eeg_encoder_path, map_location=device))
    eeg_encoder.eval()
    
    # 获取subject_id的数值表示
    subject_id_num = extract_id_from_string(subject_id)
    subject_ids = torch.full((num_samples,), subject_id_num, dtype=torch.long).to(device)
    
    # 通过EEGEncoder获取特征
    with torch.no_grad():
        eeg_features = eeg_encoder(original_eeg_data, subject_ids)
        # print(f"EEG features shape: {eeg_features.shape} (expected: [num_samples, 1024])")
    
    # Step 2: 加载 DiffusionPrior
    print("\n=== Loading Diffusion Prior Model ===")
    diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1).to(device)
    diffusion_prior_path = f"{BASE_DIR}/Data/fintune_ckpts/{subject_id}/diffusion_prior.pt"
    if not os.path.exists(diffusion_prior_path):
        raise FileNotFoundError(f"Diffusion prior model not found at {diffusion_prior_path}")
    diffusion_prior.load_state_dict(torch.load(diffusion_prior_path, map_location=device))
    diffusion_prior.eval()
    pipe = Pipe(diffusion_prior, device=device)

    # Step 3: 加载 Generator
    print("\n=== Initializing Generator ===")
    generator = Generator4Embeds(
        num_inference_steps=4,
        device=device,
        img2img_strength=0.5
    )

    # 遍历所有脑区和 scale 组合
    for area, neg_eeg_data in without_eeg_data_dict.items():
        with torch.no_grad():
            eeg_features_neg = eeg_encoder(neg_eeg_data, subject_ids)

        for scale in contrastive_scales:
            # 计算对比特征
            delta = F.normalize(eeg_features - eeg_features_neg, dim=-1)
            new_guidance_raw = eeg_features + scale * delta
            new_guidance = F.normalize(new_guidance_raw, dim=-1) * eeg_features.norm(dim=-1, keepdim=True)
        
            # 构造输出目录
            output_dir = os.path.join(output_dir_base, f"contra_{area}_scale_{scale}")
            os.makedirs(output_dir, exist_ok=True)

            # 保存新特征
            feature_path = os.path.join(output_dir, "eeg_features_guidance.pt")
            torch.save(new_guidance.cpu(), feature_path)

            print(f"[INFO] Saved contrastive features to {feature_path}")

            all_clip_features = []
            print(f"Generating CLIP features one by one...")
            for i in range(num_samples):
                # 取单个样本
                batch_eeg_features = new_guidance[i].unsqueeze(0)  # [1, 1024]
                
                # 生成当前样本的CLIP特征
                with torch.no_grad():
                    batch_clip_features = pipe.generate(
                        c_embeds=batch_eeg_features,
                        num_inference_steps=50,
                        guidance_scale=5
                    )

                all_clip_features.append(batch_clip_features)
                print(f"Processed sample {i+1}/{num_samples}")
                del batch_eeg_features, batch_clip_features

            # 合并所有结果
            clip_features = torch.cat(all_clip_features, dim=0)
            # print(f"CLIP features shape: {clip_features.shape}")
            del all_clip_features

            # 保存CLIP特征
            clip_features_path = os.path.join(output_dir, "clip_features_guidance.pt")
            torch.save(clip_features.cpu(), clip_features_path)

            print(f"CLIP features saved to {clip_features_path}")
            
            os.makedirs(os.path.join(output_dir, "onlyCLIP-pipeline"), exist_ok=True)
            for i in range(num_samples):
                print(f"Processing sample {i+1}/{num_samples}...")
                
                # 设置随机生成器
                gen = torch.Generator(device=device)
                gen.manual_seed(seed + i)
                
                with torch.no_grad():
                    # 生成最终图像 - 使用同一个generator实例
                    image = generator.generate(
                        clip_features[i].unsqueeze(0),
                        generator=gen,
                        low_level_image=None,  # 传入样本特有的参数
                        low_level_latent=None    # 传入样本特有的参数
                    )

                # 保存最终图像
                
                image_path = f"{output_dir}/onlyCLIP-pipeline/generated_{i+1}.png"
                image.save(image_path)
                print(f"Saved generated image: {image_path}")
                # 及时释放图像内存
                del image
            del clip_features
            torch.cuda.empty_cache()  # 每轮清理显存

        # 清理当前 area 的 eeg_features_neg
        del eeg_features_neg
        torch.cuda.empty_cache()

    
    

def main():
    parser = argparse.ArgumentParser(description="EEG to Image Reconstruction Pipeline")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory of the project")
    parser.add_argument("--subject_id", type=str, default="sub-08", help="Subject ID (e.g., sub-08)")
    parser.add_argument("--eeg_dirname", type=str, default="Preprocessed_data_250Hz", help="Name of EEG data directory")
    parser.add_argument("--output_dirname", type=str, default=None, help="Directory to save generated images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                        help='Device to use, e.g. "cuda:0", "cuda:1", or "cpu". '
                             'If not set, automatically choose GPU if available.')
    args = parser.parse_args()

    print("EEG to Image Reconstruction Pipeline")
    print("==================================")

    if args.output_dirname is None:
        current_time = datetime.now().strftime("%m%d_%H%M%S")
        args.output_dirname = f"eeg_image_reconstruction_results/{current_time}"

    # 确定设备
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    BASE_DIR = args.base_dir
    print(f"Using base directory: {BASE_DIR}")
    
    # EEG 文件路径
    test_eeg_file = os.path.join(
        args.base_dir,
        'Data',
        args.eeg_dirname,
        args.subject_id,
        'preprocessed_eeg_test.npy'
    )
    
    # 加载 EEG 数据
    test_eeg_data_dict = np.load(test_eeg_file, allow_pickle=True)
    original_eeg_data = torch.from_numpy(test_eeg_data_dict['preprocessed_eeg_data']).float().mean(dim=1).to(device)

    # 加载所有 without_*.npy 数据
    brain_areas = ['Central', 'Frontal', 'Occipital', 'Parietal', 'Temporal']
    without_eeg_data_dict = {}

    for area in brain_areas:
        file_path = os.path.join(
            args.base_dir,
            'Data',
            args.eeg_dirname,
            args.subject_id,
            f'preprocessed_eeg_test_without_{area}.npy'
        )
        data_dict = np.load(file_path, allow_pickle=True).item()
        without_eeg_data = torch.from_numpy(data_dict['preprocessed_eeg_data']).float().mean(dim=1).to(device)
        without_eeg_data_dict[area] = without_eeg_data

    # 运行推理 pipeline
    eeg_to_image_pipeline(
        original_eeg_data=original_eeg_data,
        without_eeg_data_dict=without_eeg_data_dict,
        subject_id=args.subject_id,
        output_dir_base=os.path.join(BASE_DIR, args.output_dirname),
        device=device,
        seed=args.seed
    )

    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
