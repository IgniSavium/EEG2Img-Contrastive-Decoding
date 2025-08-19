import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import argparse
import datetime
import csv
import numpy as np
import json

# 配置BASE_DIR
BASE_DIR = "/path/to/EEG2Img-Contrastive-Decoding"

class EEGDataset(Dataset):
    """EEG dataset that loads precomputed VAE latent features"""
    def __init__(self, data_path, exclude_subject=None, subjects=None, train=True, time_window=[0, 1.0]):
        self.train = train
        self.n_cls = 1654 if train else 200
        
        # 确保只有一个subject
        if subjects is None:
            subjects = ['sub-01']  # 默认subject
        assert len(subjects) == 1, "Only single subject is supported"
        self.subject = subjects[0]
        
        # 加载预计算的VAE latent features
        features_filename = os.path.join(BASE_DIR, 'Data', 'train_image_latent_512.pt') if train \
            else os.path.join(BASE_DIR, 'Data', 'test_image_latent_512.pt')
        saved_features = torch.load(features_filename)
        self.img_features = saved_features['image_latent']
        
        # 加载EEG数据
        if train:
            eeg_file = os.path.join(BASE_DIR, 'Data', 'Preprocessed_data_250Hz', self.subject, 'preprocessed_eeg_training.npy')
        else:
            eeg_file = os.path.join(BASE_DIR, 'Data', 'Preprocessed_data_250Hz', self.subject, 'preprocessed_eeg_test.npy')
        
        eeg_data = np.load(eeg_file, allow_pickle=True)
        preprocessed_eeg_data = torch.from_numpy(eeg_data['preprocessed_eeg_data']).float()
        
        if train:
            # 训练数据: [16540, 4, 63, 250] -> [16540 * 4, 63, 250]
            # 16540 = 1654个类别 * 10张图片
            # 4 = 每张图片有4个EEG样本
            self.data = preprocessed_eeg_data.view(-1, 63, 250)
            # 创建标签: 每个类别重复 40 次 (10张图片 * 4个EEG样本)
            self.labels = torch.repeat_interleave(torch.arange(self.n_cls), 40)
        else:
            # 测试数据: [200, 80, 63, 250] -> 对80个样本取平均 -> [200, 63, 250]
            # 200 = 200个类别
            # 80 = 每张图片有80个EEG样本
            self.data = preprocessed_eeg_data.mean(dim=1)
            self.labels = torch.arange(self.n_cls)
        
        # 应用时间窗口
        if time_window != [0, 1.0]:
            start_idx = int(time_window[0] * 250)  # 假设250Hz采样率
            end_idx = int(time_window[1] * 250)
            self.data = self.data[:, :, start_idx:end_idx]
        
        self.total_samples = len(self.data)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        x = self.data[index]
        label = self.labels[index]
        
        # 计算对应的图像特征索引
        if self.train:
            img_index = index // 4  # 每4个EEG样本对应同一个图像
        else:
            img_index = index  # 测试时每个EEG样本对应一个图像（已取平均）
        
        img_features = self.img_features[img_index]
        placeholder_txtft = torch.zeros_like(img_features)
        return x, label, "", placeholder_txtft, "", img_features

class encoder_low_level(nn.Module):
    """EEG encoder converting (batch, 63, 250) to (batch, 4, 64, 64) latent states"""
    def __init__(self, num_channels=63, sequence_length=250):
        super().__init__()
        self.subject_wise_linear = nn.Linear(sequence_length, 128)
        
        # CNN upsampler
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
        x = self.subject_wise_linear(x)  # (batch, 63, 128)
        x = x.view(x.size(0), 8064, 1, 1)  # Reshape to (batch, 8064, 1, 1)
        return self.upsampler(x)  # (batch, 4, 64, 64)

def train_model(model, dataloader, optimizer, device):
    """Training step with MAE loss"""
    model.train()
    mae_loss = nn.L1Loss()
    total_loss = 0
    
    for eeg_data, _, _, _, _, img_features in dataloader:
        eeg_data = eeg_data.to(device)
        img_features = img_features.to(device).float()
        
        optimizer.zero_grad()
        eeg_features = model(eeg_data[:, :, :250]).float()
        loss = mae_loss(eeg_features, img_features)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    torch.cuda.empty_cache()
    return total_loss / len(dataloader)  # Return loss and dummy metrics

def evaluate_model(model, dataloader, device):
    """Evaluation step with MAE loss"""
    model.eval()
    mae_loss = nn.L1Loss()
    total_loss = 0
    
    with torch.no_grad():
        for eeg_data, _, _, _, _, img_features in dataloader:
            eeg_data = eeg_data.to(device)
            eeg_features = model(eeg_data[:, :, :250]).float()
            img_features = img_features.to(device).float()
            loss = mae_loss(eeg_features, img_features)
            total_loss += loss.item()
    
    return total_loss / len(dataloader) # Return loss and dummy metrics

def main_train_loop(sub, current_time, model, train_loader, test_loader, optimizer, device, config):
    """Main training loop with logging and checkpointing"""
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    train_losses, test_losses = [], []
    results = []
    best_accuracy = 0.0

    for epoch in range(config.epochs):
        # Training
        train_loss = train_model(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        scheduler.step()

        # Evaluation
        test_loss = evaluate_model(model, test_loader, device)
        test_losses.append(test_loss)

        # Save model checkpoint
        if (epoch + 1) % 5 == 0:
            save_dir = f"{BASE_DIR}/models/contrast/encoder_low_level/{sub}/{current_time}"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{save_dir}/{epoch+1}.pth")
            print(f"Model saved to {save_dir}/{epoch+1}.pth")

        # Record results
        epoch_results = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "test_loss": test_loss,
        }
        results.append(epoch_results)
        
        # Update best model (dummy since accuracy is not used)
        if test_loss < best_accuracy:
            best_accuracy = test_loss

        print(f"Epoch {epoch+1}/{config.epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        torch.cuda.empty_cache()

    return results

def main():
    parser = argparse.ArgumentParser(description='EEG Model Training Script')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--insubject', action='store_true', default=True, help='Within-subject training')
    parser.add_argument('--subjects', nargs='+', default=['sub-08'], help='Subject IDs')
    parser.add_argument('--gpu', type=str, default='cuda:0', help='GPU device')
    args = parser.parse_args()

    # Set device
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    
    for sub in args.subjects:
        # Initialize model and optimizer
        model = encoder_low_level().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        
        # Create datasets
        dataset_args = {'data_path': f'{BASE_DIR}/Data/Preprocessed_data_250Hz'}
        if args.insubject:
            train_dataset = EEGDataset(subjects=[sub], train=True, **dataset_args)
            test_dataset = EEGDataset(subjects=[sub], train=False, **dataset_args)
        else:
            train_dataset = EEGDataset(exclude_subject=sub, train=True, **dataset_args)
            test_dataset = EEGDataset(exclude_subject=sub, train=False, **dataset_args)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                 shuffle=True, num_workers=0, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=1, 
                                shuffle=False, num_workers=0, drop_last=True)
        
        # Train and evaluate
        results = main_train_loop(sub, current_time, model, train_loader, 
                                 test_loader, optimizer, device, args)
        
        # Save results
        results_dir = os.path.join(f'{BASE_DIR}/outputs/contrast/encoder_low_level', 
                                  sub, current_time)
        os.makedirs(results_dir, exist_ok=True)
        
        filename = f"{results_dir}/{'in' if args.insubject else 'cross'}_{sub}.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f'Results saved to {filename}')

if __name__ == '__main__':
    main()