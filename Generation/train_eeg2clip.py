import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
import numpy as np
import argparse
import datetime
import matplotlib.pyplot as plt
import csv
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import re
from einops.layers.torch import Rearrange
from subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from subject_layers.Embed import DataEmbedding


# Configuration constants
BASE_DIR = "/path/to/EEG2Img-Contrastive-Decoding"
DATA_DIR = os.path.join(BASE_DIR, "Data")
EEG_DATA_DIR = os.path.join(DATA_DIR, "Preprocessed_data_250Hz")
CLIP_FEATURES_TRAIN = os.path.join(DATA_DIR, "ViT-H-14_features_train.pt")
CLIP_FEATURES_TEST = os.path.join(DATA_DIR, "ViT-H-14_features_test.pt")

class ClipLoss(nn.Module):
    """CLIP loss function based on the original implementation"""
    def __init__(self):
        super().__init__()
        
    def forward(self, eeg_features, img_features, logit_scale):
        # In CLIP, logit_scale is stored as log(1/temp), so we need to exp it
        logit_scale = logit_scale.exp()
        
        # Compute similarity matrix
        logits = logit_scale * eeg_features @ img_features.T
        
        # Create labels - diagonal elements are positive pairs
        labels = torch.arange(logits.shape[0], device=logits.device, dtype=torch.long)
        
        # Calculate loss from both directions
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        
        # Return the average of the two losses
        return (loss_i + loss_t) / 2

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
        self.loss_func = ClipLoss()       
         
    def forward(self, x, subject_ids):
        x = self.encoder(x, None, subject_ids)
        eeg_embedding = self.enc_eeg(x)
        
        out = self.proj_eeg(eeg_embedding)
        return out

class EEGDataset(Dataset):
    """Simplified dataset for single subject EEG-CLIP alignment"""
    def __init__(self, subject_id, train=True):
        self.subject_id = subject_id
        self.train = train
        self.eeg_data_path = os.path.join(EEG_DATA_DIR, subject_id, 
                                         "preprocessed_eeg_training.npy" if train else "preprocessed_eeg_test.npy")
        
        # Load EEG data
        eeg_data = np.load(self.eeg_data_path, allow_pickle=True)
        self.eeg_data = eeg_data['preprocessed_eeg_data']
        
        # Load CLIP features
        features_path = CLIP_FEATURES_TRAIN if train else CLIP_FEATURES_TEST
        clip_features = torch.load(features_path)
        self.img_features = clip_features['img_features']
        
        # Process data based on train/test mode
        if train:
            # Training: 1654 classes, 10 images/class, 4 EEG samples/image
            # Shape: (16540, 4, 63, 250) -> flatten to (16540*4, 63, 250)
            self.eeg_data = self.eeg_data.reshape(-1, 63, 250)
            # For training, we have 4 EEG samples per image
            # We repeat the img_features to match the number of EEG samples
            # Shape: (16540, 1024) -> (16540*4, 1024)
            self.img_features = np.repeat(self.img_features, 4, axis=0)  # Repeat for each EEG sample
        else:
            # Testing: 200 classes, 1 image/class, 80 EEG samples/image
            # For test dataset, we'll handle averaging in the evaluation function
            pass
            
    def __len__(self):
        if self.train:
            return len(self.eeg_data)  # 16540 * 4 = 66160 samples
        else:
            return 200  # 200 classes for test
        
    def __getitem__(self, idx):
        if self.train:
            # Training: return individual EEG samples
            eeg_sample = self.eeg_data[idx]
            img_feature = self.img_features[idx]
            return torch.FloatTensor(eeg_sample), torch.FloatTensor(img_feature)
        else:
            # Testing: return all EEG samples for a class
            eeg_samples = self.eeg_data[idx]  # Shape: (80, 63, 250)
            img_feature = self.img_features[idx]
            return torch.FloatTensor(eeg_samples), torch.FloatTensor(img_feature)

def train_epoch(model, dataloader, optimizer, device, alpha=0.9, subject='sub-08'):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    mse_loss_fn = nn.MSELoss()
    
    # Preload all image features for the training set
    img_features_all = dataloader.dataset.img_features
    img_features_all = torch.FloatTensor(img_features_all).to(device)
    
    for eeg_data, img_features in tqdm(dataloader, desc="Training"):
        eeg_data = eeg_data.to(device)
        img_features = img_features.to(device)
        
        optimizer.zero_grad()

        batch_size = eeg_data.size(0)  # Assume the first element is the data tensor
        subject_id = extract_id_from_string(subject)
        subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)

        eeg_features = model(eeg_data, subject_ids)
        
        # Compute losses
        img_loss = model.loss_func(eeg_features, img_features, model.logit_scale)
        regress_loss = mse_loss_fn(eeg_features, img_features)
        loss = alpha * regress_loss * 10 + (1 - alpha) * img_loss * 10
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Compute accuracy
        logits = model.logit_scale.exp() * eeg_features @ img_features_all.T
        predicted = torch.argmax(logits, dim=1)
        labels = torch.arange(eeg_data.size(0), device=device)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

def evaluate(model, dataloader, device, k_values=[2, 4, 10, 50, 100, 200], subject='sub-08'):
    """Evaluate the model on the test set"""
    model.eval()
    img_features_all = dataloader.dataset.img_features
    img_features_all = torch.FloatTensor(img_features_all).to(device)
    
    results = {}
    
    with torch.no_grad():
        for k in k_values:
            total = 0
            correct = 0
            top5_correct = 0
            
            for idx, (eeg_data, img_feature) in enumerate(dataloader):

                eeg_data = eeg_data.squeeze(0).to(device)  # 移除batch维度，得到(80, 63, 250)
                eeg_data_mean = torch.mean(eeg_data, dim=0, keepdim=True)  # (1, 63, 250)

                batch_size = eeg_data_mean.size(0)  # Assume the first element is the data tensor
                subject_id = extract_id_from_string(subject)
                subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)

                eeg_feature = model(eeg_data_mean, subject_ids)
                
                img_feature = img_feature.to(device).unsqueeze(0)
                                
                # 为k-shot评估选择随机类别
                possible_classes = list(set(range(200)) - {idx})
                selected_classes = random.sample(possible_classes, k-1) + [idx]
                selected_img_features = img_features_all[selected_classes]
                
                # 计算logits
                logits = model.logit_scale.exp() * eeg_feature @ selected_img_features.T
                predicted_idx = torch.argmax(logits, dim=1).item()
                predicted = selected_classes[predicted_idx]
                
                # 检查预测是否正确
                if predicted == idx:
                    correct += 1
                
                # 检查top-5准确率
                if k >= 5:
                    _, top5_indices = torch.topk(logits, 5, dim=1)
                    top5_classes = [selected_classes[i] for i in top5_indices[0].cpu().numpy()]
                    if idx in top5_classes:
                        top5_correct += 1
                
                total += 1
            
            results[f"acc_{k}"] = correct / total
            if k >= 5:
                results[f"top5_acc_{k}"] = top5_correct / total
    
    return results

def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None

def main():
    parser = argparse.ArgumentParser(description='EEG-CLIP Alignment Training')
    parser.add_argument('--subject', type=str, default='sub-08', help='Subject ID to train on')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--gpu', type=str, default='cuda:0', help='GPU device to use')
    parser.add_argument('--save_interval', type=int, default=5, help='Save model every N epochs')
    args = parser.parse_args()
    
    # Set up device
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(f"{BASE_DIR}/models/EEGEncoder/{args.subject}", exist_ok=True)
    
    # Initialize model and optimizer
    model = EEGEncoder().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Load datasets
    train_dataset = EEGDataset(args.subject, train=True)
    test_dataset = EEGDataset(args.subject, train=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,
        drop_last=True
    )
    
    # We'll use a batch size of 1 for test since we handle averaging internally
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0
    )
    
    # Training loop
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    train_losses = []
    train_accuracies = []
    test_results = {f"acc_{k}": [] for k in [2, 4, 10, 50, 100, 200]}
    test_results.update({f"top5_acc_{k}": [] for k in [10, 50, 100, 200]})
    
    best_accuracy = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, subject=args.subject)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Evaluate
        eval_results = evaluate(model, test_loader, device, subject=args.subject)
        
        # Store results
        for k in [2, 4, 10, 50, 100, 200]:
            test_results[f"acc_{k}"].append(eval_results[f"acc_{k}"])
            if k >= 10:  # Only track top-5 for k >= 10
                test_results[f"top5_acc_{k}"].append(eval_results[f"top5_acc_{k}"])
        
        # Check for best model
        current_accuracy = eval_results["acc_200"]
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 
                      f"{BASE_DIR}/models/EEGEncoder/{args.subject}/best_model.pth")
        
        # Save model periodically
        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), 
                      f"{BASE_DIR}/models/EEGEncoder/{args.subject}/epoch_{epoch+1}.pth")
        
        # Print results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test Acc (k=2): {eval_results['acc_2']:.4f}")
        print(f"Test Acc (k=4): {eval_results['acc_4']:.4f}")
        print(f"Test Acc (k=10): {eval_results['acc_10']:.4f}")
        print(f"Test Acc (k=200): {eval_results['acc_200']:.4f}")
    
    # Save results to CSV
    results_dir = f"{BASE_DIR}/outputs/EEGEncoder/{args.subject}/{current_time}"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f"{results_dir}/results.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 
                         'acc_2', 'acc_4', 'acc_10', 'acc_50', 'acc_100', 'acc_200',
                         'top5_acc_10', 'top5_acc_50', 'top5_acc_100', 'top5_acc_200'])
        
        for i in range(args.epochs):
            row = [i+1, train_losses[i], train_accuracies[i]]
            row.extend([test_results[f"acc_{k}"][i] for k in [2, 4, 10, 50, 100, 200]])
            row.extend([test_results[f"top5_acc_{k}"][i] for k in [10, 50, 100, 200]])
            writer.writerow(row)
    
    print(f"\nTraining completed. Best model (acc={best_accuracy:.4f}) saved from epoch {best_epoch}")
    print(f"Results saved to {results_dir}/results.csv")

if __name__ == "__main__":
    main()