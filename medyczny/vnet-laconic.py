#!/usr/bin/env python

import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.PReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(DownSampling, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.PReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(UpSampling, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.PReLU()

    def forward(self, x):
        return self.relu(self.bn(self.up(x)))

class ResidualBlock(nn.Module):
    def __init__(self, channels, num_conv_layers):
        super(ResidualBlock, self).__init__()
        layers = []
        for _ in range(num_conv_layers):
            layers.append(ConvBlock(channels, channels))
        self.conv_layers = nn.Sequential(*layers)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.conv_layers(x)
        out += residual
        out = self.relu(out)
        return out

class VNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, width=16):
        super(VNet, self).__init__()
        
        self.conv_in = ConvBlock(in_channels, width)
        
        self.down_1 = DownSampling(width, width*2)
        self.block_1 = ResidualBlock(width*2, 2)
        
        self.down_2 = DownSampling(width*2, width*4)
        self.block_2 = ResidualBlock(width*4, 3)
        
        self.down_3 = DownSampling(width*4, width*8)
        self.block_3 = ResidualBlock(width*8, 3)
        
        self.down_4 = DownSampling(width*8, width*16)
        self.block_4 = ResidualBlock(width*16, 3)
        
        self.up_4 = UpSampling(width*16, width*8)
        self.block_5 = ResidualBlock(width*8, 3)
        
        self.up_3 = UpSampling(width*8, width*4)
        self.block_6 = ResidualBlock(width*4, 3)
        
        self.up_2 = UpSampling(width*4, width*2)
        self.block_7 = ResidualBlock(width*2, 2)
        
        self.up_1 = UpSampling(width*2, width)
        self.block_8 = ResidualBlock(width, 1)
        
        self.conv_out = nn.Conv3d(width, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv_in(x)
        x1 = self.block_1(x1)
        
        x2 = self.down_1(x1)
        x2 = self.block_2(x2)
        
        x3 = self.down_2(x2)
        x3 = self.block_3(x3)
        
        x4 = self.down_3(x3)
        x4 = self.block_4(x4)
        
        x = self.up_4(x4)
        x = x + x3
        x = self.block_5(x)
        
        x = self.up_3(x)
        x = x + x2
        x = self.block_6(x)
        
        x = self.up_2(x)
        x = x + x1
        x = self.block_7(x)
        
        x = self.up_1(x)
        x = self.block_8(x)
        
        x = self.conv_out(x)
        
        return x

class LiverTumorDataset3D(Dataset):
    def __init__(self, image_paths, mask_paths, task='liver', patch_size=(128, 128, 64), stride=32):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.task = task
        self.patch_size = patch_size
        self.stride = stride
        
        self.patches = self._preprocess_dataset()
        
    def _preprocess_dataset(self):
        patches = []
        target_label = 1 if self.task == 'liver' else 2
        
        for i, (img_path, mask_path) in enumerate(zip(self.image_paths, self.mask_paths)):
            try:
                img_nii = nib.load(img_path)
                mask_nii = nib.load(mask_path)
                
                img_data = img_nii.get_fdata().astype(np.float32)
                mask_data = mask_nii.get_fdata().astype(np.int64)
                
                w, h, d = img_data.shape
                pw, ph, pd = self.patch_size
                
                img_data = np.clip(img_data, -100, 400)
                img_data = (img_data - (-100)) / 500.0
                
                for z in range(0, d - pd + 1, self.stride):
                    for y in range(0, h - ph + 1, self.stride):
                        for x in range(0, w - pw + 1, self.stride):
                            mask_patch = mask_data[x:x+pw, y:y+ph, z:z+pd]
                            
                            if target_label in np.unique(mask_patch):
                                patches.append((i, (x, y, z)))
                
                print(f"processed volume {i+1}/{len(self.image_paths)} found {len(patches)} patches")
                
            except Exception as e:
                print(f"error processing {img_path}: {str(e)}")
                
        return patches
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        file_idx, (x, y, z) = self.patches[idx]
        
        img_path = self.image_paths[file_idx]
        mask_path = self.mask_paths[file_idx]
        
        img_nii = nib.load(img_path)
        mask_nii = nib.load(mask_path)
        
        img_data = img_nii.get_fdata().astype(np.float32)
        mask_data = mask_nii.get_fdata().astype(np.int64)
        
        pw, ph, pd = self.patch_size
        img_patch = img_data[x:x+pw, y:y+ph, z:z+pd]
        mask_patch = mask_data[x:x+pw, y:y+ph, z:z+pd]
        
        img_patch = np.clip(img_patch, -100, 400)
        img_patch = (img_patch - (-100)) / 500.0
        
        if self.task == 'liver':
            mask_patch = (mask_patch >= 1).astype(np.int64)
        else:
            mask_patch = (mask_patch == 2).astype(np.int64)
        
        img_tensor = torch.from_numpy(img_patch).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_patch).long()
        
        return img_tensor, mask_tensor

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        
        probs = torch.softmax(logits, dim=1)
        
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        dice_scores = 0
        for class_idx in range(num_classes):
            class_probs = probs[:, class_idx, ...]
            class_targets = targets_one_hot[:, class_idx, ...]
            
            intersection = (class_probs * class_targets).sum()
            cardinality = (class_probs + class_targets).sum()
            
            dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
            dice_scores += dice
            
        mean_dice = dice_scores / num_classes
        
        return 1.0 - mean_dice

def dice_coefficient(y_pred, y_true):
    epsilon = 1e-6
    y_pred = torch.softmax(y_pred, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).squeeze()
    
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum()
    
    dice = (2. * intersection + epsilon) / (union + epsilon)
    
    return dice.item()

def get_data_paths(data_dir):
    images = sorted([str(f) for f in Path(data_dir).glob('**/volume-*.nii*')])
    masks = sorted([str(f) for f in Path(data_dir).glob('**/segmentation-*.nii*')])
    
    if not images or not masks:
        raise ValueError(f"no images or masks found in {data_dir}")
    
    return images, masks

def train(model, train_loader, val_loader, optimizer, criterion, device, output_dir, num_epochs, task):
    best_val_dice = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"epoch {epoch+1}/{num_epochs}")
        for i, (images, masks) in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_dice += dice_coefficient(outputs, masks)
            
            progress_bar.set_postfix({
                'loss': train_loss / (i + 1),
                'dice': train_dice / (i + 1)
            })
        
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_dice += dice_coefficient(outputs, masks)
                
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        
        print(f"epoch {epoch+1}/{num_epochs} train loss {train_loss:.4f} train dice {train_dice:.4f} val loss {val_loss:.4f} val dice {val_dice:.4f}")
        
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            model_path = os.path.join(output_dir, f'best_vnet_{task}_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_loss': val_loss
            }, model_path)
            print(f"saved best model dice {val_dice:.4f}")
        
        if (epoch + 1) % 10 == 0:
            model_path = os.path.join(output_dir, f'vnet_{task}_checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_loss': val_loss
            }, model_path)
    
    model_path = os.path.join(output_dir, f'final_vnet_{task}_model.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_dice': val_dice,
        'val_loss': val_loss
    }, model_path)
    
    return best_val_dice

def main():
    parser = argparse.ArgumentParser(description='train vnet for liver/tumor segmentation')
    parser.add_argument('--data-dir', required=True, help='data directory')
    parser.add_argument('--output-dir', required=True, help='output directory')
    parser.add_argument('--task', choices=['liver', 'tumor'], default='liver', help='segmentation task')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--patch-width', type=int, default=128, help='patch width')
    parser.add_argument('--patch-height', type=int, default=128, help='patch height')
    parser.add_argument('--patch-depth', type=int, default=64, help='patch depth')
    parser.add_argument('--stride', type=int, default=32, help='stride')
    parser.add_argument('--seed', type=int, default=69, help='random seed')
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    image_paths, mask_paths = get_data_paths(args.data_dir)
    
    train_img, val_img, train_mask, val_mask = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=args.seed
    )
    
    print(f"training on {len(train_img)} volumes validating on {len(val_img)} volumes")
    
    patch_size = (args.patch_width, args.patch_height, args.patch_depth)
    train_dataset = LiverTumorDataset3D(
        train_img, train_mask, task=args.task, patch_size=patch_size, stride=args.stride
    )
    val_dataset = LiverTumorDataset3D(
        val_img, val_mask, task=args.task, patch_size=patch_size, stride=args.stride
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using {device}")
    
    model = VNet(in_channels=1, out_channels=2, width=16).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"total params {total_params:,}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = DiceLoss()
    
    best_dice = train(
        model, train_loader, val_loader, optimizer, criterion, 
        device, args.output_dir, args.epochs, args.task
    )
    
    print(f"training done best dice {best_dice:.4f}")
    
    return 0

if __name__ == '__main__':
    main()
