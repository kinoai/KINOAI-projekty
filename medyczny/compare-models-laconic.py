#!/usr/bin/env python

import os
import time
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy import ndimage
import pandas as pd
import seaborn as sns
import json

from unet3d_laconic import UNet3D
from vnet_laconic import VNet

class TestDataset3D(Dataset):
    def __init__(self, image_path, mask_path, task='liver', patch_size=(128, 128, 64), overlap=0.5):
        self.image_path = image_path
        self.mask_path = mask_path
        self.task = task
        self.patch_size = patch_size
        self.overlap = overlap
        
        self.img_nii = nib.load(image_path)
        self.mask_nii = nib.load(mask_path)
        
        self.img_data = self.img_nii.get_fdata()
        self.mask_data = self.mask_nii.get_fdata()
        
        self.patches = self._get_patches()
        
    def _get_patches(self):
        patches = []
        w, h, d = self.img_data.shape
        pw, ph, pd = self.patch_size
        
        stride_w = int(pw * (1 - self.overlap))
        stride_h = int(ph * (1 - self.overlap))
        stride_d = int(pd * (1 - self.overlap))
        
        stride_w = max(stride_w, 1)
        stride_h = max(stride_h, 1)
        stride_d = max(stride_d, 1)
        
        target_label = 1 if self.task == 'liver' else 2
        
        for z in range(0, d - pd + 1, stride_d):
            for y in range(0, h - ph + 1, stride_h):
                for x in range(0, w - pw + 1, stride_w):
                    mask_patch = self.mask_data[x:x+pw, y:y+ph, z:z+pd]
                    
                    if target_label in np.unique(mask_patch):
                        patches.append((x, y, z))
                        
        print(f"extracted {len(patches)} patches")
        return patches
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        x, y, z = self.patches[idx]
        
        pw, ph, pd = self.patch_size
        img_patch = self.img_data[x:x+pw, y:y+ph, z:z+pd]
        mask_patch = self.mask_data[x:x+pw, y:y+ph, z:z+pd]
        
        img_patch = np.clip(img_patch, -100, 400)
        img_patch = (img_patch - (-100)) / 500.0
        
        if self.task == 'liver':
            mask_patch = (mask_patch >= 1).astype(np.int64)
        else:
            mask_patch = (mask_patch == 2).astype(np.int64)
        
        img_tensor = torch.from_numpy(img_patch.astype(np.float32)).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_patch).long()
        
        return img_tensor, mask_tensor, (x, y, z)

def dice_coefficient(y_pred, y_true):
    epsilon = 1e-6
    
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum()
    
    dice = (2. * intersection + epsilon) / (union + epsilon)
    
    return dice

def hausdorff_distance(pred, gt, percentile=95):
    pred_boundary = pred ^ ndimage.binary_erosion(pred)
    gt_boundary = gt ^ ndimage.binary_erosion(gt)
    
    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        return 0
        
    pred_boundary_coords = np.array(np.where(pred_boundary)).T
    gt_boundary_coords = np.array(np.where(gt_boundary)).T
    
    def _hausdorff(points_a, points_b):
        distances = []
        for point in points_a:
            point_dists = np.sqrt(np.sum((points_b - point) ** 2, axis=1))
            distances.append(np.min(point_dists))
        return np.percentile(distances, percentile)
    
    h1 = _hausdorff(pred_boundary_coords, gt_boundary_coords)
    h2 = _hausdorff(gt_boundary_coords, pred_boundary_coords)
    
    return max(h1, h2)

def reconstruct_from_patches(patches, coords, volume_shape):
    volume = np.zeros(volume_shape, dtype=np.float32)
    count = np.zeros(volume_shape, dtype=np.float32)
    
    for patch, (x, y, z) in zip(patches, coords):
        pw, ph, pd = patch.shape
        volume[x:x+pw, y:y+ph, z:z+pd] += patch
        count[x:x+pw, y:y+ph, z:z+pd] += 1
    
    volume /= np.maximum(count, 1)
    
    return volume

def evaluate_model(model, dataloader, device, volume_shape, threshold=0.5):
    model.eval()
    
    patches = []
    coords = []
    masks = []
    
    inference_times = []
    
    with torch.no_grad():
        for img, mask, coord in tqdm(dataloader, desc="evaluating model"):
            img = img.to(device)
            
            start_time = time.time()
            output = model(img)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            probs = torch.softmax(output, dim=1)
            pred = (probs[0, 1] > threshold).cpu().numpy()
            
            patches.append(pred)
            coords.append(coord)
            masks.append(mask[0].cpu().numpy())
    
    pred_volume = reconstruct_from_patches(patches, coords, volume_shape)
    mask_volume = reconstruct_from_patches(masks, coords, volume_shape)
    
    pred_volume = (pred_volume > 0.5).astype(np.int64)
    mask_volume = (mask_volume > 0.5).astype(np.int64)
    
    dice = dice_coefficient(pred_volume, mask_volume)
    hausdorff = hausdorff_distance(pred_volume, mask_volume)
    
    pred_flat = pred_volume.flatten()
    mask_flat = mask_volume.flatten()
    
    precision = precision_score(mask_flat, pred_flat, zero_division=1)
    recall = recall_score(mask_flat, pred_flat, zero_division=1)
    f1 = f1_score(mask_flat, pred_flat, zero_division=1)
    
    avg_inference_time = np.mean(inference_times)
    
    metrics = {
        "dice": dice,
        "hausdorff": hausdorff,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "inference_time": avg_inference_time,
        "inference_times": inference_times
    }
    
    return metrics, pred_volume

def visualize_results(test_img, test_mask, unet_pred, vnet_pred, slice_indices, output_dir, task):
    for i, slice_idx in enumerate(slice_indices):
        if i >= 5:
            break
            
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        
        axs[0, 0].imshow(test_img[:, :, slice_idx], cmap='gray')
        axs[0, 0].set_title('original image')
        axs[0, 0].axis('off')
        
        axs[0, 1].imshow(test_img[:, :, slice_idx], cmap='gray')
        mask = test_mask[:, :, slice_idx]
        masked = np.ma.masked_where(mask == 0, mask)
        axs[0, 1].imshow(masked, cmap='autumn', alpha=0.7)
        axs[0, 1].set_title('ground truth')
        axs[0, 1].axis('off')
        
        axs[0, 2].imshow(test_img[:, :, slice_idx], cmap='gray')
        mask = unet_pred[:, :, slice_idx]
        masked = np.ma.masked_where(mask == 0, mask)
        axs[0, 2].imshow(masked, cmap='cool', alpha=0.7)
        axs[0, 2].set_title('unet3d prediction')
        axs[0, 2].axis('off')
        
        axs[1, 0].imshow(test_img[:, :, slice_idx], cmap='gray')
        mask = vnet_pred[:, :, slice_idx]
        masked = np.ma.masked_where(mask == 0, mask)
        axs[1, 0].imshow(masked, cmap='summer', alpha=0.7)
        axs[1, 0].set_title('vnet prediction')
        axs[1, 0].axis('off')
        
        axs[1, 1].imshow(test_img[:, :, slice_idx], cmap='gray')
        true_positive = np.logical_and(test_mask[:, :, slice_idx] == 1, unet_pred[:, :, slice_idx] == 1)
        false_positive = np.logical_and(test_mask[:, :, slice_idx] == 0, unet_pred[:, :, slice_idx] == 1)
        false_negative = np.logical_and(test_mask[:, :, slice_idx] == 1, unet_pred[:, :, slice_idx] == 0)
        
        tp_masked = np.ma.masked_where(true_positive == 0, true_positive)
        fp_masked = np.ma.masked_where(false_positive == 0, false_positive)
        fn_masked = np.ma.masked_where(false_negative == 0, false_negative)
        
        axs[1, 1].imshow(tp_masked, cmap='Greens', alpha=0.7)
        axs[1, 1].imshow(fp_masked, cmap='Reds', alpha=0.7)
        axs[1, 1].imshow(fn_masked, cmap='Blues', alpha=0.7)
        axs[1, 1].set_title('unet3d errors (tp=green fp=red fn=blue)')
        axs[1, 1].axis('off')
        
        axs[1, 2].imshow(test_img[:, :, slice_idx], cmap='gray')
        true_positive = np.logical_and(test_mask[:, :, slice_idx] == 1, vnet_pred[:, :, slice_idx] == 1)
        false_positive = np.logical_and(test_mask[:, :, slice_idx] == 0, vnet_pred[:, :, slice_idx] == 1)
        false_negative = np.logical_and(test_mask[:, :, slice_idx] == 1, vnet_pred[:, :, slice_idx] == 0)
        
        tp_masked = np.ma.masked_where(true_positive == 0, true_positive)
        fp_masked = np.ma.masked_where(false_positive == 0, false_positive)
        fn_masked = np.ma.masked_where(false_negative == 0, false_negative)
        
        axs[1, 2].imshow(tp_masked, cmap='Greens', alpha=0.7)
        axs[1, 2].imshow(fp_masked, cmap='Reds', alpha=0.7)
        axs[1, 2].imshow(fn_masked, cmap='Blues', alpha=0.7)
        axs[1, 2].set_title('vnet errors (tp=green fp=red fn=blue)')
        axs[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{task}_comparison_slice_{slice_idx}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_metrics_comparison(unet_metrics, vnet_metrics, output_dir, task):
    models = ['unet3d', 'vnet']
    metrics_to_plot = ['dice', 'precision', 'recall', 'f1']
    metrics_data = {
        'Model': models + models + models + models,
        'Metric': ['dice'] * 2 + ['precision'] * 2 + ['recall'] * 2 + ['f1'] * 2,
        'Value': [
            unet_metrics['dice'], vnet_metrics['dice'],
            unet_metrics['precision'], vnet_metrics['precision'],
            unet_metrics['recall'], vnet_metrics['recall'],
            unet_metrics['f1'], vnet_metrics['f1']
        ]
    }
    
    df = pd.DataFrame(metrics_data)
    
    plt.figure(figsize=(12, 6))
    chart = sns.barplot(x='Metric', y='Value', hue='Model', data=df)
    chart.set_title(f'{task} segmentation metrics', fontsize=16)
    chart.set_ylim(0, 1.0)
    
    for p in chart.patches:
        chart.annotate(f'{p.get_height():.3f}', 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha = 'center', va = 'bottom', 
                      fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{task}_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    inference_data = {
        'Model': models,
        'Inference Time (ms)': [unet_metrics['inference_time']*1000, vnet_metrics['inference_time']*1000]
    }
    
    df_time = pd.DataFrame(inference_data)
    
    plt.figure(figsize=(8, 5))
    time_chart = sns.barplot(x='Model', y='Inference Time (ms)', data=df_time)
    time_chart.set_title(f'{task} inference time', fontsize=16)
    
    for p in time_chart.patches:
        time_chart.annotate(f'{p.get_height():.2f} ms', 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha = 'center', va = 'bottom', 
                           fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{task}_inference_time_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def get_test_data_paths(data_dir):
    images = sorted([str(f) for f in Path(data_dir).glob('**/volume-*.nii*')])
    masks = sorted([str(f) for f in Path(data_dir).glob('**/segmentation-*.nii*')])
    
    if not images or not masks:
        raise ValueError(f"no images or masks found in {data_dir}")
    
    return images, masks

def main():
    parser = argparse.ArgumentParser(description='compare unet3d and vnet performance')
    parser.add_argument('--data-dir', required=True, help='test data directory')
    parser.add_argument('--unet-model', required=True, help='unet3d model path')
    parser.add_argument('--vnet-model', required=True, help='vnet model path')
    parser.add_argument('--output-dir', required=True, help='output directory')
    parser.add_argument('--task', choices=['liver', 'tumor'], default='liver', help='segmentation task')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--patch-width', type=int, default=128, help='patch width')
    parser.add_argument('--patch-height', type=int, default=128, help='patch height')
    parser.add_argument('--patch-depth', type=int, default=64, help='patch depth')
    parser.add_argument('--overlap', type=float, default=0.5, help='patch overlap')
    parser.add_argument('--threshold', type=float, default=0.5, help='prediction threshold')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    image_paths, mask_paths = get_test_data_paths(args.data_dir)
    print(f"found {len(image_paths)} test volumes")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using {device}")
    
    unet = UNet3D(in_channels=1, out_channels=2, filters=16).to(device)
    vnet = VNet(in_channels=1, out_channels=2, width=16).to(device)
    
    unet.load_state_dict(torch.load(args.unet_model, map_location=device)['model_state_dict'])
    vnet.load_state_dict(torch.load(args.vnet_model, map_location=device)['model_state_dict'])
    
    print("models loaded")
    
    all_unet_metrics = []
    all_vnet_metrics = []
    
    for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        print(f"processing volume {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
        
        patch_size = (args.patch_width, args.patch_height, args.patch_depth)
        
        test_dataset = TestDataset3D(
            img_path, mask_path, task=args.task, 
            patch_size=patch_size, overlap=args.overlap
        )
        
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        img_nii = nib.load(img_path)
        volume_shape = img_nii.get_fdata().shape
        
        unet_metrics, unet_pred_volume = evaluate_model(unet, test_loader, device, volume_shape, args.threshold)
        vnet_metrics, vnet_pred_volume = evaluate_model(vnet, test_loader, device, volume_shape, args.threshold)
        
        unet_metrics['model'] = 'unet3d'
        vnet_metrics['model'] = 'vnet'
        
        all_unet_metrics.append(unet_metrics)
        all_vnet_metrics.append(vnet_metrics)
        
        metrics = {
            'volume': os.path.basename(img_path),
            'unet3d': {k: v for k, v in unet_metrics.items() if not isinstance(v, list)},
            'vnet': {k: v for k, v in vnet_metrics.items() if not isinstance(v, list)}
        }
        
        with open(os.path.join(args.output_dir, f'{args.task}_metrics_volume_{i+1}.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        img_data = img_nii.get_fdata()
        mask_nii = nib.load(mask_path)
        mask_data = mask_nii.get_fdata()
        
        if args.task == 'liver':
            mask_data = (mask_data >= 1).astype(np.int64)
        else:
            mask_data = (mask_data == 2).astype(np.int64)
        
        target_sums = [mask_data[:, :, z].sum() for z in range(mask_data.shape[2])]
        sorted_indices = np.argsort(target_sums)[::-1]
        vis_indices = sorted_indices[:5]
        
        visualize_results(img_data, mask_data, unet_pred_volume, vnet_pred_volume, 
                         vis_indices, args.output_dir, args.task)
    
    avg_unet_metrics = {k: np.mean([m[k] for m in all_unet_metrics]) 
                      for k in all_unet_metrics[0].keys() if not isinstance(all_unet_metrics[0][k], list)}
    
    avg_vnet_metrics = {k: np.mean([m[k] for m in all_vnet_metrics])
                       for k in all_vnet_metrics[0].keys() if not isinstance(all_vnet_metrics[0][k], list)}
    
    avg_metrics = {
        'task': args.task,
        'num_volumes': len(image_paths),
        'unet3d': avg_unet_metrics,
        'vnet': avg_vnet_metrics
    }
    
    with open(os.path.join(args.output_dir, f'{args.task}_average_metrics.json'), 'w') as f:
        json.dump(avg_metrics, f, indent=4)
    
    plot_metrics_comparison(avg_unet_metrics, avg_vnet_metrics, args.output_dir, args.task)
    
    print("comparison done")
    
    return 0

if __name__ == '__main__':
    main()
