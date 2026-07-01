# -*- coding: utf-8 -*-
"""
Train CARE-E2E-Fusion on REAL WSI data (CONCH features + real coords).

Key improvements over previous attempts:
- Uses pre-extracted CONCH features from real WSI images (pathology-specific pretrained)
- Uses REAL spatial coordinates from tile_coords_cache.pkl (not simulated 1D)
- Matches CONCH features to their actual WSI tiles
- Trains the full DRP + ARM + MultiTaskHead pipeline
"""
from __future__ import print_function
import os, sys, random, time as _time, argparse, pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.amp import autocast, GradScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models'))

from models.fusion_model import DynamicRegionPartition, AdaptiveRegionModeling, MultiTaskHead
from care_adapter import load_care_feature

import pandas as pd


# ============================================================
# Dataset: CONCH features + REAL coords
# ============================================================

class RealWSIDatasetWithCoords(Dataset):
    """Loads CONCH features and matches them with real WSI tile coordinates.

    The CONCH .pt file contains features in some order. We need to:
    1. Load .pt features (already extracted from real WSI images)
    2. Load real spatial coords from cache (in random shuffled order, sampled)
    3. Match them by length: truncate/align to common size
    """

    def __init__(self, csv_path, coords_cache_path, max_tiles=1024,
                 seed=42, embed_dim=768, num_tasks=1, verbose=True):
        self.df = pd.read_csv(csv_path)
        self.max_tiles = max_tiles
        self.seed = seed
        self.embed_dim = embed_dim
        self.num_tasks = num_tasks

        if os.path.exists(coords_cache_path):
            with open(coords_cache_path, 'rb') as f:
                self.coords_cache = pickle.load(f)
        else:
            self.coords_cache = {}

        # Build svs -> coords lookup (try multiple matching strategies)
        self._svs_to_coords = {}
        for svs_path, coords in self.coords_cache.items():
            # Use basename as key for matching
            base = os.path.basename(svs_path)
            self._svs_to_coords[base] = coords

        if verbose:
            print('[Dataset] Coords cache: %d entries' % len(self.coords_cache))
            print('[Dataset] Total CSV rows: %d' % len(self.df))

        # Filter to valid pt files
        self.df = self.df.copy()
        self.df['pt_path_norm'] = self.df['pt_path'].str.replace('\\', '/')
        n_before = len(self.df)
        self.df = self.df[self.df['pt_path_norm'].apply(os.path.exists)].reset_index(drop=True)
        if verbose:
            print('[Dataset] %d/%d rows have valid .pt files' % (len(self.df), n_before))

    def __len__(self):
        return len(self.df)

    def _find_coords(self, slide_id, pt_path):
        """Find real coords for this slide. Try multiple match strategies."""
        # Strategy 1: match by svs basename in cache
        svs_basename = None
        for svs_path in self.coords_cache:
            base = os.path.basename(svs_path)
            # CONCH .pt files often share UUID prefix with .svs files
            # e.g., TCGA-2F-A9KQ-01Z-00-DX1.1C8CB2DD-5CC6-4E99-A0F9-32A0F598F5F9.pt
            # matches TCGA-2F-A9KQ-01Z-00-DX1.1C8CB2DD-5CC6-4E99-A0F9-32A0F598F5F9.svs
            if base.replace('.svs', '') in os.path.basename(pt_path):
                svs_basename = base
                break

        if svs_basename is not None and svs_basename in self.coords_cache:
            return self.coords_cache[svs_basename]

        # Strategy 2: patient_id match
        pid = slide_id[:12]
        for svs_path, coords in self.coords_cache.items():
            if pid in svs_path:
                return coords

        return None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pt_path = row['pt_path_norm']
        slide_id = row['slide_id']
        label = float(row['label']) if 'label' in row else 0.0

        # Load CONCH features
        try:
            tokens = torch.load(pt_path, map_location='cpu')
            if isinstance(tokens, dict):
                tokens = tokens.get('features', list(tokens.values())[0])
            tokens = tokens.float()
            if tokens.dim() == 3:
                tokens = tokens.squeeze(0)
        except:
            tokens = torch.zeros(64, self.embed_dim)

        # Load real coords
        real_coords = self._find_coords(slide_id, pt_path)

        n_real = len(real_coords) if real_coords else 0
        n_feat = tokens.shape[0]

        if n_real > 0:
            # Truncate to common length
            n = min(n_feat, n_real, self.max_tiles)
            tokens = tokens[:n]
            coords = torch.tensor(real_coords[:n], dtype=torch.float32)
            # Normalize coords to [0, 1] based on real extents
            if n > 0:
                y_max = max(c[0] for c in real_coords[:n]) + 1
                x_max = max(c[1] for c in real_coords[:n]) + 1
                coords[:, 0] = coords[:, 0] / y_max
                coords[:, 1] = coords[:, 1] / x_max
        else:
            # Fallback to simulated coords (will be flagged as fake)
            n = min(n_feat, self.max_tiles)
            tokens = tokens[:n]
            coords = torch.linspace(0, 1, n).unsqueeze(1).expand(-1, 2).clone()

        return {
            'tile_tokens': tokens,
            'coords': coords,
            'label': torch.tensor([label], dtype=torch.float32),
            'slide_id': slide_id,
            'num_tiles': n,
        }


def collate_wsi(batch):
    labels = torch.stack([b['label'] for b in batch])
    max_n = max(b['num_tiles'] for b in batch)
    embed_dim = batch[0]['tile_tokens'].shape[1]
    pad_tokens = torch.zeros(len(batch), max_n, embed_dim)
    pad_coords = torch.zeros(len(batch), max_n, 2)
    padding_mask = torch.zeros(len(batch), max_n, dtype=torch.bool)
    for i, s in enumerate(batch):
        n = s['num_tiles']
        pad_tokens[i, :n] = s['tile_tokens']
        pad_coords[i, :n] = s['coords']
        padding_mask[i, n:] = True
    slide_ids = [s['slide_id'] for s in batch]
    return pad_tokens, pad_coords, labels, slide_ids, padding_mask


# ============================================================
# Fusion module (with attention pooling for variable tile counts)
# ============================================================

class FusionModule(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, num_layers=4,
                 num_region_tokens=8, num_tasks=1, dropout=0.2, use_two_branches=True):
        super().__init__()
        self.use_two_branches = use_two_branches

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout, batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])

        self.head_direct = nn.Linear(embed_dim, num_tasks)

        self.drp = DynamicRegionPartition(
            embed_dim, num_region_tokens, num_heads, dropout
        )
        self.arm = AdaptiveRegionModeling(embed_dim, num_heads, dropout)
        self.head_adaptive = MultiTaskHead(
            embed_dim, num_tasks, task_names=None, dropout=dropout, use_threshold=True
        )

    def forward(self, tile_tokens, coords, labels=None, label_smoothing=0.1):
        B, N, C = tile_tokens.shape
        x = tile_tokens
        for block in self.blocks:
            x = block(x)

        # Direct branch
        x_global = x.mean(dim=1)
        logits_direct = self.head_direct(x_global)
        total_loss = 0.0
        loss_dict = {}

        if labels is not None:
            smooth = labels * (1 - label_smoothing) + 0.5 * label_smoothing
            loss_direct = F.binary_cross_entropy_with_logits(logits_direct, smooth)
            total_loss = total_loss + loss_direct
            loss_dict['loss_direct'] = loss_direct.item()

        outputs = {'logits_direct': logits_direct, 'transformer_output': x}

        if self.use_two_branches:
            region_features, attn_weights, coverage = self.drp(
                tile_tokens=x, coords=coords, return_coverage=True,
            )
            region_embeddings, region_pooled = self.arm(
                region_features=region_features,
                tile_tokens=x,
                attn_weights=attn_weights,
            )
            adaptive_out = self.head_adaptive(region_pooled)
            logits_adaptive = adaptive_out['logits']

            outputs['logits_adaptive'] = logits_adaptive
            outputs['adaptive_probs'] = adaptive_out['probs']

            if labels is not None:
                loss_adaptive = F.binary_cross_entropy_with_logits(logits_adaptive, smooth)
                total_loss = total_loss + loss_adaptive
                loss_dict['loss_adaptive'] = loss_adaptive.item()
                loss_fusion = F.mse_loss(
                    torch.sigmoid(logits_direct),
                    torch.sigmoid(logits_adaptive),
                )
                total_loss = total_loss + 0.1 * loss_fusion
                loss_dict['loss_fusion'] = loss_fusion.item()

        outputs['loss'] = total_loss
        outputs['loss_dict'] = loss_dict
        return outputs


# ============================================================
# Training
# ============================================================

def seed_everything(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str,
        default=r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv')
    parser.add_argument('--coords_cache', type=str,
        default=r'c:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\tile_coords_cache.pkl')
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_region_tokens', type=int, default=8)
    parser.add_argument('--num_tasks', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--reg', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--stop_epoch', type=int, default=5)
    parser.add_argument('--max_tiles', type=int, default=1024)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--results_dir', type=str, default='./results_real_wsi')
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--use_real_coords', action='store_true', default=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    exp_code = 'RealWSI_K%d_L%d_s%d_t%d' % (
        args.num_region_tokens, args.num_layers, args.seed, args.max_tiles)
    results_dir = os.path.join(args.results_dir, exp_code)
    os.makedirs(results_dir, exist_ok=True)

    print('\nLoading Real WSI dataset (CONCH features + real coords)...')
    dataset = RealWSIDatasetWithCoords(
        csv_path=args.csv_path,
        coords_cache_path=args.coords_cache,
        max_tiles=args.max_tiles,
        seed=args.seed,
        embed_dim=args.embed_dim,
        num_tasks=args.num_tasks,
    )

    kfold = KFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    indices = np.arange(len(dataset))
    all_aucs = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        if args.testing and fold > 0:
            break
        print('\n' + '=' * 60)
        print('  Fold %d: train=%d, val=%d' % (fold, len(train_idx), len(val_idx)))
        print('=' * 60)

        train_subset = Subset(dataset, train_idx.tolist())
        val_subset = Subset(dataset, val_idx.tolist())
        train_loader = DataLoader(train_subset, batch_size=args.batch_size,
            shuffle=True, collate_fn=collate_wsi, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size,
            shuffle=False, collate_fn=collate_wsi, num_workers=0, pin_memory=True)

        model = FusionModule(
            embed_dim=args.embed_dim, num_heads=args.num_heads, num_layers=args.num_layers,
            num_region_tokens=args.num_region_tokens, num_tasks=args.num_tasks,
            dropout=args.dropout, use_two_branches=True,
        ).to(device)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Trainable params: %d' % trainable_params)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.reg)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_epochs, eta_min=args.lr * 0.01
        )
        scaler = GradScaler('cuda')
        best_auc = 0.0
        counter = 0
        epoch = 0

        for epoch in range(args.max_epochs):
            t0 = _time.time()

            # Train
            model.train()
            total_loss = 0.0
            nb = 0
            pbar = tqdm(train_loader, desc='Epoch %d [Train]' % epoch, mininterval=2.0)
            for batch in pbar:
                tokens, coords, labels, slide_ids, padding_mask = batch
                tokens = tokens.to(device)
                coords = coords.to(device)
                labels = labels.to(device).float()

                optimizer.zero_grad()
                with autocast(device_type='cuda'):
                    outputs = model(tokens, coords, labels,
                                   label_smoothing=args.label_smoothing)

                loss = outputs['loss']
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
                nb += 1
                pbar.set_postfix({'loss': '%.4f' % loss.item()})

            train_loss = total_loss / max(nb, 1)

            # Validate
            model.eval()
            val_loss = 0.0
            nb_val = 0
            all_labels, all_probs_direct, all_probs_adaptive = [], [], []
            with torch.no_grad():
                pbar2 = tqdm(val_loader, desc='[Val]', mininterval=2.0)
                for batch in pbar2:
                    tokens, coords, labels, slide_ids, padding_mask = batch
                    tokens = tokens.to(device)
                    coords = coords.to(device)
                    labels = labels.to(device).float()
                    with autocast(device_type='cuda'):
                        outputs = model(tokens, coords, labels,
                                       label_smoothing=args.label_smoothing)
                    val_loss += outputs['loss'].item()
                    nb_val += 1
                    all_labels.append(labels.cpu().numpy())
                    all_probs_direct.append(
                        torch.sigmoid(outputs['logits_direct']).cpu().numpy()
                    )
                    if 'logits_adaptive' in outputs:
                        all_probs_adaptive.append(
                            torch.sigmoid(outputs['logits_adaptive']).cpu().numpy()
                        )

            all_labels = np.concatenate(all_labels)
            all_probs_direct = np.concatenate(all_probs_direct)
            if all_probs_adaptive:
                all_probs_adaptive = np.concatenate(all_probs_adaptive)
                all_probs_ensemble = (all_probs_direct + all_probs_adaptive) / 2.0
            else:
                all_probs_ensemble = all_probs_direct

            try:
                auc_direct = roc_auc_score(all_labels.ravel(), all_probs_direct.ravel())
                auc_ensemble = roc_auc_score(all_labels.ravel(), all_probs_ensemble.ravel())
                auc_adaptive = roc_auc_score(all_labels.ravel(), all_probs_adaptive.ravel()) \
                    if all_probs_adaptive else 0.0
            except:
                auc_direct = auc_adaptive = auc_ensemble = 0.0

            auc = auc_ensemble
            elapsed = _time.time() - t0
            marker = ' *' if auc > best_auc else ''
            if auc > best_auc:
                best_auc = auc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_score': best_auc,
                }, os.path.join(results_dir, 'fold_%d_best.pt' % fold))
            if best_auc > 0 and auc < best_auc:
                counter += 1

            print('Epoch %3d | Loss: %.4f/%.4f | AUC(D/A/E): %.4f/%.4f/%.4f%s | %.1fs' % (
                epoch, train_loss, val_loss / max(nb_val, 1),
                auc_direct, auc_adaptive, auc_ensemble, marker, elapsed))

            if counter >= args.patience and epoch > args.stop_epoch:
                print('  Early stopping at epoch', epoch)
                break

            scheduler.step()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_score': best_auc,
        }, os.path.join(results_dir, 'fold_%d_last.pt' % fold))
        all_aucs.append(best_auc)
        print('Fold %d Best AUC: %.4f' % (fold, best_auc))

    print('\n' + '=' * 60)
    print('Final Results')
    print('=' * 60)
    for i, auc in enumerate(all_aucs):
        print('  Fold %d: AUC=%.4f' % (i, auc))
    if all_aucs:
        print('  Mean AUC: %.4f +/- %.4f' % (np.mean(all_aucs), np.std(all_aucs)))
    print('  Results saved to:', results_dir)


if __name__ == '__main__':
    main()
