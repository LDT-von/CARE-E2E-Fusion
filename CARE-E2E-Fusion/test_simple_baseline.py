# -*- coding: utf-8 -*-
"""Quick sanity check: CONCH features + simple MLP on real coords."""
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
import pandas as pd

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))


class SimpleDataset(Dataset):
    def __init__(self, csv_path, coords_cache_path, max_tiles=512, seed=42):
        self.df = pd.read_csv(csv_path)
        self.max_tiles = max_tiles
        random.seed(seed)
        np.random.seed(seed)
        
        with open(coords_cache_path, 'rb') as f:
            self.coords_cache = pickle.load(f)
        
        self.df['pt_path_norm'] = self.df['pt_path'].str.replace('\\', '/')
        self.df = self.df[self.df['pt_path_norm'].apply(os.path.exists)].reset_index(drop=True)
        print('[SimpleDataset] %d samples' % len(self.df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pt_path = row['pt_path_norm']
        label = float(row['label'])
        
        tokens = torch.load(pt_path, map_location='cpu').float()
        if tokens.dim() == 3:
            tokens = tokens.squeeze(0)
        
        n = min(len(tokens), self.max_tiles)
        # Random sample tiles (not just first N)
        if len(tokens) > self.max_tiles:
            indices = random.sample(range(len(tokens)), self.max_tiles)
            indices.sort()
            tokens = tokens[indices]
        else:
            tokens = tokens[:n]
        
        return tokens, label, row['slide_id']


def collate_simple(batch):
    tokens, labels, ids = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.float32)
    max_n = max(t.shape[0] for t in tokens)
    dim = tokens[0].shape[1]
    pad_tokens = torch.zeros(len(tokens), max_n, dim)
    mask = torch.zeros(len(tokens), max_n, dtype=torch.bool)
    for i, t in enumerate(tokens):
        n = t.shape[0]
        pad_tokens[i, :n] = t
        mask[i, n:] = True
    return pad_tokens, labels, mask, list(ids)


class SimpleModel(nn.Module):
    def __init__(self, embed_dim=768, hidden=256, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, mask):
        # Average pooling with mask
        masked = x.masked_fill(mask.unsqueeze(-1), 0)
        pooled = masked.sum(dim=1) / ((~mask).sum(dim=1, keepdim=True).float() + 1e-8)
        return self.proj(pooled)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default=r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv')
    parser.add_argument('--coords_cache', type=str, default=r'c:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\tile_coords_cache.pkl')
    parser.add_argument('--max_tiles', type=int, default=512)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    dataset = SimpleDataset(args.csv_path, args.coords_cache, args.max_tiles, args.seed)

    kfold = KFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    indices = np.arange(len(dataset))
    all_aucs = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        print('\n--- Fold %d: train=%d val=%d ---' % (fold, len(train_idx), len(val_idx)))
        train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=args.batch_size,
            shuffle=True, collate_fn=collate_simple, num_workers=0, pin_memory=True)
        val_loader = DataLoader(Subset(dataset, val_idx.tolist()), batch_size=args.batch_size,
            shuffle=False, collate_fn=collate_simple, num_workers=0, pin_memory=True)

        model = SimpleModel(768, args.hidden, args.dropout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        scaler = GradScaler('cuda')
        
        best_auc = 0
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for tokens, labels, mask, ids in train_loader:
                tokens, labels, mask = tokens.to(device), labels.to(device).float(), mask.to(device)
                opt.zero_grad()
                with autocast(device_type='cuda'):
                    logits = model(tokens, mask).squeeze(-1)
                    loss = F.binary_cross_entropy_with_logits(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                total_loss += loss.item()
            
            model.eval()
            all_labels, all_probs = [], []
            with torch.no_grad():
                for tokens, labels, mask, ids in val_loader:
                    tokens, labels, mask = tokens.to(device), labels.to(device).float(), mask.to(device)
                    with autocast(device_type='cuda'):
                        logits = model(tokens, mask).squeeze(-1)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_probs.append(probs)
                    all_labels.append(labels.cpu().numpy())
            
            all_labels = np.concatenate(all_labels)
            all_probs = np.concatenate(all_probs)
            try:
                auc = roc_auc_score(all_labels, all_probs)
            except:
                auc = 0
            if auc > best_auc:
                best_auc = auc
            scheduler.step()
            if (epoch + 1) % 5 == 0:
                print('  Epoch %d: loss=%.4f auc=%.4f best=%.4f' % (epoch+1, total_loss/len(train_loader), auc, best_auc))

        all_aucs.append(best_auc)
        print('  Fold %d Best AUC: %.4f' % (fold, best_auc))

    print('\n=== Final ===')
    for i, a in enumerate(all_aucs):
        print('  Fold %d: %.4f' % (i, a))
    print('  Mean: %.4f +/- %.4f' % (np.mean(all_aucs), np.std(all_aucs)))


if __name__ == '__main__':
    main()
