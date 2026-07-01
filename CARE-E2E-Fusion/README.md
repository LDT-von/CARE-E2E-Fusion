# DARF: Dual-Adaptive Region Fusion for WSI Analysis

This project implements **DARF**, a new dual-adaptive region fusion method for
whole-slide image analysis. It combines the spatially ordered WSI token stream
from E2E-ViT with CARE-style adaptive region modeling, but the method is not a
plain code merge: it adds a learnable region partition stage before CARE region
aggregation.

## Core Innovation

DARF introduces a two-stage adaptive region pipeline:

1. **Learnable Dynamic Region Partition**
   - Uses trainable region tokens as queries.
   - Lets the model learn which WSI tiles should form each region.
   - Avoids fixed-size hand-crafted region grouping.

2. **CARE-Inspired Adaptive Region Aggregation**
   - Uses region-level attention to aggregate pathology features.
   - Preserves CARE's strength in regional WSI representation.
   - Works on dynamically discovered regions instead of predefined ones.

3. **Dual-Branch Prediction**
   - Keeps a direct global branch for stable slide-level prediction.
   - Adds an adaptive region branch for fine-grained pathology structure.
   - Fuses the two representations for the final molecular marker prediction.

The central idea is:

```text
CARE:      patch features -> fixed/adaptive region aggregation -> prediction
E2E-ViT:   ordered WSI tokens -> transformer encoding -> prediction
DARF:      ordered WSI tokens -> learnable region partition -> CARE-style region aggregation -> fused prediction
```

## Project Contents

- `models/fusion_model.py`: the runnable fusion model.
- `train.py`: training and validation flow, including dummy-data smoke testing.
- `care_adapter.py`: loader for CARE `.npy` feature files with `feature` plus
  `index` or `coords`.
- `references/CARE`: CARE paper/code notes and the local paper PDF.
- `references/E2E-ViT`: early fusion notes from the E2E-ViT side.

Large experiment outputs, checkpoints, cache files, and Python bytecode are
ignored by `.gitignore`.

## Suggested Ablation Study

To demonstrate that DARF is a new method, compare:

| Setting | Purpose |
| --- | --- |
| Direct branch only | Tests the E2E-ViT-style global representation. |
| Adaptive branch only | Tests the region pathway independently. |
| Fixed regions + ARM | Tests CARE-style aggregation without learnable partition. |
| Learnable partition + mean pooling | Tests whether dynamic regions help without ARM. |
| Full DARF | Tests the complete dual-adaptive fusion design. |

The most important claim is that **learnable dynamic partition + CARE-style ARM**
captures stronger WSI region structure than either fixed global pooling or
single-stage region aggregation.

## Quick Check

```bash
pip install -r requirements.txt
python main.py --help
python main.py --dataset dummy --num_samples 4 --max_epochs 1 --k 2 --batch_size 2 --embed_dim 128 --num_heads 4 --num_layers 2 --num_region_tokens 4 --num_tasks 1 --task_names BAP1 --testing
```

## Train With CARE Features

Prepare a CSV containing at least:

```text
slide_id,label
19579,1
```

Then point `--data_root_dir` to the folder containing CARE `.npy` features, for
example `data/MUT/conch_v1_5/19579_0_1024.npy`.

```bash
python main.py \
  --dataset real \
  --csv_path dataset_csv/t1_gene_clean_MUT_BAP1.csv \
  --data_root_dir data/MUT/conch_v1_5 \
  --num_tasks 1 \
  --task_names BAP1 \
  --batch_size 1
```

The adapter supports CARE dictionaries shaped like:

```python
{
    "feature": features,
    "index": patch_coordinate_names,
}
```

or:

```python
{
    "feature": features,
    "coords": coords,
}
```

## Upload Notes

Upload this folder instead of the two source folders. Do not upload `results/`,
`__pycache__/`, or model checkpoints unless a platform explicitly asks for
trained weights.
