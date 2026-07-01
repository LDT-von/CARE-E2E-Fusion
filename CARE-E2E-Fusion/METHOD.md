# DARF Method Note

## Method Name

**DARF: Dual-Adaptive Region Fusion for Whole Slide Image Analysis**

## Motivation

Whole-slide images contain many heterogeneous tissue regions. A fixed grid can
preserve spatial order, but it may split meaningful tissue structures. CARE
models WSI regions adaptively, but the region grouping is still tied to the
feature layout and preprocessing strategy. DARF adds a learnable partition
stage so that the model can discover task-relevant regions before region-level
aggregation.

## Main Pipeline

```text
CARE feature .npy
      |
      v
Tile tokens + coordinates
      |
      v
Transformer token encoding with ALiBi
      |
      +-------------------------+
      |                         |
      v                         v
Direct global branch     Dynamic region tokens
                                |
                                v
                     Learnable region partition
                                |
                                v
                     CARE-style ARM aggregation
                                |
                                v
                      Adaptive region branch
      |                         |
      +-----------+-------------+
                  |
                  v
          Fused molecular prediction
```

## Innovative Components

### 1. Learnable Region Partition

DARF uses trainable region tokens as queries and WSI tile tokens as keys and
values. Cross-attention produces a soft assignment from each learned region to
the slide tiles. This means the region boundaries are learned from data instead
of being fixed by a grid size or manual tissue segmentation rule.

### 2. Two-Stage Adaptive Region Modeling

The model first learns **where regions are**, then learns **how to aggregate
each region**. This is different from one-step pooling or one-step attention
aggregation. The separation makes the method easier to analyze and ablate.

### 3. Global-Regional Fusion

The direct branch keeps stable slide-level context, while the adaptive branch
focuses on discriminative local tissue organization. The final prediction uses
both signals, reducing the risk that the model relies only on sparse local
evidence or only on coarse global pooling.

## Expected Contribution Statement

DARF proposes a dual-adaptive WSI representation framework that combines
spatially ordered E2E token encoding with learnable region discovery and
CARE-style region attention. The method is designed for molecular marker
prediction from pathology WSIs and can be evaluated with controlled ablations
over global pooling, fixed regions, learnable partitioning, and adaptive region
aggregation.
