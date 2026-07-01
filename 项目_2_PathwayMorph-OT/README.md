# 项目 2：PathwayMorph-OT

**Interpretable Pathway-to-Morphology Optimal Transport for multimodal pathology**

这个项目是第二个独立方向，不是三篇论文的直接融合。它把：

```text
morphology-topology atoms
    <-> pathway / mutation / clinical tokens
```

写成 unbalanced optimal transport matching 问题，用 transport plan 解释“哪个分子通路对应哪个病理区域”。

## 目录

```text
pathway_morph_ot/     模型、Sinkhorn OT、数据接口
scripts/smoke_test.py 最小测试
paper/                论文方案与问题复查
```

