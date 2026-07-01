# 项目 1：MoTo-CARE

**Molecularly guided Topological Adaptive Region Encoding for WSI modeling**

这个项目是三篇论文融合方向的主项目：

```text
E2E-ViT spatial tokens
    -> CARE adaptive pathology regions
        -> TopoSlide topology-guided region supervision
            -> optional RNA/protein/pathway region alignment
```

它不是三个 embedding concat，而是把 topology prior 写进 patch-to-region assignment，再用 region token 预测 topology descriptor。

## 目录

```text
moto_care/            模型、loss、数据接口
scripts/smoke_test.py 最小前向+反向测试
paper/                论文方案与复查
```

## 快速测试

```bash
python scripts/smoke_test.py
```

