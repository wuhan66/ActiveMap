# ActiveMap-SN7 v1 数据卡

更新时间：2026-07-11

## 1. 用途

ActiveMap-SN7 v1 从 SpaceNet 7 月度影像与可追踪建筑矢量中构造可编辑地图更新样本。任务不是单次
building extraction，而是给定旧矢量对象和新时相影像，预测 `KEEP/ADD/DELETE/RESHAPE`，并为
region-time-scale 主动取证提供离线 episode。

适用范围：prior-conditioned updater、预算化 evidence selector、地图写回与消融。该版本不用于道路、
室内地图或自动驾驶 HD map 的直接数值比较。

## 2. 上游数据与完整性

- 上游：SpaceNet 7 Multi-Temporal Urban Development Challenge building train archive；
- 本机压缩包：`SN7_buildings_train.tar.gz`；
- 大小：`9161814623` bytes；
- SHA-256：`00a4c862a78da923100c59679db0917b60defbeb7d16ab44a65798b645a775bf`；
- 实际 raw root：`/mnt/mydisk/wh/ActiveMap/datasets/sn7/train/train`；
- 索引：1423 个时相，60 个 AOI；优先使用 `images_masked`、`labels_match` 和可用 UDM。

SpaceNet 7 原始影像与官方 label 不随本仓库分发。公开任何派生数据前必须单独完成上游许可审计；
下载脚本不改变上游许可条款。

## 3. 划分

固定 seed 为 `20260710`，在生成 pair/crop/episode 之前按完整 AOI 分组：

| Split | AOI | 时相行数 |
|---|---:|---:|
| train | 42 | 1002 |
| val | 9 | 212 |
| test | 9 | 209 |

同一 AOI 的所有月份、对象及其合成增强只能出现在一个 split。`audit-manifest` 已验证无跨 split AOI。

## 4. 标签推导

1. 仅使用相差 1 个月的标签快照；
2. 先按稳定 `Id` 匹配，但同 ID 质心移动超过 20 m 时拒绝强制对应；剩余对象用 STRtree 稀疏候选和
   分量内 Hungarian 匹配；
3. IoU `>=0.80` 为 KEEP，匹配但低于阈值为 RESHAPE；未匹配新/旧对象为 ADD/DELETE；
4. ADD、DELETE、RESHAPE 必须在下一连续月维持新状态；没有足够未来帧时不生成变化标签；
5. 面积小于 16 m²的 polygon 过滤，fallback 质心距离不超过 20 m；
6. 每个 pair 每类最多采样 5 个，按 seed 和事件身份哈希确定性选择。

正式 derivation version 为 `sn7-adjacent-v3-distance-gated`。超距且持续的新旧位置分别形成局部
ADD/DELETE，不能作为一个跨大范围 RESHAPE。

几何先 `make_valid` 并转换到影像 CRS。非空小建筑若未覆盖任何像素中心，只补 representative point
对应的一个像素，不对所有 polygon 使用 `all_touched`。

## 5. 样本与质量

每条 updater 样本包含 3×128×128 `image`、128×128 `prior/target/valid`、edit type、对象 ID、
geometry delta、真实 CRS/crop transform、prior/target GeoJSON geometry、clear fraction 和质量来源。

有 UDM 时 `valid=(UDM==0) ∩ 非黑影像`；无 UDM 时使用 masked image 非黑区域。无效比例超过 0.50
的 updater crop 被过滤。episode 保留低质量场景用于主动取证，但每个 evidence 在 oracle 阶段重新计算
实际 valid fraction 和质量调整成本。

通过审计的 updater v3 统计：

| 项目 | 数量 |
|---|---:|
| 总样本 | 14,314 |
| train / val / test | 10,063 / 2,111 / 2,140 |
| KEEP | 6,378 (44.56%) |
| ADD | 5,116 (35.74%) |
| DELETE | 2,050 (14.32%) |
| RESHAPE | 770 (5.38%) |
| masked-image quality | 12,628 |
| UDM + masked-image quality | 1,686 |
| 因无效比例过滤 | 258 |

数组文件 57,256 个，占用约 5.5GB。valid fraction 最小 0.5、p05 0.625、中位数 1.0；
RESHAPE 质心距离中位数 6.52m、p95 17.59m、最大 19.991m。严格 audit 为 `passed=true`：
error 0、空 mask 契约 0、valid/black 错位 0、超距 RESHAPE 0。

存在 2 条 `vector-visible / raster-indistinguishable` RESHAPE warning：矢量坐标发生亚像素变化，但
128×128 prior/target mask 相同。它们保留在完整 v3 manifest 和独立 QC 子集中；raster-only 训练消融
应排除，vector-level 评估可保留。

核心文件 SHA-256：

- `updater_samples.jsonl`: `60e935b64c59cbe9f681627368a75e5ab0cc533ae4011629b406a07f82204d58`；
- `summary.json`: `b13a1b66174859019311ab432361dfd100eff666b871250b46113a500aa1f05b`；
- `audit.json`: `f5e2257196a3aa704acfc43ad5637f7737f88ce4809ca24b35fb1a3285e47b5e`。

通过审计的 v3 evidence episodes：14,572 条、338MB，train/val/test 为
10,286/2,123/2,163；KEEP/ADD/DELETE/RESHAPE 为 6,415/5,152/2,224/781。
每条 episode 有 54–78 个 region-time-scale evidence 候选，中位数 72；60 个 AOI 无泄漏，重复 episode
ID、路径缺失、编辑契约错误均为 0。episodes 比 updater 多 258 条，来自 updater 因锚点无效而过滤、
但主动取证任务有意保留的困难事件。

Episodes 核心文件 SHA-256：

- `episodes_v3.jsonl`: `fd75717949267f96142989b6b198d9607a4c242391c800292cb063ddfb45105c`；
- `episodes_v3.summary.json`: `ce33afb1d5f05c3f65e2bf2d0c3f2edb16e7a8ccdee242a3812b67399fe8c8f2`；
- `episodes_v3.audit.json`: `d184b36bb49e3d811b4942c5aef4511e5f710486a4e7ec4beeae898690ceee8f`；
- `episodes_v3.progress.json`: `fc5a396ae7f4cc3034cbf3aa10ab400bfecd9cfb4f1d0a304796768b1b7b155e`。

## 6. 已知限制

- UDM 只覆盖一部分时相，其余样本依赖 masked-image 非黑区域作为保守质量代理；
- 两期持续性提高标签精度，但会漏掉序列末尾和持续不足两期的真实快速变化；
- 真实 RESHAPE 稀少，训练时需报告 real-only 与 train-only controlled synthetic augmentation 两组；
- labels_match 的对象跟踪误差仍可能传入 edit label；应在失败案例中单列 ID switch 与 annotation jitter；
- 建筑任务不能证明方法已泛化到道路拓扑或 HD map polyline/lane graph。

## 7. 复现与验收

正式入口为 `scripts/prepare_sn7.sh`。关键产物：

```text
manifests/sn7.parquet
manifests/sn7_split.parquet
splits/{train,val,test}_aois.txt
processed/sn7_v1/pair_counts.parquet
processed/sn7_v1/updater_v3/updater_samples.jsonl
processed/sn7_v1/updater_v3/summary.json
processed/sn7_v1/updater_v3/audit.json
processed/sn7_v1/episodes_v3.jsonl
processed/sn7_v1/qc_v3/
processed/sn7_v1/qc_v3_warnings/
```

发布或训练前要求 updater audit `passed=true`，且 error、空标签契约错误、valid/black 错位均为 0。
episode audit 同样必须 `passed=true`，并验证 episode ID、AOI split、来源路径、编辑几何契约和 derivation
version。所有主结果使用真实 val/test；测试集只在阈值和配置冻结后评估一次。
