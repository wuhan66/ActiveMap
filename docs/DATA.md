# 数据集制作

## 1. 目录约定

```text
data/raw/                  # 原始压缩包/解压数据，不进 Git
data/manifests/            # 月份、AOI、影像、label、UDM 索引
data/processed/sn7_v1/
  updater/arrays/          # image/prior/target/valid .npy crop
  updater/updater_samples.jsonl
  updater/summary.json
  updater/audit.json
  episodes.jsonl           # 有限证据目录
  selector_samples.jsonl   # counterfactual oracle 监督
  qc/                      # 抽样叠加图
splits/                    # 固定 AOI 列表和 seed
```

`updater_samples.jsonl` 内部使用相对路径，整个 processed 目录可从一个服务器盘移动到另一个盘。

## 2. 不泄漏划分

先按 AOI 划分 train/val/test，再生成相邻月份 pair、对象 crop 和 selector 状态。一个 AOI 的任何月份、
对象和合成扰动只能属于同一 split。`audit-manifest` 会检查重复键、路径、时间格式和跨 split AOI。

SN7 v1 固定 seed `20260710`，当前 manifest 为 60 个 AOI、1423 个时相；AOI/行数分别为
train 42/1002、val 9/212、test 9/209。

## 3. 四类编辑推导

相邻月份 `M_t -> M_{t+1}` 按以下顺序匹配：

1. 若存在稳定 building/object ID，先做 ID 对应；
2. 剩余对象先用 STRtree 建立局部候选边，再在每个二部连通分量内做 Hungarian 最优匹配；
3. 匹配且 IoU 高于 `keep_iou_min` 为 `KEEP`；
4. 匹配但形状变化为 `RESHAPE`；
5. 仅新图存在为 `ADD`；仅旧图存在为 `DELETE`。

几何在匹配前执行 `make_valid`，并统一到影像 CRS。ID 缺失时每个快照生成不同临时 ID，避免把相同
行号错误当作跨时相 ID。

正式 SN7 v1 只保留间隔 1 个月的 pair。`ADD/DELETE/RESHAPE` 必须在当前月和下一连续月保持新状态；
序列末尾没有足够未来帧时不生成这三类变化。每个 pair 每类最多保留 5 个事件，按
`seed + AOI + 月份 + edit + object_id` 的 SHA-1 排名确定性抽样，不能按 object ID 取前若干个。

固定阈值：`keep_iou_min=0.80`、fallback `match_iou_min=0.20`、质心距离上限 20 m、最小面积
16 m²、`max_invalid_fraction=0.50`。

## 4. Updater 样本

每个 edit event 以新时相影像裁固定尺寸 crop：

- `image`: 最多前三通道，归一化到 `[0,1]`；
- `prior_mask`: 旧地图对象栅格化；
- `target_mask`: 新地图对象栅格化；
- `valid_mask`: 有 UDM 时为 `(UDM == 0) ∩ 非黑影像`，无 UDM 时退化为 masked image 非黑区域；
- `geometry_delta`: 目标 bbox 4 维 + 相对旧 bbox 的 delta 4 维；
- `crop_transform/CRS`: 用于把输出 mask 还原为真实坐标 Polygon。

非空 polygon 默认按像素中心栅格化；若小建筑没有覆盖任何像素中心，仅补其 representative point
所在的一个像素。该规则由 updater 数据与 counterfactual oracle 共用，不使用全局 `all_touched` 膨胀边界。

真实数据命令：

```bash
bash scripts/prepare_sn7.sh /data/activemap/sn7/train
```

正式训练前的硬门槛：

```bash
activemap audit-updater \
  data/processed/sn7_v1/updater/updater_samples.jsonl \
  data/processed/sn7_v1/updater/audit.json
```

必须满足 `passed=true`、`error_count=0`、`valid_black_mismatch_count=0`、
`empty_contract_violations=0`、`reshape_distance_violations=0`，并保留 summary、audit 和固定 seed QC 图。

长任务会原子更新 `progress.json`（episode 使用同名 `.progress.json`），包含当前 AOI、月份 pair、已写
样本/episode、类别计数和跳过计数。只有最终 JSONL/summary 才代表可供下游使用的完整数据。
episode 先流式写入隐藏临时 JSONL，完成后原子替换正式文件，因此候选目录不会全部堆在内存，也不会
向下游暴露半写文件。

## 5. Selector episode 与 oracle

一个 episode 固定候选集合，每条 evidence 包含时间、像素区域、尺度、影像/UDM 路径、clear fraction
和成本。训练好 updater 后，`build-selector-oracle` 对每条候选离线推理：

```text
quality = 0.5 * raster IoU + 0.5 * typed-edit correctness
utility = marginal quality gain - lambda_cost * cost - lambda_false * false-edit risk
```

随后按预算 `[1,2,4,8]` 展开 oracle 中间状态；每个状态只保留当前可负担候选，记录剩余预算、已选
证据、当前质量和 STOP target。真值只用于离线 utility，不进入部署时 selector feature。

## 6. Synthetic 数据用途

`generate-updater-smoke` 与 `generate-selector-smoke` 只用于 CI、接口检查和初期调试，不能作为论文结果。
真实训练可用受控 perturbation 补足稀有 `ADD/DELETE/RESHAPE`，尤其是持续性过滤后数量很少的
`RESHAPE`。合成样本只能加入 train；val/test 必须保持真实时序事件，且主表需分别报告 real-only 与
real+synthetic 训练。任何合成比例、扰动幅度和随机种子都必须落盘，并作为消融变量。

## 7. 质量控制

`render-updater-qc` 固定 seed 抽样，依次显示 image、prior、target 和 invalid overlay。全量训练前至少检查：

- 四类编辑各 50 个；
- 每个 split 与多个 AOI；
- 边界对象、空 target 的 DELETE、空 prior 的 ADD；
- UDM 对齐、CRS、异常大/小几何；
- edit 类别分布和每 AOI 数量。

## 8. 发布约束

仓库不提交原始影像、官方 label、生成 crop 或 checkpoint。公开代码/派生标注前，分别核验 SpaceNet 7
与所有复用项目的 license；文档中的 S3 下载脚本不改变数据本身的许可条件。

## 9. 外部 updater 数据

Inria Aerial Image Labeling 原始包由 `aerialimagelabeling.7z.001` 至 `.005` 五个分卷组成。
`scripts/download_inria.sh` 顺序断点下载五卷，验证每卷的官方精确字节数，运行 `7z t`，
使用第一卷解压 7z，并在存在 `NEW2-AerialImageDataset.zip` 时继续展开内部 ZIP。全部解压
成功后才写 `EXTRACTION_COMPLETE`；不能在最后一卷未完整时手工解压。

Inria 可独立于慢速 MUNO21 准备：

```bash
bash scripts/watch_prepare_inria_updater.sh logs/prepare_inria_updater.log 120
```

watcher 必须同时看到 `EXTRACTION_COMPLETE`、`train/images/*.tif` 和 `train/gt/*.tif`，再执行
`scripts/prepare_inria_updater.sh`。后者生成 `processed/inria_v1/updater`，运行严格 audit，
渲染 96 个固定 seed QC 样本，并在自动步骤成功后写 `READY`。人工检查 audit、类别/split
统计和 QC 图后才可创建 `APPROVED`；`scripts/run_inria_sn7_transfer.sh` 只接受该人工门控。
Inria builder 默认要求 `min_valid_fraction=0.5`，在写数组前丢弃黑边/越界区域超过一半的
crop，并在 summary 中记录 `skipped_low_valid`。质量策略变化时，旧版本应移入带日期的
`updater_unfiltered_*` 目录保留审计证据，不能复用旧 `READY` 或 QC。
分割预训练不得直接使用对象 updater 的单建筑 target 配合 `use_prior=false`，因为同一 crop
可能含多个建筑而目标不可判定。`scripts/prepare_inria_segmentation.sh` 另建 512 px 场景窗口，
target 包含窗口内全部官方建筑，缩放到 256 px 后写入 `processed/inria_v1/segmentation`。
现有 `audit.json` 为 `passed=true` 时准备脚本幂等跳过。`prepare_external_updaters.sh` 复用
该子脚本，因此 MUNO21 日后完成时不会重复制作已经审计通过的 Inria 数据。
