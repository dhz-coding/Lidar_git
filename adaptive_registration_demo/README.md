# 船舱点云质量评估与自适应配准 Demo

这个 demo 先完成一条清晰的验证链路：

1. 读取装沙前和装沙后的 PCD 文件或目录
2. 评价点数、有效点比例、密度均匀性、重复点和局部粗糙度
3. 根据点云质量等级生成配准初始参数
4. 分别完成装沙前多点位配准和装沙后多点位配准
5. 将装沙后结果配准到装沙前坐标系
6. 输出合并点云、前后叠加点云、参数变化记录和质量报告

## 环境

建议在项目原来的 `Lidar` 环境中运行。依赖包括：

```bash
pip install open3d numpy scipy pyyaml
```

## 运行

最简单的方式是直接修改 `config.yaml` 开头的三个路径：

```yaml
before_path: "E:/data/boat2/before"
after_path: "E:/data/boat2/after"
output_dir: "E:/data/boat2/adaptive_output"
accept_review: true
```

然后运行：

```bash
python adaptive_registration_demo.py
```

也可以用命令行参数临时覆盖配置文件中的路径。

单个 PCD 文件：

```bash
python adaptive_registration_demo.py \
  --before E:/data/boat2/before/boat_before.pcd \
  --after E:/data/boat2/after/boat_after.pcd \
  --output_dir E:/data/boat2/adaptive_output
```

目录输入：

```bash
python adaptive_registration_demo.py \
  --before E:/data/boat2/before \
  --after E:/data/boat2/after \
  --output_dir E:/data/boat2/adaptive_output
```

装沙后和前后整体配准默认会把处于 `review` 状态的结果保存并合并，同时在 JSON 报告中标记状态。装沙前内部配准固定采用严格模式，会对候选结果做反向配准一致性检查。

如果装沙前和装沙后的扫描位置明确一一对应，可以将 `config.yaml` 中的 `use_before_pose_prior` 改为 `true`，让装沙前相对变换参与装沙后 ICP 初始值。不同采集位置下必须保持 `false`，程序默认关闭这个先验。

也可以临时使用：

```bash
--accept_review
```

默认使用二进制 PCD 保存。调试时需要文本格式可以增加 `--ascii`，但文件会明显变大。

## 主要输出

每次运行会在 `output_dir/时间戳/` 下生成结果。

| 文件 | 含义 |
|---|---|
| `before_registered.pcd` | 装沙前各扫描配准合并结果 |
| `after_registered.pcd` | 装沙后各扫描配准合并结果 |
| `after_aligned_to_before.pcd` | 装沙后结果变换到装沙前坐标系 |
| `before_after_aligned_coloured.pcd` | 前后叠加查看，蓝色为装沙前，红色为装沙后 |
| `quality_and_adaptive_report.json` | 完整质量、参数和配准报告 |
| `adaptive_parameters.json` | 每个配准阶段的参数初值和重试参数 |
| `before_after_registration_report.json` | 装沙后到装沙前的每轮配准记录 |
| `before_registered_scans/` | 装沙前每个扫描的配准结果和变换矩阵 |
| `after_registered_scans/` | 装沙后每个扫描的配准结果和变换矩阵 |

## 当前自适应逻辑

点云质量使用最近邻距离中位数 `s50` 作为密度尺度。初始体素近似为：

```text
voxel_size = clip(8 × max(before_s50, after_s50) × quality_factor)
```

质量等级会影响体素、曲率百分位、FPFH 搜索半径、RANSAC 距离和 ICP 距离。装沙前和装沙后的曲率百分位分别计算，避免泥沙遮挡导致源点云有效结构不足。

配准失败后按照失败现象依次尝试：

- 降低曲率百分位并扩大特征搜索范围
- 增大体素和对应距离，关闭过强的聚类筛选
- 使用更细体素、较少 RANSAC 采样点和更宽的 ICP 捕获范围
- 关闭 FPFH 互相过滤，生成额外的粗配准候选
- 使用 `ransac_n=3` 和更低曲率百分位生成第二组候选

所有候选都会完成固定双向验证，再按 fitness、RMSE 和候选状态排序。装沙后点云特征变化较大时，额外候选可以避免单一 mutual-filter 结果直接决定最终位姿。

每次尝试都会使用同一份固定验证点云，避免验证标准随着重试参数一起变宽。

## 说明

这个版本的重点是把质量指标、参数初值和重试过程完整记录下来。每个候选还会记录双向重叠、稳定船壁重合、法向一致性、对应距离分布和多候选变换共识。质量等级阈值仍需要用你自己的船舱数据校准。`quality_and_adaptive_report.json` 是后续统计不同数据集成功率、救援成功率和错误接受率的基础。

