# FED-PsyAU

FED-PsyAU 的 MER 网络实现：SPAT → AFE → DPK-GAT → DSI → ESA。

本目录是 TAC 2026 期刊扩写版（arXiv:2507.20557 的扩写版）的代码补丁，ICCV 2025 版本仍在上级的 `FED-PsyAU/` 与 `Fed_Framework/` 中。

本目录只包含集中式 MER 部分：SPAT、AFE、DPK-GAT、DSI、ESA 以及训练、测试、预处理。联邦实验沿用上级 `Fed_Framework/`，这里不重复实现。

## 引用

```bibtex
@article{li2026fedpsyau,
  title   = {FED-PsyAU: A Unified Framework for Micro-Expression Recognition With
             Psychological Topology Learning and Federated Privacy Preservation},
  author  = {Li, Jingting and Qian, Yu and Zhao, Lin and Wang, Su-Jing},
  journal = {IEEE Transactions on Affective Computing},
  year    = {2026},
  doi     = {10.1109/TAFFC.2026.3723127}
}
```

## 环境

```
pip install -r requirements.txt
```

所有命令都在本目录内执行；`config.py` 里的 `../data`、`../data_coding`、`../experiments` 相对于当前工作目录，因此数据目录应与本目录同级。

## 数据

使用 DFME 数据集及其预计算光流。目录结构如下：

```
data/dfme_augment/
  USTC_MEv4_partA_sub0001_sub0015/
    micro_static_aligned_croped/<emotion>/<sample>/
      flow28.png  flow_pre28.png  flow_next28.png
      flow128.png flow_pre128.png flow_next128.png
      landmarks_v2.json
      au.csv
  USTC_MEv4_partB_sub0016_sub0110/
  USTC_MEv4_partC_sub0111_sub0300/
  USTC_MEv4_partD_sub0301_sub0671_200fps/
data_coding/DFME_data_coding.xlsx
```

`landmarks_v2.json` 为 161 个对齐 landmark 坐标；代码取 `[0:23] + [94:123] + [-13:]` 共 65 个 ROI，在 `flow128.png` 上裁 5×5 patch，`flow28.png` 作为全局光流。训练使用 pre / apex / next 三帧，测试只用 apex 帧。

## 数据准备

仓库提供四个脚本，按顺序执行即可从已对齐的关键帧生成上表中的特征文件：

```
python preprocess/make_landmarks.py ../data/dfme_augment
python preprocess/make_flow.py      ../data/dfme_augment
python preprocess/make_au_labels.py ../data_coding/DFME_data_coding.xlsx ../data/dfme_augment
python preprocess/check_features.py ../data/dfme_augment
```

`make_landmarks.py` 需要 dlib 和 `shape_predictor_68_face_landmarks.dat`，在 128×128 的 `apex.png` 上检测 68 点，加上 7 个几何中点（眉间、外颊），按上脸 / 鼻嘴 / 下脸三组拼成 161 点写入 `landmarks_v2.json`。不同 dlib 版本检测出的点会有 1~2 像素差异，重跑结果与论文所用数据不完全相同。

`make_flow.py` 需要 opencv-contrib（`cv2.optflow`）。它以 `first.png` 为参考帧，分别与 `apex.png`、`apex_pre.png`、`apex_next.png` 计算 TV-L1 光流，生成 28×28 与 128×128 的三通道图（光应变、垂直分量、水平分量）。

边界说明：

- 输入是已对齐并裁剪好的 `first / apex_pre / apex / apex_next` 帧。从原始 DFME 视频到对齐帧的人脸对齐与裁剪按 DFME 官方协议处理，不在本仓库内，因为裁剪框依赖 dlib 的具体版本。
- `au.csv` 由 `DFME_data_coding.xlsx` 的 `AU_Split` 列按逗号切分得到；上游存在 `23`、`24` 这类拼接编码，不在 12 类内，训练时会被忽略。
- 12×12 先验矩阵 `D` 对应论文 Sec. IV-B2，由 DFME 训练集的 AU 共现统计得到，数值已直接写在 `models/dpk_gat.py` 中。
- 受 DFME 使用协议限制，数据集及衍生特征不随代码分发。

## 训练

```
python train.py --excel_path ../data_coding/DFME_data_coding.xlsx \
                --data_path ../data/dfme_augment
```

默认按被试 7:3 随机划分、跑 10 次独立运行，输出 UF1 / UAR 的均值与标准差，最优模型保存到 `../experiments/fed_psyau.pth`。超参见 `config.py`，可用同名命令行参数覆盖。

## 测试

```
python test.py --checkpoint ../experiments/fed_psyau.pth
```
