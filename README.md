# 融合残差稠密块与可变形卷积的轻量级超分辨率网络：RDDAN


## 环境要求 (Requirements)
- PyTorch
- NumPy
- PIL (Python Imaging Library)
- h5py
- tqdm
- matplotlib

## 项目结构 (Project Structure)
- `datasets.py`: 用于训练和评估的数据集类
- `inference.py`: 模型推理脚本
- `prepare.py`: 用于创建训练和测试数据集的数据准备脚本
- `test.py`: 具有可视化功能的综合测试脚本
- `param_search.py`: 超参数搜索实现
- `run_experiments.py`: 用于运行多个实验配置的脚本
- `plot_results.py`: 用于可视化和比较结果的工具

## 数据集准备 (Dataset Preparation)
本项目支持RGB图像处理。使用 `prepare.py` 创建数据集：

```bash
python prepare.py --train-dir "path/to/training/images" \
                 --test-dir "path/to/test/images" \
                 --train-output "dataset/train_data.h5" \
                 --test-output "dataset/test_data.h5" \
                 --patch-size 96 \
                 --stride 48 \
                 --scale 4
```

## 训练模型

使用以下命令来训练模型：

```bash
python train.py \
    --train-file "训练数据集路径" \
    --eval-file "评估数据集路径" \
    --outputs-dir "输出目录路径" \
    --scale 4 \
    --batch-size 16 \
    --num-epochs 800 \
    --num-features 64 \
    --growth-rate 32 \
    --num-blocks 3 \
    --num-layers 3 \
    --lr 1e-4
```
## 测试与评估 (Testing and Evaluation)
测试框架提供全面的评估功能：

```bash
python test.py --model-path "path/to/model/checkpoint" \
               --test-file "path/to/test/dataset" \
               --output-dir "test_results" \
               --save-results \
               --save-comparison
```

功能特点：
- PSNR和SSIM指标计算
- 并排比较可视化
- 详细的结果记录
- 支持多种模型架构

## 可视化 (Visualization)
项目包含多个可视化工具：
- 训练进度图
- PSNR与训练轮数对比
- 模型大小与性能分析
- 图像并排比较（低分辨率、超分辨率、高分辨率）

## 参数搜索 (Parameter Search)
参数搜索功能允许系统地探索不同的模型配置：

```bash
python param_search.py --data-dir "dataset" \
                      --train-file "train_data.h5" \
                      --test-file "test_data.h5" \
                      --results-dir "param_search_results"
```

## 结果比较 (Results Comparison)
使用 `plot_PSNR_Epoches.py` 比较不同的模型配置：

```bash
python plot_PSNR_Epoches.py --model-dirs path/to/model1 path/to/model2 \
                           --save-path "comparison.png" \
                           --metrics psnr ssim
```

## 贡献 (Contributing)
欢迎提交问题和改进建议！

## 许可证 (License)
本项目是开源的，基于MIT许可证发布。
