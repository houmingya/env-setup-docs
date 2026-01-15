# Llama Factory 完整安装教程（适配 CUDA 12.1 + Python 3.11）

本文提供一套可直接落地的 Llama Factory 安装流程，专门适配 CUDA 12.1 和 Python 3.11 环境，涵盖环境准备、核心安装、验证测试和问题排查全流程，确保安装过程顺畅、环境适配稳定。

# 一、安装前准备

## 1. 环境要求

|类别|具体要求|
|---|---|
|操作系统|Linux（Ubuntu 20.04+/CentOS 7+）、Windows 10/11（WSL2）、macOS（仅 CPU 模式）|
|Python 版本|3.11.x（适配最新版 Llama Factory，需严格匹配）|
|GPU 配置|NVIDIA 显卡 + CUDA 12.1（显存 ≥8GB，显卡驱动版本 ≥530.30.02，确保 GPU 加速正常）|
## 2. 安装 Conda（环境隔离工具）

Conda 用于创建独立的 Python 环境，避免与系统环境或其他项目依赖冲突，步骤如下：

```bash

# Linux/macOS 系统：一键安装 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Windows 系统：手动下载安装包并双击安装
# 安装包下载地址：https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
```

安装完成后，重启终端（或命令提示符），执行以下命令验证 Conda 是否安装成功：

```bash

conda --version  # 正常输出类似 "conda 23.x.x" 即代表安装成功
```

# 二、核心安装步骤

## 步骤 1：创建并激活 Python 3.11 环境

创建专门用于 Llama Factory 的 Conda 环境，指定 Python 3.11 版本：

```bash

# 创建名为 llama_factory 的环境，指定 Python 3.11
conda create -n llama_factory python=3.11 -y

# 激活环境（每次使用 Llama Factory 前必须执行此命令）
conda activate llama_factory
```

## 步骤 2：克隆 Llama Factory 官方仓库

从 GitHub 克隆最新版 Llama Factory 源码，并进入仓库目录：

```bash

# 克隆官方仓库（需提前安装 git 工具）
git clone https://github.com/hiyouga/LlamaFactory.git

# 进入仓库根目录
cd LlamaFactory
```

## 步骤 3：安装适配 CUDA 12.1 的 PyTorch

PyTorch 是 Llama Factory 运行的核心依赖，需严格安装适配 CUDA 12.1 的版本：

```bash

# 官方源安装（适合网络环境较好的情况）
pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# 国内用户推荐：使用清华源加速安装（避免下载缓慢）
pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 步骤 4：安装 Llama Factory 核心依赖

安装仓库所需的评估指标、工具库等核心依赖：

```bash

pip install -e .[metrics]
```

# 三、验证安装结果

执行以下命令逐项验证，所有步骤无报错且输出符合预期，即代表基础环境搭建完成：

```bash

# 1. 检查 Python 版本（需输出 3.11.x）
python --version

# 2. 验证 PyTorch 与 CUDA 适配性（GPU 环境关键验证步骤）
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA版本:', torch.version.cuda); print('GPU可用:', torch.cuda.is_available())"

# 3. 检查 Llama Factory 安装状态（输出版本号即正常）
llama_factory-cli -v

# 4. 运行推理测试（无报错且输出对话回复即完成）
python examples/inference/chat.py --model_name_or_path hiyouga/mini-llama-2-1.1b --template llama2
```

## 正常输出参考

```plaintext

PyTorch版本: 2.1.2+cu121
CUDA版本: 12.1
GPU可用: True
llama_factory-cli, version 0.10.x
# 推理测试会输出类似以下的模型对话回复：
Hello! How can I help you today?
```

# 四、常见问题解决

## 1. CUDA 版本不匹配（报错含 “CUDA version mismatch”）

先查看系统实际安装的 CUDA 版本，再替换对应 PyTorch 安装命令：

```bash

# 查看系统 CUDA 版本
nvcc --version

# 若为 CUDA 11.8，执行此命令安装适配版本
pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# 若无 GPU（仅 CPU 环境），执行此命令
pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
```

## 2. Python 版本报错（提示 “Package requires Python >=3.11.0”）

原因是当前环境 Python 版本低于 3.11，需重建环境：

```bash

# 退出当前环境
conda deactivate

# 删除旧环境
conda remove -n llama_factory --all -y

# 重建指定 Python 3.11 的环境并激活
conda create -n llama_factory python=3.11 -y
conda activate llama_factory
```

## 3. 依赖安装速度慢或超时

临时使用国内 PyPI 源（如清华源）加速安装：

```bash

# 格式：pip install -i 清华源地址 依赖包名
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple [依赖包名]

# 例：用清华源安装步骤 4 的核心依赖
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .[metrics]
```

# 五、关键总结

- 环境核心：Python 3.11 + CUDA 12.1 + PyTorch 2.1.2（cu121）是适配最新版 Llama Factory 的最优组合，版本需严格匹配。

- 验证重点：GPU 环境下，执行 `torch.cuda.is_available()` 必须输出 True，否则 GPU 加速无法生效。

- 避坑要点：① 手动指定 PyTorch 版本和 CUDA 版本安装，避免使用仓库一键脚本导致版本不匹配；② 每次使用 Llama Factory 前，务必先执行 `conda activate llama_factory` 激活专属环境。
> （注：文档部分内容可能由 AI 生成）