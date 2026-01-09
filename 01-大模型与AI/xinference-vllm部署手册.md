# Xinference & vLLM 部署手册

本手册提供 Xinference 和 vLLM 的完整部署、配置和使用指南。

---

## 目录

- [一、Xinference 部署](#一xinference-部署)
- [二、vLLM 模型服务部署](#二vllm-模型服务部署)
- [三、常见问题与注意事项](#三常见问题与注意事项)

---

## 一、Xinference 部署

Xinference 是一个强大的模型推理框架，支持多种大语言模型的部署和管理。

### 1.1 创建虚拟环境

使用 Conda 创建独立的 Python 虚拟环境：

```bash
conda create -n xinference python=3.10
```

### 1.2 安装依赖

#### 步骤 1：检查 CUDA 版本

首先安装 CUDA 工具包并查看 CUDA 版本：

```bash
# 安装 CUDA 工具包（需要 root 权限）
apt install nvidia-cuda-toolkit

# 查看 CUDA 版本
nvcc --version
```

> **重要：** 记录显示的 CUDA 版本号（如 12.4），后续安装 PyTorch 时需要保持一致。

#### 步骤 2：安装 PyTorch

根据 CUDA 版本选择对应的 PyTorch 安装命令。

**方法一：使用 pip 安装**

```bash
# 注意：cu124 中的版本号需要与 CUDA 版本一致
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**方法二：使用 mamba/conda 安装**

```bash
# 注意：pytorch-cuda=12.4 中的版本号需要与 CUDA 版本一致
mamba install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# 如果没有安装 mamba，也可以使用 conda
# conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

> **版本对应说明：**
> - CUDA 11.8 → `cu118` 或 `pytorch-cuda=11.8`
> - CUDA 12.1 → `cu121` 或 `pytorch-cuda=12.1`
> - CUDA 12.4 → `cu124` 或 `pytorch-cuda=12.4`

#### 步骤 3：安装 Xinference

安装 Xinference 及其 vLLM 扩展：

```bash
pip install "xinference[vllm]"
```

**其他可选安装选项：**
```bash
# 仅安装基础版本
pip install xinference

# 安装带 transformers 支持
pip install "xinference[transformers]"

# 安装所有扩展
pip install "xinference[all]"
```

### 1.3 启动 Xinference

#### 基本启动命令

激活环境并启动服务：

```bash
# 激活虚拟环境
conda activate xinference

# 启动 Xinference 服务
xinference-local --host 0.0.0.0 --port 8188
```

**参数说明：**
- `--host 0.0.0.0`：允许所有 IP 访问（用于远程访问）
- `--port 8188`：指定服务端口号

#### 指定模型缓存位置

如果需要自定义模型存储路径，使用环境变量 `XINFERENCE_HOME`：

```bash
XINFERENCE_HOME=/data/xinference/models xinference-local --host 0.0.0.0 --port 8188
```

> **提示：** 确保指定的路径有足够的存储空间，大模型可能占用数十 GB。

#### 后台运行模式

使用 nohup 在后台持续运行服务：

```bash
sudo nohup xinference-local --host 0.0.0.0 --port 8188 > ~/xinference.log 2>&1 &
```

**后台运行说明：**
- `nohup`：即使终端关闭，进程仍继续运行
- `> ~/xinference.log`：将标准输出重定向到日志文件
- `2>&1`：将错误输出也重定向到同一日志文件
- `&`：在后台运行

**查看日志：**
```bash
# 实时查看日志
tail -f ~/xinference.log

# 查看最后 100 行日志
tail -n 100 ~/xinference.log
```

**停止后台服务：**
```bash
# 查找进程 ID
ps aux | grep xinference

# 停止进程（替换 <PID> 为实际的进程 ID）
kill <PID>

# 强制停止
kill -9 <PID>
```

### 1.4 模型下载源配置

> **重要提示：** 请根据网络环境配置合适的模型下载源。

在中国大陆环境下，建议配置使用 ModelScope 或其他国内镜像源以加速模型下载。

**设置环境变量：**
```bash
# 使用 ModelScope
export XINFERENCE_MODEL_SRC=modelscope

# 或在启动时指定
XINFERENCE_MODEL_SRC=modelscope xinference-local --host 0.0.0.0 --port 8188
```

### 1.5 访问 Web UI

启动成功后，在浏览器中访问：

```
http://<服务器IP>:8188
```

通过 Web 界面可以：
- 浏览可用模型
- 部署和管理模型
- 测试模型推理
- 查看系统资源使用情况

---

## 二、vLLM 模型服务部署

vLLM 是一个高性能的大语言模型推理引擎，支持高吞吐量和低延迟的模型服务。

### 2.1 准备工作

#### 进入模型目录

切换到模型文件所在的目录：

```bash
cd /path/to/your/models
```

**示例：**
```bash
cd /home/username/projects/vllm/pretrained_models
```

### 2.2 启动模型服务

使用 vLLM 启动模型推理服务：

```bash
VLLM_USE_MODELSCOPE=true vllm serve <模型名称> \
  --tensor-parallel-size 8 \
  --max-model-len 131072 \
  --max-num-seqs 128 \
  --port 8188
```

**完整示例（Qwen3 模型）：**
```bash
VLLM_USE_MODELSCOPE=true vllm serve Qwen3-235B-A22B-Instruct-2507 \
  --tensor-parallel-size 8 \
  --max-model-len 131072 \
  --max-num-seqs 128 \
  --port 8188
```

**参数详解：**

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `VLLM_USE_MODELSCOPE` | 使用 ModelScope 下载模型 | `true` |
| `--tensor-parallel-size` | 张量并行度（GPU 数量） | `8`（使用 8 张 GPU） |
| `--max-model-len` | 最大上下文长度（tokens） | `131072` |
| `--max-num-seqs` | 最大并发请求数 | `128` |
| `--port` | 服务端口号 | `8188` |

**其他常用参数：**

```bash
# GPU 显存利用率（默认 0.9）
--gpu-memory-utilization 0.95

# 指定使用的 GPU
--tensor-parallel-size 4

# 启用量化（如 AWQ、GPTQ）
--quantization awq

# 设置数据类型
--dtype bfloat16

# 后台运行
nohup vllm serve <模型名称> [参数] > vllm.log 2>&1 &
```

### 2.3 测试模型服务

#### 使用 cURL 测试

启动服务后，使用 cURL 命令测试 API 接口：

```bash
curl http://localhost:8188/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-235B-A22B-Instruct-2507",
    "messages": [
      {"role": "user", "content": "你好，请介绍一下你自己。"}
    ]
  }'
```

**测试参数说明：**
- `model`：模型名称（需与启动时的名称一致）
- `messages`：对话消息列表
  - `role`：角色类型（`system`、`user`、`assistant`）
  - `content`：消息内容

#### 带高级参数的测试

```bash
curl http://localhost:8188/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-235B-A22B-Instruct-2507",
    "messages": [
      {"role": "system", "content": "你是一个有帮助的AI助手。"},
      {"role": "user", "content": "什么是机器学习？"}
    ],
    "temperature": 0.7,
    "max_tokens": 2000,
    "top_p": 0.9,
    "stream": false
  }'
```

**参数说明：**
- `temperature`：采样温度（0-2），越高越随机
- `max_tokens`：生成的最大 token 数
- `top_p`：核采样参数（0-1）
- `stream`：是否流式输出

#### 使用 Python 测试

```python
import requests
import json

url = "http://localhost:8188/v1/chat/completions"
headers = {"Content-Type": "application/json"}

data = {
    "model": "Qwen3-235B-A22B-Instruct-2507",
    "messages": [
        {"role": "user", "content": "你好，请介绍一下你自己。"}
    ],
    "temperature": 0.7,
    "max_tokens": 2000
}

response = requests.post(url, headers=headers, json=data)
result = response.json()

print(json.dumps(result, ensure_ascii=False, indent=2))
```

---

## 三、常见问题与注意事项

### 3.1 CUDA 版本不匹配

**问题：** 安装的 PyTorch CUDA 版本与系统 CUDA 版本不匹配。

**解决方案：**
1. 检查系统 CUDA 版本：`nvcc --version` 或 `nvidia-smi`
2. 安装对应版本的 PyTorch
3. 如果无法匹配，考虑升级或降级系统 CUDA

### 3.2 显存不足

**问题：** 加载模型时显存不足（OOM）。

**解决方案：**
```bash
# 减小最大上下文长度
--max-model-len 32768

# 降低 GPU 显存利用率
--gpu-memory-utilization 0.8

# 使用量化模型
--quantization awq

# 增加张量并行度（使用更多 GPU）
--tensor-parallel-size 16
```

### 3.3 端口被占用

**问题：** 指定的端口已被其他服务占用。

**解决方案：**
```bash
# 查看端口占用情况
netstat -tulpn | grep 8188

# 或使用 lsof
lsof -i :8188

# 更换端口号
--port 8189
```

### 3.4 模型下载慢或失败

**问题：** 从 HuggingFace 下载模型速度慢或连接失败。

**解决方案：**
```bash
# 使用 ModelScope 镜像
export VLLM_USE_MODELSCOPE=true
export XINFERENCE_MODEL_SRC=modelscope

# 或使用 HuggingFace 镜像站点
export HF_ENDPOINT=https://hf-mirror.com
```

### 3.5 性能优化建议

1. **启用 FlashAttention**
   ```bash
   pip install flash-attn
   ```

2. **使用量化模型**
   - AWQ：平衡性能和精度
   - GPTQ：更高压缩率
   - SmoothQuant：保持更高精度

3. **调整并发参数**
   ```bash
   --max-num-seqs 256  # 根据显存大小调整
   ```

4. **使用 BF16 数据类型**
   ```bash
   --dtype bfloat16
   ```

---

## 附录

### A. 快速启动脚本

创建一个启动脚本 `start_xinference.sh`：

```bash
#!/bin/bash

# 激活环境
conda activate xinference

# 设置环境变量
export XINFERENCE_HOME=/data/xinference/models
export XINFERENCE_MODEL_SRC=modelscope

# 启动服务
nohup xinference-local --host 0.0.0.0 --port 8188 > ~/xinference.log 2>&1 &

echo "Xinference 服务已启动，日志文件：~/xinference.log"
echo "访问地址：http://localhost:8188"
```

赋予执行权限：
```bash
chmod +x start_xinference.sh
./start_xinference.sh
```

### B. vLLM 启动脚本

创建 `start_vllm.sh`：

```bash
#!/bin/bash

MODEL_NAME="Qwen3-235B-A22B-Instruct-2507"
MODEL_PATH="/home/username/projects/vllm/pretrained_models"

cd $MODEL_PATH

nohup VLLM_USE_MODELSCOPE=true vllm serve $MODEL_NAME \
  --tensor-parallel-size 8 \
  --max-model-len 131072 \
  --max-num-seqs 128 \
  --port 8188 \
  --gpu-memory-utilization 0.95 \
  --dtype bfloat16 \
  > ~/vllm.log 2>&1 &

echo "vLLM 服务已启动，日志文件：~/vllm.log"
echo "API 地址：http://localhost:8188/v1/chat/completions"
```

### C. 常用命令速查

```bash
# 查看 GPU 状态
nvidia-smi

# 查看 CUDA 版本
nvcc --version

# 激活环境
conda activate xinference

# 查看运行中的进程
ps aux | grep xinference
ps aux | grep vllm

# 查看端口占用
netstat -tulpn | grep 8188

# 查看日志
tail -f ~/xinference.log
tail -f ~/vllm.log

# 测试 API
curl http://localhost:8188/v1/models
```

### D. API 端点参考

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/models` | GET | 列出可用模型 |
| `/v1/chat/completions` | POST | 聊天补全 |
| `/v1/completions` | POST | 文本补全 |
| `/v1/embeddings` | POST | 文本嵌入 |
| `/health` | GET | 健康检查 |

### E. 推荐配置

**小型模型（7B-13B）：**
```bash
--tensor-parallel-size 1
--max-model-len 32768
--max-num-seqs 64
```

**中型模型（30B-70B）：**
```bash
--tensor-parallel-size 4
--max-model-len 65536
--max-num-seqs 128
```

**大型模型（100B+）：**
```bash
--tensor-parallel-size 8
--max-model-len 131072
--max-num-seqs 256
```

---

## 相关资源

- **Xinference GitHub：** https://github.com/xorbitsai/inference
- **vLLM GitHub：** https://github.com/vllm-project/vllm
- **ModelScope：** https://modelscope.cn
- **HuggingFace：** https://huggingface.co

---

## 更新日志

| 日期 | 版本 | 说明 |
|------|------|------|
| 2026-01-09 | v1.0 | 初始版本，包含 Xinference 和 vLLM 部署说明 |

---

**安全提示：**
1. 生产环境建议配置认证和访问控制
2. 注意防火墙和端口安全配置
3. 定期备份模型和配置文件
4. 监控服务器资源使用情况

---

*本手册仅供参考，具体操作请根据实际环境和需求调整。*
