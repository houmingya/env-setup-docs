# LLaMA-Factory 安装使用手册

本手册提供 LLaMA-Factory 的完整安装、配置和使用指南。

---

## 目录

- [一、服务器连接](#一服务器连接)
- [二、安装步骤](#二安装步骤)
- [三、使用说明](#三使用说明)
- [四、常见问题与解决方案](#四常见问题与解决方案)
- [五、数据处理](#五数据处理)

---

## 一、服务器连接

### 1.1 SSH 连接

使用 SSH 公钥方式连接到服务器：

```bash
ssh -p <端口号> <用户名>@<服务器IP地址>
```

**示例：**
```bash
ssh -p 22022 username@192.168.1.100
```

### 1.2 查看 GPU 状态

连接成功后，使用以下命令查看显卡信息：

```bash
nvidia-smi
```

该命令可以显示：
- GPU 使用率
- 显存占用情况
- 运行中的进程
- GPU 温度等信息

---

## 二、安装步骤

### 2.1 创建虚拟环境

使用 Mamba（或 Conda）创建 Python 虚拟环境：

```bash
mamba create -n llama_factory python=3.10 numpy=1.26.4
```

> **提示：** 如果没有安装 Mamba，可以使用 Conda 替代：
> ```bash
> conda create -n llama_factory python=3.10 numpy=1.26.4
> ```

### 2.2 激活虚拟环境

```bash
conda activate llama_factory
```

### 2.3 从源码安装 LLaMA-Factory

克隆项目仓库并安装依赖：

```bash
# 克隆仓库（仅克隆最新版本，减少下载时间）
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git

# 进入项目目录
cd LLaMA-Factory

# 安装依赖包
pip install -e ".[torch,metrics]" --no-build-isolation
```

**依赖说明：**
- `torch`: PyTorch 深度学习框架
- `metrics`: 评估指标相关依赖

### 2.4 验证安装

安装完成后，验证是否安装成功：

```bash
llamafactory-cli version
```

如果显示版本号，则说明安装成功。

---

## 三、使用说明

### 3.1 启动 Web UI 界面

LLaMA-Factory 提供了图形化界面，方便进行模型训练和推理。

#### 基本启动命令

```bash
llamafactory-cli webui
```

#### 指定 GPU 和生成共享链接

如果需要远程访问或指定特定 GPU，使用以下命令：

```bash
CUDA_VISIBLE_DEVICES=<GPU编号> GRADIO_SHARE=1 GRADIO_SERVER_PORT=7860 llamafactory-cli webui
```

**参数说明：**
- `CUDA_VISIBLE_DEVICES=<GPU编号>`: 指定使用的 GPU（0 表示第一张显卡，1 表示第二张显卡）
- `GRADIO_SHARE=1`: 生成公共访问链接（用于非本地机连接）
- `GRADIO_SERVER_PORT=7860`: 指定 Web 服务端口号

**示例：**
```bash
# 使用第二张显卡，生成共享链接
CUDA_VISIBLE_DEVICES=1 GRADIO_SHARE=1 GRADIO_SERVER_PORT=7860 llamafactory-cli webui
```

> **注意：** 请根据实际情况选择空闲的 GPU，避免资源冲突。

---

## 四、常见问题与解决方案

### 4.1 问题：缺少 frpc_linux_amd64_v0.3 文件

#### 错误信息

```
Could not create share link. Missing file: /home/<username>/.cache/huggingface/gradio/frpc/frpc_linux_amd64_v0.3

Please check your internet connection. This can happen if your antivirus software blocks the download of this file.
```

#### 解决方案一：服务器无网络环境

当服务器无法直接访问互联网时，需要手动下载并传输文件。

**步骤 1：在有网络的设备上下载文件**

访问以下链接下载文件：
```
https://cdn-media.huggingface.co/frpc-gradio-0.3/frpc_linux_amd64
```

将下载的文件重命名为：`frpc_linux_amd64_v0.3`

**步骤 2：使用 SCP 传输文件到服务器**

在本地计算机的终端执行：

```bash
scp <本地文件路径> <用户名>@<服务器IP>:~/frpc_linux_amd64_v0.3
```

**示例：**
```bash
scp D:\downloads\frpc_linux_amd64_v0.3 username@192.168.1.100:~/frpc_linux_amd64_v0.3
```

**步骤 3：在服务器上移动文件并设置权限**

SSH 登录服务器后，执行以下命令：

```bash
# 创建目标目录（如果不存在）
mkdir -p ~/.cache/huggingface/gradio/frpc

# 移动文件到正确位置
mv ~/frpc_linux_amd64_v0.3 ~/.cache/huggingface/gradio/frpc/

# 添加执行权限
chmod +x ~/.cache/huggingface/gradio/frpc/frpc_linux_amd64_v0.3
```

#### 解决方案二：服务器有网络环境

如果服务器可以访问互联网，直接执行以下命令：

```bash
# 创建目标目录
mkdir -p ~/.cache/huggingface/gradio/frpc

# 使用 wget 下载文件
wget https://cdn-media.huggingface.co/frpc-gradio-0.3/frpc_linux_amd64 \
  -O ~/.cache/huggingface/gradio/frpc/frpc_linux_amd64_v0.3

# 或使用 curl（如果 wget 不可用）
# curl -L https://cdn-media.huggingface.co/frpc-gradio-0.3/frpc_linux_amd64 \
#   -o ~/.cache/huggingface/gradio/frpc/frpc_linux_amd64_v0.3

# 添加执行权限
chmod +x ~/.cache/huggingface/gradio/frpc/frpc_linux_amd64_v0.3
```

#### 验证文件是否正确

完成上述步骤后，验证文件：

```bash
# 检查文件是否存在
ls -la ~/.cache/huggingface/gradio/frpc/

# 检查文件权限
ls -l ~/.cache/huggingface/gradio/frpc/frpc_linux_amd64_v0.3

# 测试运行（应显示版本信息）
~/.cache/huggingface/gradio/frpc/frpc_linux_amd64_v0.3 --version
```

---

## 五、数据处理

### 5.1 模型下载

LLaMA-Factory 支持从 ModelScope 下载预训练模型。

#### 安装 ModelScope

```bash
pip install modelscope
```

#### 下载模型

下载完整的模型库（以 Qwen2.5-7B-Instruct 为例）：

```bash
modelscope download --model Qwen/Qwen2.5-7B-Instruct
```

### 5.2 数据集准备

#### 5.2.1 数据集格式说明

关于数据集文件的格式，请参考项目中的 `data/README_zh.md` 文件。

LLaMA-Factory 支持以下数据源：
- HuggingFace 数据集
- ModelScope 数据集
- Modelers 数据集
- 本地自定义数据集

#### 5.2.2 指令监督微调数据格式

使用 JSON 格式存储训练数据，每条数据包含以下字段：

```json
[
  {
    "instruction": "用户指令（必填）",
    "input": "用户输入（选填）",
    "output": "模型回答（必填）",
    "system": "系统提示词（选填）",
    "history": [
      ["第一轮指令（选填）", "第一轮回答（选填）"],
      ["第二轮指令（选填）", "第二轮回答（选填）"]
    ]
  }
]
```

**字段说明：**
- `instruction`：用户的指令或问题（必填）
- `input`：用户的额外输入信息（选填）
- `output`：模型的期望回答（必填）
- `system`：系统级提示词，定义模型角色（选填）
- `history`：多轮对话历史记录（选填）

#### 5.2.3 配置数据集信息

编辑 `data/dataset_info.json` 文件，添加自定义数据集的配置信息。

**使用 vim 编辑器：**

```bash
vim data/dataset_info.json
```

- 按 `i` 进入编辑模式
- 修改内容
- 按 `Esc` 退出编辑模式
- 输入 `:wq` 保存并退出

**添加数据集配置示例：**

```json
{
  "数据集名称": {
    "file_name": "data.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "system": "system",
      "history": "history"
    }
  }
}
```

**字段映射说明：**
- `file_name`: 数据文件名称
- `columns`: 字段映射关系，将数据文件中的字段映射到标准字段

### 5.3 上传数据集到服务器

#### 方法一：使用 SCP 命令

**步骤 1：从本地上传文件到服务器家目录**

```bash
scp <本地文件路径> <用户名>@<服务器IP>:~/
```

**示例：**
```bash
scp D:\data\train.json username@192.168.1.100:~/
```

**步骤 2：SSH 登录服务器**

```bash
ssh <用户名>@<服务器IP>
```

**步骤 3：移动文件到项目数据目录**

```bash
# 移动文件到 LLaMA-Factory 数据目录
mv ~/train.json ~/LLaMA-Factory/data/

# 验证文件是否在正确位置
ls ~/LLaMA-Factory/data/
```

#### 方法二：使用 SFTP 工具

也可以使用图形化 SFTP 工具（如 FileZilla、WinSCP）进行文件传输。

---

## 附录

### A. 常用命令速查

```bash
# 查看 GPU 状态
nvidia-smi

# 激活虚拟环境
conda activate llama_factory

# 启动 WebUI
llamafactory-cli webui

# 查看已安装版本
llamafactory-cli version

# 退出虚拟环境
conda deactivate
```

### B. 目录结构说明

```
LLaMA-Factory/
├── data/                    # 数据集目录
│   ├── dataset_info.json   # 数据集配置文件
│   └── README_zh.md        # 数据集格式说明
├── examples/               # 示例脚本
├── src/                    # 源代码
└── README.md              # 项目说明文档
```

### C. 相关资源

- **项目地址：** https://github.com/hiyouga/LLaMA-Factory
- **官方文档：** 参见项目 README
- **问题反馈：** 项目 Issues 页面

---

## 更新日志

| 日期 | 版本 | 说明 |
|------|------|------|
| 2026-01-09 | v1.0 | 初始版本，包含基础安装和使用说明 |

---

**注意事项：**
1. 请确保服务器有足够的存储空间和 GPU 显存
2. 训练大模型时请注意监控 GPU 温度和功耗
3. 定期备份重要的模型检查点和训练数据
4. 使用共享链接时注意网络安全

---

*本手册仅供参考，具体操作请根据实际环境调整。*
