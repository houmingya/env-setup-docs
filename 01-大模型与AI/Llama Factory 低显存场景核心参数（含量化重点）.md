# Llama Factory 低显存场景核心参数完整指南（含量化重点）

> **文档版本**: v1.0  
> **更新日期**: 2026年1月15日  
> **适用场景**: 消费级显卡（RTX 3060/4060）、入门级服务器、显存 ≤ 12GB 环境

---

## 📋 目录

- [一、核心量化参数（低显存核心依赖）](#一核心量化参数低显存核心依赖)
- [二、模型加载与优化参数](#二模型加载与优化参数)
- [三、训练与推理批次参数](#三训练与推理批次参数)
- [四、内存与显存管理参数](#四内存与显存管理参数)
- [五、高级优化参数](#五高级优化参数)
- [六、实战配置示例](#六实战配置示例)
- [七、常见问题与排查](#七常见问题与排查)
- [八、显存需求对照表](#八显存需求对照表)

---

## 一、核心量化参数（低显存核心依赖）

> 💡 **核心提示**: 此类参数直接降低模型显存占用，是低显存场景的核心配置，优先配置可使模型显存需求减少 **50%-75%**。

### 1.1 量化方法与位数

| 参数（短格式/长格式） | 功能说明（低显存适配重点） | 取值范围/格式 | 推荐值（低显存） |
|---|---|---|---|
| `--quantization_method`<br/>`--quantization-method` | 指定量化方法，核心降显存手段。不同方法适配不同场景，低显存优先选成熟轻量化方案。 | `bnb`、`gptq`、`awq`、`aqlm`、`quanto`、`eetq`、`hqq`、`mxfp4`、`fp8` | **4G-8G显存**: `gptq`/`awq`<br/>**2G-4G显存**: `aqlm`/`hqq` |
| `--quantization_bit`<br/>`--quantization-bit` | 量化位数，位数越低显存占用越少，但需平衡精度。低显存优先选4位，精度不足时再选8位。 | 整数（如 `4`、`8`） | **优先 4 位**<br/>8G 以上显存可尝试 8 位 |
| `--quantization_type`<br/>`--quantization-type` | 仅 bitsandbytes（bnb）量化时生效，控制量化数据类型，`nf4` 比 `fp4` 更适配通用场景。 | `fp4`、`nf4` | **nf4**（精度更优，显存占用一致） |
| `--double_quantization`<br/>`--no_double_quantization` | bnb 量化专属，开启双重量化可进一步降低显存占用（对精度影响极小）。 | 布尔值（`--no_xxx` 表示禁用） | `--double_quantization`（强制开启） |

### 1.2 量化设备与导出

| 参数（短格式/长格式） | 功能说明（低显存适配重点） | 取值范围/格式 | 推荐值（低显存） |
|---|---|---|---|
| `--quantization_device_map`<br/>`--quantization-device-map` | 指定量化模型的设备映射，`auto` 模式自动分配显存，避免手动配置出错。 | `auto`、`cuda:0`、自定义映射 | **auto**（低显存必选，自动优化分配） |
| `--export_quantization_bit`<br/>`--export-quantization-bit` | 导出模型时的量化位数，低显存场景导出时直接量化，避免后续重复处理占用显存。 | 整数（如 `4`、`8`） | 与训练/推理量化位数一致（**优先 4 位**） |
| `--export_quantization_dataset`<br/>`--export-quantization-dataset` | GPTQ/AWQ 导出时用于校准的数据集，影响量化后模型精度。 | 数据集名称或路径 | `alpaca_gpt4` 或自定义小型数据集 |
| `--export_quantization_maxlen` | GPTQ/AWQ 校准时的最大序列长度，越长越精确但耗时更多。 | 整数 | **512**（平衡精度与速度） |

---

## 二、模型加载与优化参数

### 2.1 内存与显存加载优化

| 参数（短格式/长格式） | 功能说明（低显存适配重点） | 取值范围/格式 | 推荐值（低显存） |
|---|---|---|---|
| `--low_cpu_mem_usage`<br/>`--no_low_cpu_mem_usage` | 内存高效加载模式，减少模型加载时的显存峰值占用（加载阶段最易爆显存）。 | 布尔值 | `--low_cpu_mem_usage`（**强制开启**） |
| `--offload_folder`<br/>`--offload-folder` | 模型权重卸载路径，将部分权重临时存到硬盘，缓解显存不足（牺牲少量速度）。 | 本地文件夹路径 | `./offload` 或 `./cache/offload` |
| `--device_map` | 指定模型在设备间的分配策略，`auto` 可自动跨 GPU/CPU 分配。 | `auto`、`balanced`、`sequential` | **auto**（自动平衡分配） |
| `--max_memory` | 限制每个设备的最大显存使用量，防止 OOM。 | 字典格式 `{0: "6GB", "cpu": "30GB"}` | 根据实际硬件设置 |

### 2.2 数据类型优化

| 参数（短格式/长格式） | 功能说明（低显存适配重点） | 取值范围/格式 | 推荐值（低显存） |
|---|---|---|---|
| `--infer_dtype`<br/>`--infer-dtype` | 推理时数据类型，`float16` 比 `float32` 显存占用少一半，`auto` 模式会自动适配最优轻量化类型。 | `auto`、`float16`、`bfloat16`、`float32` | **float16**（4G 以上显存）<br/>**auto**（2G-4G 显存） |
| `--compute_dtype`<br/>`--compute-dtype` | 训练时计算数据类型，影响训练精度与显存占用。 | `float16`、`bfloat16`、`float32` | **bfloat16**（精度更稳定，支持 Ampere 架构及以上） |
| `--model_dtype` | 模型权重加载时的数据类型。 | `auto`、`float16`、`bfloat16` | **auto** 或 **float16** |

---

## 三、训练与推理批次参数

### 3.1 批次大小控制

| 参数（短格式/长格式） | 功能说明（低显存适配重点） | 取值范围/格式 | 推荐值（低显存） |
|---|---|---|---|
| `--per_device_train_batch_size`<br/>`--per-device-train-batch-size` | 控制单设备训练批次大小，低显存需最小化批次，避免批量处理时爆显存。 | 整数 | **1-2**（4G-6G 显存）<br/>**2-4**（8G+ 显存） |
| `--per_device_eval_batch_size`<br/>`--per-device-eval-batch-size` | 控制单设备评估批次大小，推理时可适当增大。 | 整数 | **2-4**（推理无梯度，可比训练大） |
| `--gradient_accumulation_steps`<br/>`--gradient-accumulation-steps` | 梯度累积步数，批次太小时通过累积梯度保证训练效果，避免因批次小导致训练不稳定。 | 整数 | **4-8**（与小批次搭配使用）<br/>实际批次 = batch_size × 累积步数 |

### 3.2 序列长度优化

| 参数（短格式/长格式） | 功能说明（低显存适配重点） | 取值范围/格式 | 推荐值（低显存） |
|---|---|---|---|
| `--cutoff_len`<br/>`--cutoff-len` | 最大序列长度，越长显存占用越高（与批次大小成二次方关系）。 | 整数 | **512**（4G-6G 显存）<br/>**1024**（8G+ 显存） |
| `--max_samples` | 限制训练/评估使用的样本数量，减少数据加载显存。 | 整数 | **1000-5000**（快速实验） |

---

## 四、内存与显存管理参数

### 4.1 梯度检查点与优化器

| 参数（短格式/长格式） | 功能说明（低显存适配重点） | 取值范围/格式 | 推荐值（低显存） |
|---|---|---|---|
| `--gradient_checkpointing`<br/>`--no_gradient_checkpointing` | 梯度检查点技术，以计算换显存（重算中间激活值），可节省 **30-50%** 显存。 | 布尔值 | `--gradient_checkpointing`（**训练必开**） |
| `--optim` | 优化器选择，8bit 优化器可显著降低优化器状态显存占用。 | `adamw_torch`、`adamw_8bit`、`paged_adamw_8bit` | **paged_adamw_8bit**（显存最优）<br/>`adamw_8bit`（次优） |
| `--optim_target_modules` | 指定优化器作用的模块，配合 LoRA 使用时减少优化器状态。 | 模块名列表 | 默认（全部可训练参数） |

### 4.2 混合精度训练

| 参数（短格式/长格式） | 功能说明（低显存适配重点） | 取值范围/格式 | 推荐值（低显存） |
|---|---|---|---|
| `--fp16`<br/>`--no_fp16` | 启用 FP16 混合精度训练，降低显存占用并加速训练（需 GPU 支持）。 | 布尔值 | `--fp16`（老架构如 Pascal/Volta） |
| `--bf16`<br/>`--no_bf16` | 启用 BF16 混合精度训练，比 FP16 数值更稳定（需 Ampere 架构及以上）。 | 布尔值 | `--bf16`（**RTX 3060+ 优先**） |
| `--pure_bf16` | 纯 BF16 训练（不使用 FP32 主权重），进一步降低显存。 | 布尔值 | 实验性（可能影响收敛） |

---

## 五、高级优化参数

### 5.1 LoRA / QLoRA 参数微调

| 参数（短格式/长格式） | 功能说明（低显存适配重点） | 取值范围/格式 | 推荐值（低显存） |
|---|---|---|---|
| `--lora_rank`<br/>`--lora-rank` | LoRA 秩，越低可训练参数越少，显存占用越小。 | 整数 | **8-16**（平衡精度与显存）<br/>极低显存可用 4 |
| `--lora_alpha`<br/>`--lora-alpha` | LoRA 缩放系数，通常设为 rank 的 2 倍。 | 整数 | **16-32**（与 rank 配合） |
| `--lora_dropout` | LoRA 的 dropout 比例，防止过拟合。 | 浮点数 | **0.05-0.1** |
| `--lora_target` | LoRA 应用的目标模块，越少显存占用越低。 | 模块名列表 | `q_proj,v_proj`（最小配置）<br/>`all`（全覆盖） |
| `--use_qlora`<br/>`--no_use_qlora` | 启用 QLoRA（量化 + LoRA），4bit 量化基座 + LoRA 微调，低显存最优方案。 | 布尔值 | `--use_qlora`（**4G-8G 显存必选**） |

### 5.2 Flash Attention 与高效注意力

| 参数（短格式/长格式） | 功能说明（低显存适配重点） | 取值范围/格式 | 推荐值（低显存） |
|---|---|---|---|
| `--flash_attn`<br/>`--flash-attn` | 启用 Flash Attention 2，降低注意力机制显存占用（需安装 `flash-attn`）。 | `auto`、`fa2`、`sdpa`、`disabled` | **fa2**（A100/H100）<br/>**sdpa**（RTX 30/40 系列兼容） |
| `--shift_attn` | 启用位移短注意力（S²-Attn），降低长序列显存。 | 布尔值 | 实验性（长文本场景） |

### 5.3 其他优化

| 参数（短格式/长格式） | 功能说明（低显存适配重点） | 取值范围/格式 | 推荐值（低显存） |
|---|---|---|---|
| `--upcast_layernorm` | 将 LayerNorm 提升到 FP32 计算，提高数值稳定性（轻微增加显存）。 | 布尔值 | 默认（混合精度下自动处理） |
| `--disable_gradient_checkpointing` | 禁用梯度检查点（加速但增加显存）。 | 布尔值 | **不推荐**（低显存场景） |
| `--use_unsloth` | 使用 Unsloth 优化加速库（需单独安装）。 | 布尔值 | 可选（提升训练速度） |

---

## 六、实战配置示例

### 6.1 极低显存场景（4GB RTX 3060）

**目标**: 微调 7B 模型（如 Llama-2-7B）

```bash
llamafactory-cli train \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --dataset alpaca_gpt4 \
  --output_dir ./output/qlora_4g \
  --quantization_method bnb \
  --quantization_bit 4 \
  --quantization_type nf4 \
  --double_quantization \
  --use_qlora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_target q_proj,v_proj \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing \
  --bf16 \
  --optim paged_adamw_8bit \
  --cutoff_len 512 \
  --max_samples 2000 \
  --num_train_epochs 3 \
  --learning_rate 1e-4 \
  --low_cpu_mem_usage \
  --offload_folder ./offload \
  --flash_attn sdpa
```

**预期显存占用**: ~3.5GB

---

### 6.2 中等显存场景（8GB RTX 4060 Ti）

**目标**: 微调 7B 模型，适当提升精度

```bash
llamafactory-cli train \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --dataset alpaca_gpt4 \
  --output_dir ./output/qlora_8g \
  --quantization_method gptq \
  --quantization_bit 4 \
  --use_qlora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_target all \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --bf16 \
  --optim paged_adamw_8bit \
  --cutoff_len 1024 \
  --num_train_epochs 3 \
  --learning_rate 2e-4 \
  --low_cpu_mem_usage \
  --flash_attn fa2
```

**预期显存占用**: ~7.2GB

---

### 6.3 推理场景（12GB RTX 4070）

**目标**: 部署量化后的 13B 模型

```bash
llamafactory-cli chat \
  --model_name_or_path path/to/quantized-model \
  --quantization_method awq \
  --quantization_bit 4 \
  --infer_dtype float16 \
  --per_device_eval_batch_size 4 \
  --cutoff_len 2048 \
  --low_cpu_mem_usage \
  --flash_attn sdpa
```

**预期显存占用**: ~10GB

---

## 七、常见问题与排查

### 7.1 CUDA Out of Memory (OOM) 错误

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
1. **降低批次大小**: 将 `per_device_train_batch_size` 减至 1
2. **增加梯度累积**: 提高 `gradient_accumulation_steps` 至 8 或 16
3. **缩短序列长度**: 将 `cutoff_len` 从 2048 降至 512
4. **启用梯度检查点**: 确保 `--gradient_checkpointing` 开启
5. **使用更激进的量化**: 尝试 4bit + QLoRA
6. **启用权重卸载**: 设置 `--offload_folder`
7. **减少 LoRA 秩**: 将 `lora_rank` 从 16 降至 8 或 4

### 7.2 量化后精度下降严重

**症状**: 模型输出质量明显下降

**解决方案**:
1. **使用更高位数量化**: 从 4bit 提升至 8bit
2. **更换量化方法**: GPTQ/AWQ 通常比 HQQ 精度更高
3. **启用双重量化**: 确保 `--double_quantization` 开启
4. **使用 nf4**: 对于 bnb 量化，使用 `--quantization_type nf4`
5. **增加校准数据**: 提高 `--export_quantization_maxlen`

### 7.3 训练速度过慢

**症状**: 每个 step 耗时过长

**解决方案**:
1. **启用 Flash Attention**: `--flash_attn fa2` 或 `sdpa`
2. **禁用权重卸载**: 如果显存足够，移除 `--offload_folder`
3. **增大批次大小**: 在显存允许下提高 batch size
4. **使用混合精度**: 启用 `--bf16` 或 `--fp16`
5. **减少数据加载时间**: 使用 `--preprocessing_num_workers 4`

### 7.4 加载模型时卡住

**症状**: 模型加载阶段长时间无响应

**解决方案**:
1. **启用低内存模式**: 确保 `--low_cpu_mem_usage` 开启
2. **检查磁盘空间**: 确保缓存目录有足够空间
3. **使用本地模型**: 避免重复从 Hugging Face 下载
4. **清理缓存**: 删除 `~/.cache/huggingface/` 中的损坏文件

---

## 八、显存需求对照表

### 8.1 训练场景（QLoRA 微调）

| 模型规模 | 量化方式 | LoRA 配置 | 批次大小 | 序列长度 | 预计显存 |
|---|---|---|---|---|---|
| 7B | 4bit NF4 | rank=8 | 1 | 512 | **3.5-4GB** |
| 7B | 4bit NF4 | rank=16 | 2 | 1024 | **6-7GB** |
| 7B | 8bit | rank=16 | 2 | 1024 | **10-11GB** |
| 13B | 4bit NF4 | rank=8 | 1 | 512 | **6-7GB** |
| 13B | 4bit NF4 | rank=16 | 1 | 1024 | **9-10GB** |
| 30B | 4bit GPTQ | rank=8 | 1 | 512 | **12-14GB** |

### 8.2 推理场景（仅推理）

| 模型规模 | 量化方式 | 批次大小 | 序列长度 | 预计显存 |
|---|---|---|---|---|
| 7B | 4bit AWQ | 1 | 2048 | **3-4GB** |
| 7B | 8bit | 1 | 2048 | **7-8GB** |
| 13B | 4bit GPTQ | 1 | 2048 | **6-7GB** |
| 13B | 8bit | 1 | 2048 | **13-14GB** |
| 30B | 4bit AWQ | 1 | 2048 | **15-17GB** |
| 70B | 4bit GPTQ | 1 | 2048 | **35-40GB** |

---

## 九、参考资源

### 9.1 官方文档
- [LLaMA-Factory GitHub](https://github.com/hiyouga/LLaMA-Factory)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

### 9.2 量化工具
- [GPTQ](https://github.com/IST-DASLab/gptq)
- [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)

### 9.3 社区资源
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)
- [Efficient Fine-Tuning 最佳实践](https://huggingface.co/docs/transformers/perf_train_gpu_one)

---

## 十、更新日志

| 版本 | 日期 | 更新内容 |
|---|---|---|
| v1.0 | 2026-01-15 | 初始版本，包含核心量化参数与配置示例 |

---

> **注意**: 本文档部分内容基于 LLaMA-Factory 主流版本（v0.7.x），不同版本参数可能略有差异，请以官方文档为准。建议在使用前通过 `llamafactory-cli train --help` 查看完整参数列表。

> **免责声明**: 文档中的显存占用数据为估算值，实际占用受模型架构、数据集、硬件环境等多种因素影响，仅供参考。
