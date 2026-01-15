# Llama Factory 命令行参数中文说明文档

本文档对 Llama Factory 核心命令行参数进行分类整理，清晰说明各参数的功能、取值范围及默认值，方便用户在模型训练、推理、量化等场景下快速查阅和使用。

# 一、核心基础参数

此类参数用于模型与分词器的基础配置，是多数操作的必备选项。

|参数（短格式/长格式）|功能说明|取值范围/格式|默认值|
|---|---|---|---|
|--model_name_or_path--model-name-or-path|指定模型权重路径，或 Hugging Face/ModelScope 上的模型标识符|本地路径、模型仓库标识符（如 hiyouga/mini-llama-2-1.1b）|None|
|--adapter_name_or_path--adapter-name-or-path|指定适配器权重路径或 Hugging Face 标识符，多适配器用逗号分隔|本地路径、模型仓库标识符，多值用逗号分隔|None|
|--adapter_folder--adapter-folder|指定包含多个待加载适配器权重的文件夹路径|本地文件夹路径|None|
|--cache_dir--cache-dir|指定从 Hugging Face/ModelScope 下载预训练模型的缓存目录|本地文件夹路径|None|
|--use_fast_tokenizer/--no_use_fast_tokenizer--use-fast-tokenizer/--no-use-fast-tokenizer|控制是否使用 tokenizers 库提供的快速分词器|布尔值（--no_xxx 表示禁用）|--use_fast_tokenizer=True|
|--resize_vocab--resize-vocab|控制是否调整分词器词汇表和嵌入层的大小|布尔值|False|
|--add_tokens/--add-special-tokens|分别用于添加非特殊令牌和特殊令牌到分词器，多令牌用逗号分隔|令牌字符串，多值用逗号分隔|None|
|--new_special_tokens_config--new-special-tokens-config|指定包含特殊令牌描述的 YAML 配置文件路径，优先级高于 add_special_tokens|YAML 文件路径，格式：{'<token>': '描述文本'}|None|
|--init_special_tokens--init-special-tokens|新特殊令牌的初始化方法，desc_init 类方法需配合 new_special_tokens_config 使用|noise_init（默认）、desc_init、desc_init_w_noise|noise_init|
|--model_revision--model-revision|指定要使用的模型版本（分支名、标签名或提交 ID）|分支名、标签名、提交 ID 字符串|main|
# 二、模型优化与加速参数

此类参数用于优化模型加载、训练及推理性能，适配不同硬件环境。

|参数（短格式/长格式）|功能说明|取值范围/格式|默认值|
|---|---|---|---|
|--low_cpu_mem_usage/--no_low_cpu_mem_usage--low-cpu-mem-usage/--no-low-cpu-mem-usage|控制是否使用内存高效的模型加载方式|布尔值（--no_xxx 表示禁用）|--low_cpu_mem_usage=True|
|--rope_scaling--rope-scaling|指定 RoPE 嵌入采用的缩放策略，用于适配长文本场景|linear、dynamic、yarn、llama3|None|
|--flash_attn--flash-attn|启用 FlashAttention 以加快训练和推理速度|auto、disabled、sdpa、fa2、fa3|AttentionFunction.AUTO|
|--shift_attn--shift-attn|启用 LongLoRA 提出的移位短注意力（S²-Attn），提升长文本处理能力|布尔值|False|
|--mixture_of_depths--mixture-of-depths|将模型转换为深度混合（MoD）模型或加载已有的 MoD 模型|convert、load|None|
|--use_unsloth/--use_unsloth_gc--use-unsloth/--use-unsloth-gc|分别控制在 LoRA 训练中是否使用 unsloth 优化、是否使用 unsloth 梯度检查点（无需安装 unsloth）|布尔值|均为 False|
|--enable_liger_kernel--enable-liger-kernel|启用 liger 内核以加快训练速度|布尔值|False|
|--upcast_layernorm/--upcast_lmhead_output--upcast-layernorm/--upcast-lmhead-output|分别控制是否将层归一化权重、lm_head 输出提升到 fp32 精度，提升计算稳定性|布尔值|均为 False|
# 三、推理相关参数

此类参数专门用于模型推理场景，控制推理引擎、精度、缓存等关键配置。

|参数（短格式/长格式）|功能说明|取值范围/格式|默认值|
|---|---|---|---|
|--infer_backend--infer-backend|指定推理时使用的后端引擎|huggingface、vllm、sglang、ktransformers|EngineName.HF|
|--offload_folder--offload-folder|指定模型权重卸载路径，用于缓解显存不足问题|本地文件夹路径|offload|
|--use_kv_cache/--no_use_kv_cache--use-kv-cache/--no-use-kv-cache|控制在生成过程中是否使用 KV 缓存，启用可提升推理速度|布尔值（--no_xxx 表示禁用）|--use_kv_cache=True|
|--infer_dtype--infer-dtype|指定推理时模型权重和激活值的数据类型|auto、float16、bfloat16、float32|auto|
|--hf_hub_token/--ms_hub_token/--om_hub_token--hf-hub-token/--ms-hub-token/--om-hub-token|分别用于登录 Hugging Face Hub、ModelScope Hub、Modelers Hub 的授权令牌|字符串令牌|均为 None|
|--trust_remote_code--trust-remote-code|控制是否信任来自 Hub 上定义的数据集/模型的代码执行|布尔值|False|
# 四、量化相关参数

此类参数用于模型量化，降低显存占用，适配低显存硬件环境。

|参数（短格式/长格式）|功能说明|取值范围/格式|默认值|
|---|---|---|---|
|--quantization_method--quantization-method|指定用于动态量化的量化方法|bnb、gptq、awq、aqlm、quanto、eetq、hqq、mxfp4、fp8|QuantizationMethod.BNB|
|--quantization_bit--quantization-bit|指定动态量化时模型的量化位数|整数（如 4、8）|None|
|--quantization_type--quantization-type|指定 bitsandbytes int4 训练中使用的量化数据类型|fp4、nf4|nf4|
|--double_quantization/--no_double_quantization--double-quantization/--no-double-quantization|控制在 bitsandbytes int4 训练中是否使用双重量化|布尔值（--no_xxx 表示禁用）|--double_quantization=True|
|--quantization_device_map--quantization-device-map|指定用于推理 4 位量化模型的设备映射，需 bitsandbytes>=0.43.0|auto|None|
# 五、多媒体相关参数

此类参数用于处理图像、视频、音频等多媒体输入，适配多模态模型。

|参数（短格式/长格式）|功能说明|取值范围/格式|默认值|
|---|---|---|---|
|--image_max_pixels/--image_min_pixels--image-max-pixels/--image-min-pixels|分别指定图像输入的最大、最小像素数|整数|最大：589824；最小：1024|
|--image_do_pan_and_scan/--crop_to_patches--image-do-pan-and-scan/--crop-to-patches|分别控制对 gemma3 模型是否使用平移扫描处理图像、对 internvl 模型是否将图像裁剪为补丁|布尔值|均为 False|
|--video_max_pixels/--video_min_pixels--video-max-pixels/--video-min-pixels|分别指定视频输入的最大、最小像素数|整数|最大：65536；最小：256|
|--video_fps/--video_maxlen--video-fps/--video-maxlen|分别指定视频输入的每秒采样帧数、最大采样帧数|fps：浮点数；maxlen：整数|fps：2.0；maxlen：128|
|--use_audio_in_video--use-audio-in-video|控制在视频输入中是否使用音频|布尔值|False|
|--audio_sampling_rate--audio-sampling-rate|指定音频输入的采样率|整数|16000|
# 六、模型导出参数

此类参数用于将训练后的模型导出为指定格式，便于部署和分享。

|参数（短格式/长格式）|功能说明|取值范围/格式|默认值|
|---|---|---|---|
|--export_dir--export-dir|指定保存导出模型的目录路径|本地文件夹路径|None|
|--export_size--export-size|指定导出模型的文件分片大小（以 GB 为单位）|浮点数|5|
|--export_device--export-device|指定模型导出时使用的设备，auto 可加速导出|cpu、auto|cpu|
|--export_quantization_bit--export-quantization-bit|指定导出模型的量化位数|整数（如 4、8）|None|
|--export_legacy_format--export-legacy-format|控制是否保存为 .bin 文件而非 .safetensors 文件|布尔值|False|
|--export_hub_model_id--export-hub-model-id|指定将模型推送到 Hugging Face Hub 时的仓库名称|字符串|None|
# 七、数据集相关参数

此类参数用于训练/评估数据集的配置，包括数据路径、预处理、混合策略等。

|参数（短格式/长格式）|功能说明|取值范围/格式|默认值|
|---|---|---|---|
|--template|指定用于在训练和推理中构造提示词的模板|模板名称字符串|None|
|--dataset/--eval_dataset--eval-dataset|分别指定用于训练、评估的数据集名称，多数据集用逗号分隔|数据集名称字符串，多值用逗号分隔|均为 None|
|--dataset_dir/--media_dir--dataset-dir/--media-dir|分别指定包含数据集、多媒体文件（图像/视频/音频）的文件夹路径，media_dir 默认为 dataset_dir|本地文件夹路径|dataset_dir：data；media_dir：None|
|--cutoff_len--cutoff-len|指定数据集中分词后输入的截断长度|整数|2048|
|--train_on_prompt/--mask_history--train-on-prompt/--mask-history|分别控制是否禁用对提示词的掩码、是否掩码历史记录并仅在最后一轮进行训练|布尔值|均为 False|
|--mix_strategy--mix-strategy|指定数据集混合使用的策略|concat（拼接）、interleave_under（交错下采样）、interleave_over（交错上采样）|concat|
|--val_size--val-size|指定验证集的大小，可为整数或 [0,1) 范围内的浮点数|整数/浮点数|0.0|
|--packing/--neat_packing--neat-packing|分别控制在训练中是否启用序列打包、是否启用无交叉注意力的序列打包（预训练时自动启用打包）|packing：None/布尔值；neat_packing：布尔值|packing：None；neat_packing：False|
# 八、训练核心参数

此类参数是模型训练的核心配置，包括训练模式、优化器、学习率等关键设置。

|参数（短格式/长格式）|功能说明|取值范围/格式|默认值|
|---|---|---|---|
|--output_dir--output-dir|指定用于写入模型预测结果和检查点的输出目录|本地文件夹路径|trainer_output|
|--do_train/--do_eval/--do_predict--do-train/--do-eval/--do-predict|分别控制是否运行训练、是否在开发集上评估、是否在测试集上预测|布尔值|均为 False|
|--eval_strategy--eval-strategy|指定使用的评估策略|no（不评估）、steps（按步骤）、epoch（按轮次）|no|
|--per_device_train_batch_size/--per_device_eval_batch_size--per-device-train-batch-size/--per-device-eval-batch-size|分别指定训练、评估时每个设备加速器核心/CPU 的批次大小|整数|均为 8|
|--gradient_accumulation_steps--gradient-accumulation-steps|指定在执行反向传播/更新步骤之前累积的更新步骤数|整数|1|
|--learning_rate--learning-rate|指定 AdamW 优化器的初始学习率|浮点数|5e-05|
|--num_train_epochs/--max_steps--num-train-epochs/--max-steps|分别指定训练的总轮次、总步骤数（max_steps>0 时覆盖 num_train_epochs）|num_train_epochs：浮点数；max_steps：整数|num_train_epochs：3.0；max_steps：-1|
|--lr_scheduler_type--lr-scheduler-type|指定使用的学习率调度器类型|linear、cosine、cosine_with_restarts 等多种类型|linear|
|--warmup_ratio/--warmup_steps--warmup-ratio/--warmup-steps|分别指定线性预热占总步骤的比例、线性预热的步骤数|warmup_ratio：浮点数；warmup_steps：整数|均为 0.0/0|
# 九、日志与保存参数

此类参数控制训练过程中的日志记录和模型检查点保存策略。

|参数（短格式/长格式）|功能说明|取值范围/格式|默认值|
|---|---|---|---|
|--log_level/--log_level_replica--log-level/--log-level-replica|分别指定主节点、副本节点上使用的日志级别|detail、debug、info、warning、error、critical、passive|主节点：passive；副本节点：warning|
|--logging_dir--logging-dir|指定 Tensorboard 日志目录|本地文件夹路径|None|
|--logging_strategy/--logging_steps--logging-strategy/--logging-steps|分别指定日志记录策略、每多少个更新步骤记录一次日志（steps 可为整数或比例）|strategy：no、steps、epoch；steps：整数/浮点数|strategy：steps；steps：500|
|--save_strategy/--save_steps--save-strategy/--save-steps|分别指定检查点保存策略、每多少个更新步骤保存一次检查点（steps 可为整数或比例）|strategy：no、steps、epoch、best；steps：整数/浮点数|strategy：steps；steps：500|
|--save_total_limit--save-total-limit|限制检查点的总数，删除较旧的检查点|整数|None（无限制）|
|--save_safetensors/--no_save_safetensors--save-safetensors/--no-save-safetensors|控制是否使用 safetensors 保存和加载状态字典|布尔值（--no_xxx 表示禁用）|--save_safetensors=True|
# 十、设备与精度参数

此类参数用于配置训练/推理的硬件设备和计算精度，适配不同硬件环境。

|参数（短格式/长格式）|功能说明|取值范围/格式|默认值|
|---|---|---|---|
|--use_cpu--use-cpu|控制是否使用 CPU 进行计算（禁用则自动使用可用的 CUDA/MPS 等设备）|布尔值|False|
|--seed/--data_seed--data-seed|分别指定训练开始时的随机种子、用于数据采样器的随机种子|整数|seed：42；data_seed：None|
|--bf16/--fp16--bf16/--fp16|分别控制是否使用 bf16、fp16 混合精度计算（bf16 需 Ampere 及以上 GPU）|布尔值|均为 False|
|--fp16_opt_level--fp16-opt-level|指定 fp16 精度下的 Apex AMP 优化级别|O0、O1、O2、O3|O1|
|--tf32|控制是否启用 tf32 模式，适用于 Ampere 及更新版本的 GPU 架构|布尔值/None|None|
# 十一、分布式训练参数

此类参数用于多节点、多设备分布式训练场景的配置。

|参数（短格式/长格式）|功能说明|取值范围/格式|默认值|
|---|---|---|---|
|--local_rank--local-rank|指定分布式训练时的本地秩|整数|-1|
|--ddp_backend--ddp-backend|指定用于分布式训练的后端|nccl、gloo、mpi、ccl、hccl、cncl、mccl|None|
|--fsdp|控制在分布式训练中是否使用 PyTorch 完全分片数据并行（FSDP）训练|full_shard、shard_grad_op、no_shard 等（可搭配 offload/auto_wrap）|None|
|--deepspeed|启用 deepspeed 并指定配置文件路径|DeepSpeed 配置文件路径或字典|None|
# 十二、补充说明

- 参数格式：短横线连接的参数名（如 --model-name-or-path）与下划线连接的参数名（如 --model_name_or_path）功能完全一致，可任选其一使用。

- 布尔值参数：部分参数通过前缀 `--no_` 实现反向控制（如 --use_fast_tokenizer 对应 --no_use_fast_tokenizer），未指定时使用默认值。

- 版本适配：部分参数（如 --flash_attn、--use_unsloth）需依赖特定版本的依赖库，使用前请确保相关库已正确安装。

- 优先级：当多个参数对同一功能进行配置时（如 --new_special_tokens_config 与 --add_special_tokens），以文档中标注的优先级为准。
> （注：文档部分内容可能由 AI 生成）