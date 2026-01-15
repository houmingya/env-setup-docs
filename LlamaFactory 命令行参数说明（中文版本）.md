# LlamaFactory 命令行参数说明（中文版本）

### 核心参数

--model_name_or_path, --model-name-or-path
模型权重路径或来自 huggingface.co/models 或 modelscope.cn/models 的模型标识符。（默认值：None）

--adapter_name_or_path, --adapter-name-or-path
适配器权重路径或来自 huggingface.co/models 的标识符。多个适配器用逗号分隔。（默认值：None）

--adapter_folder, --adapter-folder
包含待加载适配器权重的文件夹路径。（默认值：None）

--cache_dir, --cache-dir
用于存储从 huggingface.co 或 modelscope.cn 下载的预训练模型的目录。（默认值：None）

--use_fast_tokenizer [USE_FAST_TOKENIZER], --use-fast-tokenizer [USE_FAST_TOKENIZER]
是否使用快速分词器（由 tokenizers 库提供支持）。（默认值：True）

--no_use_fast_tokenizer, --no-use-fast-tokenizer
不使用快速分词器（由 tokenizers 库提供支持）。（默认值：False）

--resize_vocab [RESIZE_VOCAB], --resize-vocab [RESIZE_VOCAB]
是否调整分词器词汇表和嵌入层的大小。（默认值：False）

--split_special_tokens [SPLIT_SPECIAL_TOKENS], --split-special-tokens [SPLIT_SPECIAL_TOKENS]
在分词过程中是否拆分特殊令牌。（默认值：False）

--add_tokens ADD_TOKENS, --add-tokens ADD_TOKENS
要添加到分词器中的非特殊令牌。多个令牌用逗号分隔。（默认值：None）

--add_special_tokens ADD_SPECIAL_TOKENS, --add-special-tokens ADD_SPECIAL_TOKENS
要添加到分词器中的特殊令牌。多个令牌用逗号分隔。（默认值：None）

--new_special_tokens_config NEW_SPECIAL_TOKENS_CONFIG, --new-special-tokens-config NEW_SPECIAL_TOKENS_CONFIG
包含特殊令牌描述的 YAML 配置文件路径，用于语义初始化。若设置此参数，其优先级高于 add_special_tokens。YAML 格式：{'<token>': '描述文本', ...}（默认值：None）

--init_special_tokens {noise_init,desc_init,desc_init_w_noise}, --init-special-tokens {noise_init,desc_init,desc_init_w_noise}
新特殊令牌的初始化方法：'noise_init'（默认，围绕均值的随机噪声）、'desc_init'（基于描述的语义初始化）、'desc_init_w_noise'（语义初始化+随机噪声）。注意：'desc_init' 类方法需要 new_special_tokens_config。（默认值：noise_init）

--model_revision MODEL_REVISION, --model-revision
要使用的特定模型版本（可以是分支名、标签名或提交 ID）。（默认值：main）

--low_cpu_mem_usage [LOW_CPU_MEM_USAGE], --low-cpu-mem-usage [LOW_CPU_MEM_USAGE]
是否使用内存高效的模型加载方式。（默认值：True）

--no_low_cpu_mem_usage, --no-low-cpu-mem-usage
不使用内存高效的模型加载方式。（默认值：False）

--rope_scaling {linear,dynamic,yarn,llama3}, --rope-scaling {linear,dynamic,yarn,llama3}
RoPE 嵌入所采用的缩放策略。（默认值：None）

--flash_attn {auto,disabled,sdpa,fa2,fa3}, --flash-attn {auto,disabled,sdpa,fa2,fa3}
启用 FlashAttention 以加快训练和推理速度。（默认值：AttentionFunction.AUTO）

--shift_attn [SHIFT_ATTN], --shift-attn [SHIFT_ATTN]
启用 LongLoRA 提出的移位短注意力（S²-Attn）。（默认值：False）

--mixture_of_depths {convert,load}, --mixture-of-depths {convert,load}
将模型转换为深度混合（MoD）模型或加载已有的 MoD 模型。（默认值：None）

--use_unsloth [USE_UNSLOTH], --use-unsloth [USE_UNSLOTH]
在 LoRA 训练中是否使用 unsloth 的优化。（默认值：False）

--use_unsloth_gc [USE_UNSLOTH_GC], --use-unsloth-gc [USE_UNSLOTH_GC]
是否使用 unsloth 的梯度检查点（无需安装 unsloth）。（默认值：False）

--enable_liger_kernel [ENABLE_LIGER_KERNEL], --enable-liger-kernel [ENABLE_LIGER_KERNEL]
是否启用 liger 内核以加快训练速度。（默认值：False）

--moe_aux_loss_coef MOE_AUX_LOSS_COEF, --moe-aux-loss-coef MOE_AUX_LOSS_COEF
混合专家模型中辅助路由器损失的系数。（默认值：None）

--disable_gradient_checkpointing [DISABLE_GRADIENT_CHECKPOINTING], --disable-gradient-checkpointing [DISABLE_GRADIENT_CHECKPOINTING]
是否禁用梯度检查点。（默认值：False）

--use_reentrant_gc [USE_REENTRANT_GC], --use-reentrant-gc [USE_REENTRANT_GC]
是否使用可重入梯度检查点。（默认值：True）

--no_use_reentrant_gc, --no-use-reentrant-gc
不使用可重入梯度检查点。（默认值：False）

--upcast_layernorm [UPCAST_LAYERNORM], --upcast-layernorm [UPCAST_LAYERNORM]
是否将层归一化权重提升到 fp32 精度。（默认值：False）

--upcast_lmhead_output [UPCAST_LMHEAD_OUTPUT], --upcast-lmhead-output [UPCAST_LMHEAD_OUTPUT]
是否将 lm_head 的输出提升到 fp32 精度。（默认值：False）

--train_from_scratch [TRAIN_FROM_SCRATCH], --train-from-scratch [TRAIN_FROM_SCRATCH]
是否随机初始化模型权重。（默认值：False）

### 推理相关参数

--infer_backend {huggingface,vllm,sglang,ktransformers}, --infer-backend {huggingface,vllm,sglang,ktransformers}
推理时使用的后端引擎。（默认值：EngineName.HF）

--offload_folder OFFLOAD_FOLDER, --offload-folder OFFLOAD_FOLDER
模型权重卸载路径。（默认值：offload）

--use_kv_cache [USE_KV_CACHE], --use-kv-cache [USE_KV_CACHE]
在生成过程中是否使用 KV 缓存。（默认值：True）

--no_use_kv_cache, --no-use-kv-cache
在生成过程中不使用 KV 缓存。（默认值：False）

--use_v1_kernels [USE_V1_KERNELS], --use-v1-kernels [USE_V1_KERNELS]
在训练中是否使用高性能内核。（默认值：False）

--infer_dtype {auto,float16,bfloat16,float32}, --infer-dtype {auto,float16,bfloat16,float32}
推理时模型权重和激活值的数据类型。（默认值：auto）

--hf_hub_token HF_HUB_TOKEN, --hf-hub-token HF_HUB_TOKEN
用于登录 Hugging Face Hub 的授权令牌。（默认值：None）

--ms_hub_token MS_HUB_TOKEN, --ms-hub-token MS_HUB_TOKEN
用于登录 ModelScope Hub 的授权令牌。（默认值：None）

--om_hub_token OM_HUB_TOKEN, --om-hub-token OM_HUB_TOKEN
用于登录 Modelers Hub 的授权令牌。（默认值：None）

--print_param_status [PRINT_PARAM_STATUS], --print-param-status [PRINT_PARAM_STATUS]
用于调试，打印模型中参数的状态。（默认值：False）

--trust_remote_code [TRUST_REMOTE_CODE], --trust-remote-code [TRUST_REMOTE_CODE]
是否信任来自 Hub 上定义的数据集/模型的代码执行。（默认值：False）

### 量化相关参数

--quantization_method {bnb,gptq,awq,aqlm,quanto,eetq,hqq,mxfp4,fp8}, --quantization-method {bnb,gptq,awq,aqlm,quanto,eetq,hqq,mxfp4,fp8}
用于动态量化的量化方法。（默认值：QuantizationMethod.BNB）

--quantization_bit QUANTIZATION_BIT, --quantization-bit QUANTIZATION_BIT
使用动态量化时模型的量化位数。（默认值：None）

--quantization_type {fp4,nf4}, --quantization-type {fp4,nf4}
bitsandbytes int4 训练中使用的量化数据类型。（默认值：nf4）

--double_quantization [DOUBLE_QUANTIZATION], --double-quantization [DOUBLE_QUANTIZATION]
在 bitsandbytes int4 训练中是否使用双重量化。（默认值：True）

--no_double_quantization, --no-double-quantization
在 bitsandbytes int4 训练中不使用双重量化。（默认值：False）

--quantization_device_map {auto}, --quantization-device-map {auto}
用于推理 4 位量化模型的设备映射，需要 bitsandbytes>=0.43.0。（默认值：None）

### 多媒体相关参数

--image_max_pixels IMAGE_MAX_PIXELS, --image-max-pixels IMAGE_MAX_PIXELS
图像输入的最大像素数。（默认值：589824）

--image_min_pixels IMAGE_MIN_PIXELS, --image-min-pixels IMAGE_MIN_PIXELS
图像输入的最小像素数。（默认值：1024）

--image_do_pan_and_scan [IMAGE_DO_PAN_AND_SCAN], --image-do-pan-and-scan [IMAGE_DO_PAN_AND_SCAN]
对 gemma3 模型使用平移扫描处理图像。（默认值：False）

--crop_to_patches [CROP_TO_PATCHES], --crop-to-patches [CROP_TO_PATCHES]
对 internvl 模型是否将图像裁剪为补丁。（默认值：False）

--video_max_pixels VIDEO_MAX_PIXELS, --video-max-pixels VIDEO_MAX_PIXELS
视频输入的最大像素数。（默认值：65536）

--video_min_pixels VIDEO_MIN_PIXELS, --video-min-pixels VIDEO_MIN_PIXELS
视频输入的最小像素数。（默认值：256）

--video_fps VIDEO_FPS, --video-fps VIDEO_FPS
视频输入的每秒采样帧数。（默认值：2.0）

--video_maxlen VIDEO_MAXLEN, --video-maxlen VIDEO_MAXLEN
视频输入的最大采样帧数。（默认值：128）

--use_audio_in_video [USE_AUDIO_IN_VIDEO], --use-audio-in-video [USE_AUDIO_IN_VIDEO]
在视频输入中是否使用音频。（默认值：False）

--audio_sampling_rate AUDIO_SAMPLING_RATE, --audio-sampling-rate AUDIO_SAMPLING_RATE
音频输入的采样率。（默认值：16000）

### 模型导出参数

--export_dir EXPORT_DIR, --export-dir EXPORT_DIR
保存导出模型的目录路径。（默认值：None）

--export_size EXPORT_SIZE, --export-size EXPORT_SIZE
导出模型的文件分片大小（以 GB 为单位）。（默认值：5）

--export_device {cpu,auto}, --export-device {cpu,auto}
模型导出时使用的设备，使用 `auto` 可加速导出。（默认值：cpu）

--export_quantization_bit EXPORT_QUANTIZATION_BIT, --export-quantization-bit EXPORT_QUANTIZATION_BIT
导出模型的量化位数。（默认值：None）

--export_quantization_dataset EXPORT_QUANTIZATION_DATASET, --export-quantization-dataset EXPORT_QUANTIZATION_DATASET
用于量化导出模型的数据集路径或数据集名称。（默认值：None）

--export_quantization_nsamples EXPORT_QUANTIZATION_NSAMPLES, --export-quantization-nsamples EXPORT_QUANTIZATION_NSAMPLES
用于量化的样本数量。（默认值：128）

--export_quantization_maxlen EXPORT_QUANTIZATION_MAXLEN, --export-quantization-maxlen EXPORT_QUANTIZATION_MAXLEN
用于量化的模型输入的最大长度。（默认值：1024）

--export_legacy_format [EXPORT_LEGACY_FORMAT], --export-legacy-format [EXPORT_LEGACY_FORMAT]
是否保存为 `.bin` 文件而非 `.safetensors` 文件。（默认值：False）

--export_hub_model_id EXPORT_HUB_MODEL_ID, --export-hub-model-id EXPORT_HUB_MODEL_ID
将模型推送到 Hugging Face Hub 时的仓库名称。（默认值：None）

### KTransformers 相关参数

--use_kt [USE_KT], --use-kt [USE_KT]
在 LoRA 训练中是否使用 KTransformers 优化。（默认值：False）

--kt_optimize_rule KT_OPTIMIZE_RULE, --kt-optimize-rule KT_OPTIMIZE_RULE
KTransformers 优化规则的路径；详见 https://github.com/kvcache-ai/ktransformers/。（默认值：None）

--cpu_infer CPU_INFER, --cpu-infer CPU_INFER
用于计算的 CPU 核心数。（默认值：32）

--chunk_size CHUNK_SIZE, --chunk-size CHUNK_SIZE
KTransformers 中用于 CPU 计算的块大小。（默认值：8192）

--mode MODE
Llama 模型的模式（正常模式或长上下文模式）。（默认值：normal）

--kt_maxlen KT_MAXLEN, --kt-maxlen KT_MAXLEN
KT 引擎的最大序列（提示词 + 响应）长度。（默认值：4096）

--kt_use_cuda_graph [KT_USE_CUDA_GRAPH], --kt-use-cuda-graph [KT_USE_CUDA_GRAPH]
KT 引擎是否使用 CUDA 图。（默认值：True）

--no_kt_use_cuda_graph, --no-kt-use-cuda-graph
KT 引擎不使用 CUDA 图。（默认值：False）

--kt_mode KT_MODE, --kt-mode KT_MODE
KT 引擎的模式（正常模式或长上下文模式）。（默认值：normal）

--kt_force_think [KT_FORCE_THINK], --kt-force-think [KT_FORCE_THINK]
KT 引擎的强制思考开关。（默认值：False）

### VLLM 相关参数

--vllm_maxlen VLLM_MAXLEN, --vllm-maxlen VLLM_MAXLEN
VLLM 引擎的最大序列（提示词 + 响应）长度。（默认值：4096）

--vllm_gpu_util VLLM_GPU_UTIL, --vllm-gpu-util VLLM_GPU_UTIL
VLLM 引擎使用的 GPU 内存比例（0,1）。（默认值：0.7）

--vllm_enforce_eager [VLLM_ENFORCE_EAGER], --vllm-enforce-eager [VLLM_ENFORCE_EAGER]
VLLM 引擎中是否禁用 CUDA 图。（默认值：False）

--vllm_max_lora_rank VLLM_MAX_LORA_RANK, --vllm-max-lora-rank VLLM_MAX_LORA_RANK
VLLM 引擎中所有 LoRA 的最大秩。（默认值：32）

--vllm_config VLLM_CONFIG, --vllm-config VLLM_CONFIG
用于初始化 VLLM 引擎的配置。请使用 JSON 字符串。（默认值：None）

### SGLang 相关参数

--sglang_maxlen SGLANG_MAXLEN, --sglang-maxlen SGLANG_MAXLEN
SGLang 引擎的最大序列（提示词 + 响应）长度。（默认值：4096）

--sglang_mem_fraction SGLANG_MEM_FRACTION, --sglang-mem-fraction SGLANG_MEM_FRACTION
SGLang 引擎使用的内存比例（0-1）。（默认值：0.7）

--sglang_tp_size SGLANG_TP_SIZE, --sglang-tp-size SGLANG_TP_SIZE
SGLang 引擎的张量并行大小。（默认值：-1）

--sglang_config SGLANG_CONFIG, --sglang-config SGLANG_CONFIG
用于初始化 SGLang 引擎的配置。请使用 JSON 字符串。（默认值：None）

--sglang_lora_backend {triton,flashinfer}, --sglang-lora-backend {triton,flashinfer}
用于运行 LoRA 模块 GEMM 内核的后端。建议使用 Triton LoRA 后端以获得更好的性能和稳定性。（默认值：triton）

### 数据集相关参数

--template TEMPLATE
用于在训练和推理中构造提示词的模板。（默认值：None）

--dataset DATASET
用于训练的数据集名称。多个数据集用逗号分隔。（默认值：None）

--eval_dataset EVAL_DATASET, --eval-dataset EVAL_DATASET
用于评估的数据集名称。多个数据集用逗号分隔。（默认值：None）

--dataset_dir DATASET_DIR, --dataset-dir DATASET_DIR
包含数据集的文件夹路径。（默认值：data）

--media_dir MEDIA_DIR, --media-dir MEDIA_DIR
包含图像、视频或音频的文件夹路径。默认为 `dataset_dir`。（默认值：None）

--cutoff_len CUTOFF_LEN, --cutoff-len CUTOFF_LEN
数据集中分词后输入的截断长度。（默认值：2048）

--train_on_prompt [TRAIN_ON_PROMPT], --train-on-prompt [TRAIN_ON_PROMPT]
是否禁用对提示词的掩码。（默认值：False）

--mask_history [MASK_HISTORY], --mask-history [MASK_HISTORY]
是否掩码历史记录并仅在最后一轮进行训练。（默认值：False）

--streaming [STREAMING]
启用数据集流式传输。（默认值：False）

--buffer_size BUFFER_SIZE, --buffer-size BUFFER_SIZE
数据集流式传输中用于随机采样样本的缓冲区大小。（默认值：16384）

--mix_strategy {concat,interleave_under,interleave_over}, --mix-strategy {concat,interleave_under,interleave_over}
数据集混合使用的策略（拼接/交错，下采样/上采样）。（默认值：concat）

--interleave_probs INTERLEAVE_PROBS, --interleave-probs INTERLEAVE_PROBS
从各个数据集采样数据的概率。多个数据集用逗号分隔。（默认值：None）

--overwrite_cache [OVERWRITE_CACHE], --overwrite-cache [OVERWRITE_CACHE]
覆盖缓存的训练集和评估集。（默认值：False）

--preprocessing_batch_size PREPROCESSING_BATCH_SIZE, --preprocessing-batch-size PREPROCESSING_BATCH_SIZE
预处理中每组的样本数量。（默认值：1000）

--preprocessing_num_workers PREPROCESSING_NUM_WORKERS, --preprocessing-num-workers PREPROCESSING_NUM_WORKERS
用于预处理的进程数。（默认值：None）

--max_samples MAX_SAMPLES, --max-samples MAX_SAMPLES
用于调试，截断每个数据集的样本数量。（默认值：None）

--eval_num_beams EVAL_NUM_BEAMS, --eval-num-beams EVAL_NUM_BEAMS
评估时使用的束搜索数量。此参数将传递给 `model.generate`（默认值：None）

--ignore_pad_token_for_loss [IGNORE_PAD_TOKEN_FOR_LOSS], --ignore-pad-token-for-loss [IGNORE_PAD_TOKEN_FOR_LOSS]
在损失计算中是否忽略与填充标签对应的令牌。（默认值：True）

--no_ignore_pad_token_for_loss, --no-ignore-pad-token-for-loss
在损失计算中不忽略与填充标签对应的令牌。（默认值：False）

--val_size VAL_SIZE, --val-size VAL_SIZE
验证集的大小，应为整数或 [0,1) 范围内的浮点数。（默认值：0.0）

--eval_on_each_dataset [EVAL_ON_EACH_DATASET], --eval-on-each-dataset [EVAL_ON_EACH_DATASET]
是否分别在每个数据集上进行评估。（默认值：False）

--packing PACKING
在训练中启用序列打包。预训练时将自动启用。（默认值：None）

--neat_packing [NEAT_PACKING], --neat-packing [NEAT_PACKING]
启用无交叉注意力的序列打包。（默认值：False）

--tool_format TOOL_FORMAT, --tool-format TOOL_FORMAT
用于构造函数调用示例的工具格式。（默认值：None）

--default_system DEFAULT_SYSTEM, --default-system DEFAULT_SYSTEM
覆盖模板中的默认系统消息。（默认值：None）

--enable_thinking [ENABLE_THINKING], --enable-thinking [ENABLE_THINKING]
对于推理模型是否启用思考模式。（默认值：True）

--no_enable_thinking, --no-enable-thinking
对于推理模型不启用思考模式。（默认值：False）

--tokenized_path TOKENIZED_PATH, --tokenized-path TOKENIZED_PATH
保存或加载分词后数据集的路径。若路径不存在，将保存分词后数据集；若路径存在，将加载分词后数据集。（默认值：None）

--data_shared_file_system [DATA_SHARED_FILE_SYSTEM], --data-shared-file-system [DATA_SHARED_FILE_SYSTEM]
是否使用共享文件系统存储数据集。（默认值：False）

### 训练核心参数

--output_dir OUTPUT_DIR, --output-dir OUTPUT_DIR
用于写入模型预测结果和检查点的输出目录。若未提供，默认为 'trainer_output'。（默认值：None）

--overwrite_output_dir [OVERWRITE_OUTPUT_DIR], --overwrite-output-dir [OVERWRITE_OUTPUT_DIR]
已废弃（默认值：False）

--do_train [DO_TRAIN], --do-train [DO_TRAIN]
是否运行训练。（默认值：False）

--do_eval [DO_EVAL], --do-eval [DO_EVAL]
是否在开发集上运行评估。（默认值：False）

--do_predict [DO_PREDICT], --do-predict [DO_PREDICT]
是否在测试集上运行预测。（默认值：False）

--eval_strategy {no,steps,epoch}, --eval-strategy {no,steps,epoch}
使用的评估策略。（默认值：no）

--prediction_loss_only [PREDICTION_LOSS_ONLY], --prediction-loss-only [PREDICTION_LOSS_ONLY]
执行评估和预测时，仅返回损失。（默认值：False）

--per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE, --per-device-train-batch-size PER_DEVICE_TRAIN_BATCH_SIZE
训练时每个设备加速器核心/CPU 的批次大小。（默认值：8）

--per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE, --per-device-eval-batch-size PER_DEVICE_EVAL_BATCH_SIZE
评估时每个设备加速器核心/CPU 的批次大小。（默认值：8）

--per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE, --per-gpu-train-batch-size PER_GPU_TRAIN_BATCH_SIZE
已废弃，建议使用 `--per_device_train_batch_size`。训练时每个 GPU/TPU 核心/CPU 的批次大小。（默认值：None）

--per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE, --per-gpu-eval-batch-size PER_GPU_EVAL_BATCH_SIZE
已废弃，建议使用 `--per_device_eval_batch_size`。评估时每个 GPU/TPU 核心/CPU 的批次大小。（默认值：None）

--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS, --gradient-accumulation-steps GRADIENT_ACCUMULATION_STEPS
在执行反向传播/更新步骤之前累积的更新步骤数。（默认值：1）

--eval_accumulation_steps EVAL_ACCUMULATION_STEPS, --eval-accumulation-steps EVAL_ACCUMULATION_STEPS
在将张量移至 CPU 之前累积的预测步骤数。（默认值：None）

--eval_delay EVAL_DELAY, --eval-delay EVAL_DELAY
根据评估策略，首次评估前需要等待的 epoch 数或步骤数。（默认值：0）

--torch_empty_cache_steps TORCH_EMPTY_CACHE_STEPS, --torch-empty-cache-steps TORCH_EMPTY_CACHE_STEPS
调用 `torch.<device>.empty_cache()` 之前等待的步骤数。这有助于通过降低峰值 VRAM 使用率来避免 CUDA 内存不足错误，但会牺牲约 10% 的性能（详见 https://github.com/huggingface/transformers/issues/31372）。若未设置或设为 None，将不会清空缓存。（默认值：None）

--learning_rate LEARNING_RATE, --learning-rate LEARNING_RATE
AdamW 的初始学习率。（默认值：5e-05）

--weight_decay WEIGHT_DECAY, --weight-decay WEIGHT_DECAY
AdamW 的权重衰减（若启用）。（默认值：0.0）

--adam_beta1 ADAM_BETA1, --adam-beta1 ADAM_BETA1
AdamW 优化器的 Beta1 参数（默认值：0.9）

--adam_beta2 ADAM_BETA2, --adam-beta2 ADAM_BETA2
AdamW 优化器的 Beta2 参数（默认值：0.999）

--adam_epsilon ADAM_EPSILON, --adam-epsilon ADAM_EPSILON
AdamW 优化器的 Epsilon 参数。（默认值：1e-08）

--max_grad_norm MAX_GRAD_NORM, --max-grad-norm MAX_GRAD_NORM
最大梯度范数。（默认值：1.0）

--num_train_epochs NUM_TRAIN_EPOCHS, --num-train-epochs NUM_TRAIN_EPOCHS
要执行的训练 epoch 总数。（默认值：3.0）

--max_steps MAX_STEPS, --max-steps MAX_STEPS
若 > 0：设置要执行的训练步骤总数。覆盖 num_train_epochs。（默认值：-1）

--lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup,inverse_sqrt,reduce_lr_on_plateau,cosine_with_min_lr,cosine_warmup_with_min_lr,warmup_stable_decay}, --lr-scheduler-type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup,inverse_sqrt,reduce_lr_on_plateau,cosine_with_min_lr,cosine_warmup_with_min_lr,warmup_stable_decay}
使用的学习率调度器类型。（默认值：linear）

--lr_scheduler_kwargs LR_SCHEDULER_KWARGS, --lr-scheduler-kwargs LR_SCHEDULER_KWARGS
学习率调度器的额外参数，例如用于带硬重启的余弦调度器的 {'num_cycles': 1}。（默认值：{}）

--warmup_ratio WARMUP_RATIO, --warmup-ratio WARMUP_RATIO
在总步骤的 warmup_ratio 比例内进行线性预热。（默认值：0.0）

--warmup_steps WARMUP_STEPS, --warmup-steps WARMUP_STEPS
线性预热的步骤数。（默认值：0）

### 日志与保存参数

--log_level {detail,debug,info,warning,error,critical,passive}, --log-level {detail,debug,info,warning,error,critical,passive}
主节点上使用的日志级别。可选值为日志级别字符串：'debug'、'info'、'warning'、'error' 和 'critical'，以及 'passive' 级别（不设置任何内容，由应用程序自行设置级别）。默认为 'passive'。（默认值：passive）

--log_level_replica {detail,debug,info,warning,error,critical,passive}, --log-level-replica {detail,debug,info,warning,error,critical,passive}
副本节点上使用的日志级别。可选值和默认值与 `log_level` 相同（默认值：warning）

--log_on_each_node [LOG_ON_EACH_NODE], --log-on-each-node [LOG_ON_EACH_NODE]
进行多节点分布式训练时，是否每个节点都记录日志，还是仅在主节点记录一次。（默认值：True）

--no_log_on_each_node, --no-log-on-each-node
进行多节点分布式训练时，仅在主节点记录日志，不在每个节点都记录。（默认值：False）

--logging_dir LOGGING_DIR, --logging-dir LOGGING_DIR
Tensorboard 日志目录。（默认值：None）

--logging_strategy {no,steps,epoch}, --logging-strategy {no,steps,epoch}
使用的日志记录策略。（默认值：steps）

--logging_first_step [LOGGING_FIRST_STEP], --logging-first-step [LOGGING_FIRST_STEP]
记录第一个 global_step 的日志（默认值：False）

--logging_steps LOGGING_STEPS, --logging-steps LOGGING_STEPS
每 X 个更新步骤记录一次日志。应为整数或 [0,1) 范围内的浮点数。若小于 1，将被解释为总训练步骤的比例。（默认值：500）

--logging_nan_inf_filter [LOGGING_NAN_INF_FILTER], --logging-nan-inf-filter [LOGGING_NAN_INF_FILTER]
过滤日志中的 NaN 和 Inf 损失。（默认值：True）

--no_logging_nan_inf_filter, --no-logging-nan-inf-filter
不过滤日志中的 NaN 和 Inf 损失。（默认值：False）

--save_strategy {no,steps,epoch,best}, --save-strategy {no,steps,epoch,best}
检查点保存策略。（默认值：steps）

--save_steps SAVE_STEPS, --save-steps SAVE_STEPS
每 X 个更新步骤保存一次检查点。应为整数或 [0,1) 范围内的浮点数。若小于 1，将被解释为总训练步骤的比例。（默认值：500）

--save_total_limit SAVE_TOTAL_LIMIT, --save-total-limit SAVE_TOTAL_LIMIT
若设置此值，将限制检查点的总数。删除 `output_dir` 中较旧的检查点。当启用 `load_best_model_at_end` 时，根据 `metric_for_best_model` 确定的“最佳”检查点将始终与最新的检查点一起保留。例如，若 `save_total_limit=5` 且 `load_best_model_at_end=True`，则最后四个检查点将始终与最佳模型一起保留。若 `save_total_limit=1` 且 `load_best_model_at_end=True`，可能会保存两个检查点：最后一个和最佳一个（若它们不同）。默认无检查点数量限制（默认值：None）

--save_safetensors [SAVE_SAFETENSORS], --save-safetensors [SAVE_SAFETENSORS]
使用 safetensors 保存和加载状态字典，而非默认的 torch.load 和 torch.save。（默认值：True）

--no_save_safetensors, --no-save-safetensors
不使用 safetensors 保存和加载状态字典，使用默认的 torch.load 和 torch.save。（默认值：False）

--save_on_each_node [SAVE_ON_EACH_NODE], --save-on-each-node [SAVE_ON_EACH_NODE]
进行多节点分布式训练时，是否在每个节点上保存模型和检查点，还是仅在主节点上保存（默认值：False）

--save_only_model [SAVE_ONLY_MODEL], --save-only-model [SAVE_ONLY_MODEL]
保存检查点时，是否仅保存模型，还是同时保存优化器、调度器和随机数生成器状态。注意：若设为 True，将无法从检查点恢复训练。此选项可通过不存储优化器、调度器和随机数生成器状态来节省存储空间。仅可使用 from_pretrained 加载此选项保存的模型。（默认值：False）

--restore_callback_states_from_checkpoint [RESTORE_CALLBACK_STATES_FROM_CHECKPOINT], --restore-callback-states-from-checkpoint [RESTORE_CALLBACK_STATES_FROM_CHECKPOINT]
是否从检查点恢复回调状态。若为 `True`，将覆盖传递给 `Trainer` 的回调（若它们存在于检查点中）。（默认值：False）

### 设备与精度参数

--no_cuda [NO_CUDA], --no-cuda [NO_CUDA]
此参数已废弃。将在 🤗 Transformers 5.0 版本中移除。（默认值：False）

--use_cpu [USE_CPU], --use-cpu [USE_CPU]
是否使用 CPU。若设为 False，将使用可用的 torch 设备/后端（cuda/mps/xpu/hpu 等）（默认值：False）

--use_mps_device [USE_MPS_DEVICE], --use-mps-device [USE_MPS_DEVICE]
此参数已废弃。`mps` 设备将与 `cuda` 设备类似，在可用时自动使用。将在 🤗 Transformers 5.0 版本中移除（默认值：False）

--seed SEED
训练开始时设置的随机种子。（默认值：42）

--data_seed DATA_SEED, --data-seed DATA_SEED
用于数据采样器的随机种子。（默认值：None）

--jit_mode_eval [JIT_MODE_EVAL], --jit-mode-eval [JIT_MODE_EVAL]
推理时是否使用 PyTorch jit 跟踪（默认值：False）

--bf16 [BF16]
是否使用 bf16（混合）精度而非 32 位精度。需要 Ampere 或更高版本的 NVIDIA 架构，或使用 CPU（use_cpu）或 Ascend NPU。此为实验性 API，可能会发生变化。（默认值：False）

--fp16 [FP16]
是否使用 fp16（混合）精度而非 32 位精度（默认值：False）

--fp16_opt_level FP16_OPT_LEVEL, --fp16-opt-level FP16_OPT_LEVEL
对于 fp16：Apex AMP 优化级别，可选 ['O0', 'O1', 'O2', 'O3']。详见 https://nvidia.github.io/apex/amp.html（默认值：O1）

--half_precision_backend {auto,apex,cpu_amp}, --half-precision-backend {auto,apex,cpu_amp}
用于半精度的后端。（默认值：auto）

--bf16_full_eval [BF16_FULL_EVAL], --bf16-full-eval [BF16_FULL_EVAL]
是否使用完整的 bfloat16 精度进行评估而非 32 位精度。此为实验性 API，可能会发生变化。（默认值：False）

--fp16_full_eval [FP16_FULL_EVAL], --fp16-full-eval [FP16_FULL_EVAL]
是否使用完整的 float16 精度进行评估而非 32 位精度（默认值：False）

--tf32 TF32
是否启用 tf32 模式，适用于 Ampere 及更新版本的 GPU 架构。此为实验性 API，可能会发生变化。（默认值：None）

### 分布式训练参数

--local_rank LOCAL_RANK, --local-rank LOCAL_RANK
分布式训练时的本地秩（默认值：-1）

--ddp_backend {nccl,gloo,mpi,ccl,hccl,cncl,mccl}, --ddp-backend {nccl,gloo,mpi,ccl,hccl,cncl,mccl}
用于分布式训练的后端。（默认值：None）

--tpu_num_cores TPU_NUM_CORES, --tpu-num-cores TPU_NUM_CORES
TPU：TPU 核心数（由启动脚本自动传递）（默认值：None）

--tpu_metrics_debug [TPU_METRICS_DEBUG], --tpu-metrics-debug [TPU_METRICS_DEBUG]
已废弃，建议使用 `--debug tpu_metrics_debug`。TPU：是否打印调试指标（默认值：False）

--debug DEBUG [DEBUG ...]
是否启用调试模式。当前选项：`underflow_overflow`（检测激活值和权重中的下溢和上溢）、`tpu_metrics_debug`（在 TPU 上打印调试指标）。（默认值：None）

--ddp_find_unused_parameters DDP_FIND_UNUSED_PARAMETERS, --ddp-find-unused-parameters DDP_FIND_UNUSED_PARAMETERS
进行分布式训练时，传递给 `DistributedDataParallel` 的 `find_unused_parameters` 标志的值。（默认值：None）

--ddp_bucket_cap_mb DDP_BUCKET_CAP_MB, --ddp-bucket-cap-mb DDP_BUCKET_CAP_MB
进行分布式训练时，传递给 `DistributedDataParallel` 的 `bucket_cap_mb` 标志的值。（默认值：None）

--ddp_broadcast_buffers DDP_BROADCAST_BUFFERS, --ddp-broadcast-buffers DDP_BROADCAST_BUFFERS
进行分布式训练时，传递给 `DistributedDataParallel` 的 `broadcast_buffers` 标志的值。（默认值：None）

--ddp_timeout DDP_TIMEOUT, --ddp-timeout DDP_TIMEOUT
覆盖分布式训练的默认超时时间（值应以秒为单位）。（默认值：1800）

--fsdp FSDP
在分布式训练中是否使用 PyTorch 完全分片数据并行（FSDP）训练。基本选项应为 `full_shard`、`shard_grad_op` 或 `no_shard`，可通过以下方式为 `full_shard` 或 `shard_grad_op` 添加 CPU 卸载：`full_shard offload` 或 `shard_grad_op offload`。也可通过相同语法为 `full_shard` 或 `shard_grad_op` 添加自动包装：`full_shard auto_wrap` 或 `shard_grad_op auto_wrap`。（默认值：None）

--fsdp_min_num_params FSDP_MIN_NUM_PARAMS, --fsdp-min-num-params FSDP_MIN_NUM_PARAMS
此参数已废弃。FSDP 默认自动包装的最小参数数量。（仅在传递 `fsdp` 字段时有用）。（默认值：0）

--fsdp_config FSDP_CONFIG, --fsdp-config FSDP_CONFIG
用于 FSDP（Pytorch 完全分片数据并行）的配置。值可以是 fsdp json 配置文件（例如 `fsdp_config.json`）或已加载的 json 文件（作为 `dict`）。（默认值：None）

--fsdp_transformer_layer_cls_to_wrap FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP, --fsdp-transformer-layer-cls-to-wrap FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP
此参数已废弃。要包装的 Transformer 层类名（区分大小写），例如 `BertLayer`、`GPTJBlock`、`T5Block` 等。（仅在传递 `fsdp` 标志时有用）。（默认值：None）

--deepspeed DEEPSPEED
启用 deepspeed 并传递 deepspeed json 配置文件的路径（例如 `ds_config.json`）或已加载的 json 文件（作为 dict）（默认值：None）

### 数据加载参数

--dataloader_drop_last [DATALOADER_DROP_LAST], --dataloader-drop-last [DATALOADER_DROP_LAST]
若最后一个批次不完整（无法被批次大小整除），是否丢弃该批次。（默认值：False）

--eval_steps EVAL_STEPS, --eval-steps EVAL_STEPS
每 X 个步骤运行一次评估。应为整数或 [0,1) 范围内的浮点数。若小于 1，将被解释为总训练步骤的比例。（默认值：None）

--dataloader_num_workers DATALOADER_NUM_WORKERS, --dataloader-num-workers DATALOADER_NUM_WORKERS
用于数据加载的子进程数（仅 PyTorch）。0 表示在主进程中加载数据。（默认值：0）

--dataloader_prefetch_factor DATALOADER_PREFETCH_FACTOR, --dataloader-prefetch-factor DATALOADER_PREFETCH_FACTOR
每个工作进程预先加载的批次数量。2 表示所有工作进程总共会预先加载 2 * num_workers 个批次。（默认值：None）

--dataloader_pin_memory [DATALOADER_PIN_MEMORY], --dataloader-pin-memory [DATALOADER_PIN_MEMORY]
DataLoader 是否使用固定内存。（默认值：True）

--no_dataloader_pin_memory, --no-dataloader-pin-memory
DataLoader 不使用固定内存。（默认值：False）

--dataloader_persistent_workers [DATALOADER_PERSISTENT_WORKERS], --dataloader-persistent-workers [DATALOADER
> （注：文档部分内容可能由 AI 生成）