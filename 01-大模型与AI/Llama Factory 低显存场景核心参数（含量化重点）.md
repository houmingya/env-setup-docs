# Llama Factory 低显存场景核心参数（含量化重点）

本文档聚焦低显存硬件环境（如消费级显卡、入门级服务器），筛选最常用的参数配置，重点详解量化相关参数，同时覆盖模型加载、推理优化等辅助降显存的关键参数，帮助用户在有限显存下高效运行模型训练/推理。

# 一、核心量化参数（低显存核心依赖）

此类参数直接降低模型显存占用，是低显存场景的核心配置，优先配置可使模型显存需求减少50%-75%。

|参数（短格式/长格式）|功能说明（低显存适配重点）|取值范围/格式|推荐值（低显存）|
|---|---|---|---|
|--quantization_method--quantization-method|指定量化方法，核心降显存手段。不同方法适配不同场景，低显存优先选成熟轻量化方案。|bnb、gptq、awq、aqlm、quanto、eetq、hqq、mxfp4、fp8|4G-8G显存：gptq/awq；2G-4G显存：aqlm/hqq|
|--quantization_bit--quantization-bit|量化位数，位数越低显存占用越少，但需平衡精度。低显存优先选4位，精度不足时再选8位。|整数（如 4、8）|优先4位；8G以上显存可尝试8位|
|--quantization_type--quantization-type|仅bitsandbytes（bnb）量化时生效，控制量化数据类型，nf4比fp4更适配通用场景。|fp4、nf4|nf4（精度更优，显存占用一致）|
|--double_quantization/--no_double_quantization--double-quantization/--no-double-quantization|bnb量化专属，开启双重量化可进一步降低显存占用（对精度影响极小）。|布尔值（--no_xxx 表示禁用）|--double_quantization=True（强制开启）|
|--quantization_device_map--quantization-device-map|指定量化模型的设备映射，auto模式自动分配显存，避免手动配置出错。|auto|auto（低显存必选，自动优化分配）|
|--export_quantization_bit--export-quantization-bit|导出模型时的量化位数，低显存场景导出时直接量化，避免后续重复处理占用显存。|整数（如 4、8）|与训练/推理量化位数一致（优先4位）|
|参数（短格式/长格式）|功能说明（低显存适配重点）|取值范围/格式|推荐值（低显存）|
|--low_cpu_mem_usage/--no_low_cpu_mem_usage--low-cpu-mem-usage/--no-low-cpu-mem-usage|内存高效加载模式，减少模型加载时的显存峰值占用（加载阶段最易爆显存）。|布尔值（--no_xxx 表示禁用）|--low_cpu_mem_usage=True（强制开启）|
|--offload_folder--offload-folder|模型权重卸载路径，将部分权重临时存到硬盘，缓解显存不足（牺牲少量速度）。|本地文件夹路径|默认offload（确保路径所在磁盘有足够空间）|
|--infer_dtype--infer-dtype|推理时数据类型，float16比float32显存占用少一半，auto模式会自动适配最优轻量化类型。|auto、float16、bfloat16、float32|float16（4G以上显存）；auto（2G-4G显存，自动选最优）|
|--per_device_train_batch_size/--per_device_eval_batch_size--per-device-train-batch-size/--per-device-eval-batch-size|控制单设备批次大小，低显存需最小化批次，避免批量处理时爆显存。|整数|训练：1-2；推理：2-4（根据显存剩余调整）|
|--gradient_accumulation_steps--gradient-accumulation-steps|梯度累积，批次太小时通过累积梯度保证训练效果，避免因批次小导致训练不稳定。|整数|4-8（与小批次搭配使用，平衡效果与显存）|
> （注：文档部分内容可能由 AI 生成）