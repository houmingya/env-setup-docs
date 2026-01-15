"""
Transformer 模型实现
这是一个完整的 Transformer 神经网络实现，包含自注意力机制、编码器和解码器
主要用于序列到序列(Seq2Seq)的任务，如机器翻译、文本摘要等
"""

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    自注意力机制模块
    实现了多头自注意力(Multi-Head Self-Attention)机制
    """
    def __init__(self, embed_size, heads):
        """
        初始化自注意力层
        
        参数:
            embed_size (int): 嵌入向量的维度大小
            heads (int): 注意力头的数量
        """
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size  # 嵌入维度
        self.heads = heads  # 注意力头数量
        self.head_dim = embed_size // heads  # 每个注意力头的维度

        # 确保嵌入维度可以被注意力头数量整除
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # 定义 Value, Key, Query 的线性变换层
        self.values = nn.Linear(embed_size, embed_size)  # Value 投影
        self.keys = nn.Linear(embed_size, embed_size)  # Key 投影
        self.queries = nn.Linear(embed_size, embed_size)  # Query 投影
        self.fc_out = nn.Linear(embed_size, embed_size)  # 输出层全连接

    def forward(self, values, keys, query, mask):
        """
        前向传播函数
        
        参数:
            values (Tensor): Value 向量，形状 (N, value_len, embed_size)
            keys (Tensor): Key 向量，形状 (N, key_len, embed_size)
            query (Tensor): Query 向量，形状 (N, query_len, embed_size)
            mask (Tensor): 掩码矩阵，用于屏蔽某些位置
            
        返回:
            out (Tensor): 注意力机制的输出，形状 (N, query_len, embed_size)
        """
        # 获取批次大小（训练样本数量）
        N = query.shape[0]

        # 获取 value, key, query 的序列长度
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 通过线性层进行投影变换
        values = self.values(values)  # (N, value_len, embed_size)
        keys = self.keys(keys)  # (N, key_len, embed_size)
        queries = self.queries(query)  # (N, query_len, embed_size)

        # 将嵌入向量分割成多个注意力头
        # 重塑张量以便进行多头注意力计算
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # 使用 einsum 计算注意力分数（Query 和 Key 的点积）
        # einsum 是一种高效的张量运算方式
        # "nqhd,nkhd->nhqk" 表示对 queries 和 keys 进行矩阵乘法
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries 形状: (N, query_len, heads, heads_dim)
        # keys 形状: (N, key_len, heads, heads_dim)
        # energy 形状: (N, heads, query_len, key_len)

        # 如果提供了掩码，则将被掩码的位置设置为极小值
        # 这样在 softmax 后这些位置的权重会接近于 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # 对注意力分数进行归一化处理
        # 除以缩放因子（嵌入维度的平方根）以提高训练稳定性
        # 然后应用 softmax 得到注意力权重
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention 形状: (N, heads, query_len, key_len)

        # 使用注意力权重对 values 进行加权求和
        # 然后重塑回原始的嵌入维度
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention 形状: (N, heads, query_len, key_len)
        # values 形状: (N, value_len, heads, heads_dim)
        # 矩阵乘法后: (N, query_len, heads, head_dim)
        # 重塑后将最后两个维度展平

        # 通过最后的全连接层进行输出变换
        out = self.fc_out(out)
        # 最终输出形状: (N, query_len, embed_size)

        return out


class TransformerBlock(nn.Module):
    """
    Transformer 块
    包含多头自注意力层、前馈神经网络以及残差连接和层归一化
    """
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        """
        初始化 Transformer 块
        
        参数:
            embed_size (int): 嵌入向量的维度
            heads (int): 注意力头的数量
            dropout (float): Dropout 比率，用于防止过拟合
            forward_expansion (int): 前馈网络的扩展倍数
        """
        super(TransformerBlock, self).__init__()
        # 自注意力层
        self.attention = SelfAttention(embed_size, heads)
        # 第一个层归一化（用于注意力层后）
        self.norm1 = nn.LayerNorm(embed_size)
        # 第二个层归一化（用于前馈网络后）
        self.norm2 = nn.LayerNorm(embed_size)

        # 前馈神经网络：两层全连接层，中间使用 ReLU 激活函数
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),  # 扩展维度
            nn.ReLU(),  # 激活函数
            nn.Linear(forward_expansion * embed_size, embed_size),  # 恢复维度
        )

        # Dropout 层，用于正则化
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        """
        前向传播
        
        参数:
            value (Tensor): Value 向量
            key (Tensor): Key 向量
            query (Tensor): Query 向量
            mask (Tensor): 掩码矩阵
            
        返回:
            out (Tensor): Transformer 块的输出
        """
        # 计算自注意力
        attention = self.attention(value, key, query, mask)

        # 添加残差连接，进行层归一化，然后应用 dropout
        x = self.dropout(self.norm1(attention + query))
        # 通过前馈神经网络
        forward = self.feed_forward(x)
        # 再次添加残差连接、层归一化和 dropout
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    """
    编码器
    由多个 Transformer 块堆叠而成，用于处理输入序列
    """
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):
        """
        初始化编码器
        
        参数:
            src_vocab_size (int): 源语言词汇表大小
            embed_size (int): 嵌入向量维度
            num_layers (int): Transformer 块的层数
            heads (int): 注意力头数量
            device (str): 计算设备 ('cpu' 或 'cuda')
            forward_expansion (int): 前馈网络扩展倍数
            dropout (float): Dropout 比率
            max_length (int): 最大序列长度
        """
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        # 词嵌入层：将词索引转换为嵌入向量
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        # 位置编码层：为每个位置添加位置信息
        self.position_embedding = nn.Embedding(max_length, embed_size)

        # 创建多层 Transformer 块
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        编码器前向传播
        
        参数:
            x (Tensor): 输入序列，形状 (N, seq_length)
            mask (Tensor): 掩码矩阵
            
        返回:
            out (Tensor): 编码器输出，形状 (N, seq_length, embed_size)
        """
        N, seq_length = x.shape
        # 生成位置索引
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        # 将词嵌入和位置嵌入相加，然后应用 dropout
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )

        # 在编码器中，query、key、value 都是相同的（自注意力）
        # 依次通过每一层 Transformer 块
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    """
    解码器块
    包含掩码自注意力、编码器-解码器注意力和前馈网络
    """
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        """
        初始化解码器块
        
        参数:
            embed_size (int): 嵌入向量维度
            heads (int): 注意力头数量
            forward_expansion (int): 前馈网络扩展倍数
            dropout (float): Dropout 比率
            device (str): 计算设备
        """
        super(DecoderBlock, self).__init__()
        # 层归一化
        self.norm = nn.LayerNorm(embed_size)
        # 掩码自注意力层（用于目标序列）
        self.attention = SelfAttention(embed_size, heads=heads)
        # Transformer 块（用于编码器-解码器注意力）
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        """
        解码器块前向传播
        
        参数:
            x (Tensor): 解码器输入
            value (Tensor): 来自编码器的 value
            key (Tensor): 来自编码器的 key
            src_mask (Tensor): 源序列掩码
            trg_mask (Tensor): 目标序列掩码（防止看到未来信息）
            
        返回:
            out (Tensor): 解码器块输出
        """
        # 对目标序列进行掩码自注意力计算
        attention = self.attention(x, x, x, trg_mask)
        # 残差连接和层归一化
        query = self.dropout(self.norm(attention + x))
        # 通过 Transformer 块进行编码器-解码器注意力计算
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    """
    解码器
    由多个解码器块堆叠而成，用于生成目标序列
    """
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        """
        初始化解码器
        
        参数:
            trg_vocab_size (int): 目标语言词汇表大小
            embed_size (int): 嵌入向量维度
            num_layers (int): 解码器块的层数
            heads (int): 注意力头数量
            forward_expansion (int): 前馈网络扩展倍数
            dropout (float): Dropout 比率
            device (str): 计算设备
            max_length (int): 最大序列长度
        """
        super(Decoder, self).__init__()
        self.device = device
        # 词嵌入层
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        # 位置嵌入层
        self.position_embedding = nn.Embedding(max_length, embed_size)

        # 创建多层解码器块
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        # 输出层：将嵌入向量映射到词汇表大小
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        """
        解码器前向传播
        
        参数:
            x (Tensor): 目标序列输入，形状 (N, seq_length)
            enc_out (Tensor): 编码器输出
            src_mask (Tensor): 源序列掩码
            trg_mask (Tensor): 目标序列掩码
            
        返回:
            out (Tensor): 解码器输出（词汇表概率分布），形状 (N, seq_length, trg_vocab_size)
        """
        N, seq_length = x.shape
        # 生成位置索引
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        # 词嵌入 + 位置嵌入 + dropout
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        # 依次通过每一层解码器块
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        # 通过输出层得到词汇表上的分数
        out = self.fc_out(x)

        return out


class Transformer(nn.Module):
    """
    完整的 Transformer 模型
    包含编码器和解码器，实现序列到序列的转换
    """
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cpu",
        max_length=100,
    ):
        """
        初始化 Transformer 模型
        
        参数:
            src_vocab_size (int): 源语言词汇表大小
            trg_vocab_size (int): 目标语言词汇表大小
            src_pad_idx (int): 源序列的填充索引
            trg_pad_idx (int): 目标序列的填充索引
            embed_size (int): 嵌入向量维度，默认 512
            num_layers (int): 编码器和解码器的层数，默认 6
            forward_expansion (int): 前馈网络扩展倍数，默认 4
            heads (int): 注意力头数量，默认 8
            dropout (float): Dropout 比率，默认 0
            device (str): 计算设备，默认 'cpu'
            max_length (int): 最大序列长度，默认 100
        """
        super(Transformer, self).__init__()

        # 初始化编码器
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        # 初始化解码器
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx  # 源序列填充索引
        self.trg_pad_idx = trg_pad_idx  # 目标序列填充索引
        self.device = device

    def make_src_mask(self, src):
        """
        创建源序列掩码
        将填充位置设为 0，其他位置设为 1
        
        参数:
            src (Tensor): 源序列，形状 (N, src_len)
            
        返回:
            src_mask (Tensor): 源序列掩码，形状 (N, 1, 1, src_len)
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        """
        创建目标序列掩码
        防止解码器在预测时看到未来的信息（因果掩码）
        
        参数:
            trg (Tensor): 目标序列，形状 (N, trg_len)
            
        返回:
            trg_mask (Tensor): 目标序列掩码，形状 (N, 1, trg_len, trg_len)
        """
        N, trg_len = trg.shape
        # 创建下三角矩阵，确保每个位置只能看到之前的位置
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask.to(self.device)

    def forward(self, src, trg):
        """
        Transformer 前向传播
        
        参数:
            src (Tensor): 源序列，形状 (N, src_len)
            trg (Tensor): 目标序列，形状 (N, trg_len)
            
        返回:
            out (Tensor): 模型输出（词汇表概率），形状 (N, trg_len, trg_vocab_size)
        """
        # 创建源序列和目标序列的掩码
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        # 编码源序列
        enc_src = self.encoder(src, src_mask)
        # 解码得到输出
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out


if __name__ == "__main__":
    """
    主程序入口
    演示如何创建和使用 Transformer 模型
    """
    # 检查是否有可用的 GPU，优先使用 GPU 进行计算
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建示例输入数据
    # x 是源序列，包含两个样本
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    # trg 是目标序列，包含两个样本
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    # 定义模型参数
    src_pad_idx = 0  # 源序列填充索引
    trg_pad_idx = 0  # 目标序列填充索引
    src_vocab_size = 10  # 源语言词汇表大小
    trg_vocab_size = 10  # 目标语言词汇表大小
    
    # 创建 Transformer 模型实例
    model = Transformer(
        src_vocab_size, 
        trg_vocab_size, 
        src_pad_idx, 
        trg_pad_idx, 
        device=device
    ).to(device)
    
    # 前向传播
    # 注意：解码器输入是目标序列去掉最后一个 token（教师强制）
    out = model(x, trg[:, :-1])
    
    # 打印输出形状
    # 输出形状应该是 (batch_size, target_sequence_length-1, target_vocab_size)
    print(f"模型输出形状: {out.shape}")
