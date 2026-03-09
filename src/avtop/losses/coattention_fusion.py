#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
协同注意力融合模块 (Co-Attention Fusion)
实现视频↔音频的双向交叉注意力机制

核心思想：
- 视频特征作为Query，音频特征作为Key/Value（Video attends to Audio）
- 音频特征作为Query，视频特征作为Key/Value（Audio attends to Video）
- 通过残差连接保留原始特征
- 最终输出融合的跨模态表示

输入：
  video: [B, T, D] - 视频帧序列特征
  audio: [B, T, D] - 音频时间序列特征

输出：
  fused: [B, T, D] - 融合后的特征
  attn_weights: dict - 注意力权重（可选用于可视化）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadCrossAttention(nn.Module):
    """
    多头交叉注意力
    Q来自一个模态，K/V来自另一个模态
    """
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [B, T_q, D]
            key: [B, T_k, D]
            value: [B, T_v, D] (通常T_k = T_v)
            mask: [B, T_q, T_k] (可选)
        
        Returns:
            output: [B, T_q, D]
            attn_weights: [B, num_heads, T_q, T_k]
        """
        B, T_q, _ = query.shape
        T_k = key.shape[1]
        
        # 1. 线性投影并reshape为多头
        Q = self.W_q(query).view(B, T_q, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, T_q, d_k]
        K = self.W_k(key).view(B, T_k, self.num_heads, self.d_k).transpose(1, 2)    # [B, H, T_k, d_k]
        V = self.W_v(value).view(B, T_k, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, T_k, d_k]
        
        # 2. 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, H, T_q, T_k]
        
        # 3. 应用mask（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        
        # 4. Softmax得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # [B, H, T_q, T_k]
        attn_weights = self.dropout(attn_weights)
        
        # 5. 加权求和
        output = torch.matmul(attn_weights, V)  # [B, H, T_q, d_k]
        
        # 6. 拼接多头并投影
        output = output.transpose(1, 2).contiguous().view(B, T_q, self.d_model)  # [B, T_q, D]
        output = self.W_o(output)
        
        # 7. 残差连接 + Layer Norm
        output = self.norm(query + self.dropout(output))
        
        return output, attn_weights


class CoAttentionBlock(nn.Module):
    """
    协同注意力块
    实现双向交叉注意力：Video↔Audio
    """
    def __init__(self, d_model=256, num_heads=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        
        # 视频→音频的交叉注意力（Video attends to Audio）
        self.v2a_attn = MultiHeadCrossAttention(d_model, num_heads, dropout)
        
        # 音频→视频的交叉注意力（Audio attends to Video）
        self.a2v_attn = MultiHeadCrossAttention(d_model, num_heads, dropout)
        
        # 前馈网络（对每个模态独立）
        self.ffn_video = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.ffn_audio = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer Norm
        self.norm_video = nn.LayerNorm(d_model)
        self.norm_audio = nn.LayerNorm(d_model)
        
    def forward(self, video, audio):
        """
        Args:
            video: [B, T, D] - 视频特征序列
            audio: [B, T, D] - 音频特征序列
        
        Returns:
            video_updated: [B, T, D] - 更新后的视频特征
            audio_updated: [B, T, D] - 更新后的音频特征
            attn_weights: dict - 注意力权重
        """
        # 1. 视频关注音频（Video queries Audio）
        video_attended, attn_v2a = self.v2a_attn(
            query=video,
            key=audio,
            value=audio
        )
        
        # 2. 音频关注视频（Audio queries Video）
        audio_attended, attn_a2v = self.a2v_attn(
            query=audio,
            key=video,
            value=video
        )
        
        # 3. 前馈网络 + 残差连接
        video_updated = self.norm_video(video_attended + self.ffn_video(video_attended))
        audio_updated = self.norm_audio(audio_attended + self.ffn_audio(audio_attended))
        
        # 返回注意力权重（用于可视化）
        attn_weights = {
            'v2a': attn_v2a,  # [B, H, T, T]
            'a2v': attn_a2v   # [B, H, T, T]
        }
        
        return video_updated, audio_updated, attn_weights


class AttentionPooling(nn.Module):
    """
    注意力池化：将序列特征聚合为全局表示
    用于生成 z^v 和 z^a
    """
    def __init__(self, d_model):
        super().__init__()
        
        # 学习注意力权重
        self.attn_weights = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, T, D] - 序列特征
        
        Returns:
            pooled: [B, D] - 全局表示
            weights: [B, T] - 注意力权重
        """
        # 计算注意力分数
        attn_scores = self.attn_weights(x).squeeze(-1)  # [B, T]
        attn_weights = F.softmax(attn_scores, dim=-1)   # [B, T]
        
        # 加权求和
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # [B, D]
        
        return pooled, attn_weights


class CoAttentionFusion(nn.Module):
    """
    完整的协同注意力融合模块
    
    包含：
    1. 输入投影（对齐维度）
    2. 多层协同注意力块
    3. 注意力池化（生成全局嵌入）
    4. 融合层
    
    与现有代码兼容：
    - 输入：video [B, T, D_v], audio [B, T, D_a]
    - 输出：fused [B, T, D_out]
    """
    def __init__(self, 
                 video_dim=512, 
                 audio_dim=256, 
                 d_model=256,
                 num_layers=2,
                 num_heads=8,
                 dim_feedforward=1024,
                 dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # 1. 输入投影（对齐到统一维度）
        self.video_proj = nn.Linear(video_dim, d_model)
        self.audio_proj = nn.Linear(audio_dim, d_model)
        
        # 2. 多层协同注意力块
        self.coattn_layers = nn.ModuleList([
            CoAttentionBlock(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # 3. 注意力池化（生成全局嵌入 z^v, z^a）
        self.video_pool = AttentionPooling(d_model)
        self.audio_pool = AttentionPooling(d_model)
        
        # 4. 融合层
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # 5. 输出层归一化
        self.output_norm = nn.LayerNorm(d_model)
        
    def forward(self, video, audio, return_embeddings=False):
        """
        Args:
            video: [B, T, D_v] - 视频特征序列
            audio: [B, T, D_a] - 音频特征序列
            return_embeddings: bool - 是否返回全局嵌入（用于对比学习）
        
        Returns:
            fused: [B, T, D] - 融合后的序列特征
            aux_info: dict - 辅助信息
                - video_emb: [B, D] - 视频全局嵌入 z^v
                - audio_emb: [B, D] - 音频全局嵌入 z^a
                - attn_weights: list of dict - 各层注意力权重
        """
        B, T, _ = video.shape
        
        # 1. 投影到统一维度
        video_proj = self.video_proj(video)  # [B, T, D]
        audio_proj = self.audio_proj(audio)  # [B, T, D]
        
        # 2. 通过多层协同注意力
        all_attn_weights = []
        
        for coattn_layer in self.coattn_layers:
            video_proj, audio_proj, attn_weights = coattn_layer(video_proj, audio_proj)
            all_attn_weights.append(attn_weights)
        
        # 3. 生成全局嵌入（用于对比学习）
        video_emb, video_pool_weights = self.video_pool(video_proj)  # [B, D]
        audio_emb, audio_pool_weights = self.audio_pool(audio_proj)  # [B, D]
        
        # 4. 融合全局嵌入
        global_fused = self.fusion(torch.cat([video_emb, audio_emb], dim=-1))  # [B, D]
        
        # 5. 广播回序列维度（融合的全局信息分配到每一帧）
        global_fused_expanded = global_fused.unsqueeze(1).expand(B, T, self.d_model)  # [B, T, D]
        
        # 6. 结合局部特征（视频+音频）和全局融合特征
        fused = self.output_norm(video_proj + audio_proj + global_fused_expanded)  # [B, T, D]
        
        # 构建辅助信息
        aux_info = {
            'video_emb': video_emb,              # [B, D] - 用于对比学习
            'audio_emb': audio_emb,              # [B, D] - 用于对比学习
            'video_seq': video_proj,             # [B, T, D] - 用于单模态分类
            'audio_seq': audio_proj,             # [B, T, D] - 用于单模态分类
            'attn_weights': all_attn_weights,    # 各层注意力权重
            'pool_weights': {
                'video': video_pool_weights,     # [B, T]
                'audio': audio_pool_weights      # [B, T]
            }
        }
        
        return fused, aux_info


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    # 测试协同注意力融合
    B, T = 4, 16
    video_dim, audio_dim = 512, 256
    d_model = 256
    
    # 创建模型
    fusion = CoAttentionFusion(
        video_dim=video_dim,
        audio_dim=audio_dim,
        d_model=d_model,
        num_layers=2,
        num_heads=8
    )
    
    # 创建测试数据
    video = torch.randn(B, T, video_dim)
    audio = torch.randn(B, T, audio_dim)
    
    # 前向传播
    fused, aux_info = fusion(video, audio)
    
    print("="*70)
    print("CoAttention Fusion 测试")
    print("="*70)
    print(f"输入:")
    print(f"  Video: {video.shape}")
    print(f"  Audio: {audio.shape}")
    print(f"\n输出:")
    print(f"  Fused: {fused.shape}")
    print(f"  Video Embedding (z^v): {aux_info['video_emb'].shape}")
    print(f"  Audio Embedding (z^a): {aux_info['audio_emb'].shape}")
    print(f"  Video Sequence: {aux_info['video_seq'].shape}")
    print(f"  Audio Sequence: {aux_info['audio_seq'].shape}")
    print(f"\n注意力权重:")
    for i, attn in enumerate(aux_info['attn_weights']):
        print(f"  Layer {i}:")
        print(f"    V2A: {attn['v2a'].shape}")
        print(f"    A2V: {attn['a2v'].shape}")
    
    print("\n✅ 测试通过!")
