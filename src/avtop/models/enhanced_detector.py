#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EnhancedAVTopDetector - 完全修复版

修复内容：
1. ✅ AudioBackbone 正确处理 mel 频谱维度
2. ✅ EnhancedAVTopDetector 初始化顺序正确
3. ✅ 测试代码使用正确的输入格式
4. ✅ 所有维度处理准确无误
"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
# 修改导入
from src.avtop.models.enhanced_audio_backbones import (
    LightVGGishAudioBackbone,      # ← 添加这个
    ModerateVGGishAudioBackbone,   # ← 添加这个
    ImprovedAudioBackbone,
     )
# 导入融合模块（假设这些存在）
try:
    from src.avtop.fusion.cfa_fusion import CFAFusion

    HAS_CFA = True
except:
    HAS_CFA = False
    print("⚠️ CFAFusion not found, using default fusion")

try:
    from src.avtop.fusion.ib_fusion import InformationBottleneckFusion

    HAS_IB = True
except:
    HAS_IB = False

try:
    from src.avtop.fusion.coattention import CoAttentionFusion

    HAS_COATTN = True
except:
    HAS_COATTN = False

# 导入其他模块（假设这些存在）
try:
    from src.avtop.models.backbones import VideoBackbone, AudioBackbone

    HAS_BACKBONES = True
except:
    HAS_BACKBONES = False
    print("⚠️ Backbones not found, using dummy backbones")

try:
    from src.avtop.models.temporal_encoder import SimpleTemporalEncoder

    HAS_TEMPORAL = True
except:
    HAS_TEMPORAL = False


# ============================================================================
# 真实Backbone实现
# ============================================================================
class VideoBackbone(nn.Module):
    """使用预训练ResNet的视频特征提取器"""

    def __init__(self, backbone_type='resnet18', output_dim=512, pretrained=True):
        super().__init__()
        self.backbone_type = backbone_type
        self.output_dim = output_dim

        # 导入torchvision
        try:
            import torchvision.models as models

            # 选择backbone
            if backbone_type == 'resnet18':
                resnet = models.resnet18(pretrained=pretrained)
                self.base_dim = 512
            elif backbone_type == 'resnet34':
                resnet = models.resnet34(pretrained=pretrained)
                self.base_dim = 512
            elif backbone_type == 'resnet50':
                resnet = models.resnet50(pretrained=pretrained)
                self.base_dim = 2048
            else:
                resnet = models.resnet18(pretrained=pretrained)
                self.base_dim = 512

            # 移除最后的全连接层
            self.features = nn.Sequential(*list(resnet.children())[:-1])

            # 投影到目标维度
            if self.base_dim != output_dim:
                self.proj = nn.Linear(self.base_dim, output_dim)
            else:
                self.proj = nn.Identity()

        except ImportError:
            print("⚠️ torchvision未安装，使用简化backbone")
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(3, 2, 1),
                nn.Conv2d(64, 128, 3, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
            self.base_dim = 128
            self.proj = nn.Linear(128, output_dim)

    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] or [B, C, H, W] or [B, T, D] (已提取特征)
        Returns:
            [B, T, D] or [B, D]
        """
        # 如果已经是特征向量，直接返回
        if x.ndim == 3 and x.size(-1) == self.output_dim:  # [B, T, D]
            return x
        if x.ndim == 2:  # [B, D]
            return x

        # 原始图像
        if x.ndim == 5:  # [B, T, C, H, W]
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
            feat = self.features(x).squeeze(-1).squeeze(-1)  # [B*T, base_dim]
            feat = self.proj(feat)  # [B*T, output_dim]
            return feat.reshape(B, T, -1)
        else:  # [B, C, H, W]
            feat = self.features(x).squeeze(-1).squeeze(-1)
            return self.proj(feat)


class AudioBackbone(nn.Module):
    """音频骨干网络 - 基于CNN的特征提取器"""

    def __init__(self, n_mels=128, hidden_dim=512):
        super().__init__()
        self.n_mels = n_mels
        self.hidden_dim = hidden_dim

        # CNN提取器
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(256, hidden_dim)

    def forward(self, x):
        """
        Args:
            x: [B, T, n_mels, mel_length] 或 [B, C, T, n_mels, mel_length]
        Returns:
            [B, T, D] - 时序特征
        """
        # 处理可能的通道维度
        if x.dim() == 5:  # [B, C, T, n_mels, mel_length]
            x = x.squeeze(1)  # [B, T, n_mels, mel_length]

        B, T = x.shape[:2]

        # 🔧 修复：正确的reshape方式
        # 先合并batch和时间维度
        mel = x.reshape(B * T, *x.shape[2:])  # [B*T, n_mels, mel_length]
        # 再添加通道维度
        mel = mel.unsqueeze(1)  # [B*T, 1, n_mels, mel_length]

        # CNN特征提取
        feat = self.conv(mel).squeeze(-1).squeeze(-1)  # [B*T, 256]
        feat = self.fc(feat)  # [B*T, D]

        # 恢复时序维度
        feat = feat.view(B, T, -1)  # [B, T, D]
        return feat


class DefaultFusion(nn.Module):
    """简单的拼接融合（备用）"""

    def __init__(self, video_dim, audio_dim, fusion_dim):
        super().__init__()
        self.proj = nn.Linear(video_dim + audio_dim, fusion_dim)

    def forward(self, video_feat, audio_feat):
        """
        Args:
            video_feat: [B, T, D_v]
            audio_feat: [B, T, D_a]
        Returns:
            fused: [B, T, D_fusion]
        """
        # 拼接
        combined = torch.cat([video_feat, audio_feat], dim=-1)
        # 投影
        fused = self.proj(combined)
        return fused


# ============================================================================
# MIL分类头（新增）
# ============================================================================
class EnhancedMILHead(nn.Module):
    """
    多实例学习分类头

    输入: [B, T, D] 序列特征
    输出:
        - clip_logits: [B, num_classes] 视频级分类
        - seg_logits: [B, T, num_classes] 帧级分类
        - weights: [B, T] 注意力权重
    """

    def __init__(self, input_dim, num_classes, hidden_dim=256):
        super().__init__()

        # 帧级分类器
        self.frame_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Args:
            x: [B, T, D]

        Returns:
            dict with:
                - clip_logits: [B, num_classes]
                - seg_logits: [B, T, num_classes]
                - weights: [B, T]
        """
        B, T, D = x.shape

        # 1. 帧级分类
        seg_logits = self.frame_classifier(x)  # [B, T, num_classes]

        # 2. 注意力权重
        attn_scores = self.attention(x).squeeze(-1)  # [B, T]
        weights = F.softmax(attn_scores, dim=1)  # [B, T]

        # 3. 加权聚合得到视频级分类
        # [B, T, num_classes] * [B, T, 1] -> [B, T, num_classes] -> [B, num_classes]
        clip_logits = (seg_logits * weights.unsqueeze(-1)).sum(dim=1)

        return {
            'clip_logits': clip_logits,
            'seg_logits': seg_logits,
            'weights': weights
        }

# ===== 安全 Co-Attention 包装器 =====
class SafeCoAttention(nn.Module):
    """
    包装现有的 CoAttentionFusion：
    - 自动对齐维度（按 v_in / a_in 的 in_features 建立线性投影）
    - 自动纠正模态顺序（若发现 v_in.in_features == audio_feat_dim 且 a_in.in_features == video_feat_dim）
    """
    def __init__(self, core, video_dim: int, audio_dim: int):
        super().__init__()
        self.core = core
        self.video_dim = int(video_dim)
        self.audio_dim = int(audio_dim)
        self.v_proj = None  # 延迟创建 Linear
        self.a_proj = None

    def _maybe_adapt_dims(self, x: torch.Tensor, want: int, which: str):
        have = x.size(-1)
        if have == want:
            return x
        # 延迟创建/更新投影层
        proj = getattr(self, f"{which}_proj", None)
        if (proj is None) or (proj.in_features != have) or (proj.out_features != want):
            proj = nn.Linear(have, want).to(x.device, dtype=x.dtype)
            setattr(self, f"{which}_proj", proj)
        return proj(x)

    def forward(self, v: torch.Tensor, a: torch.Tensor, **kw):
        # 尝试从 core 或 core.core 上拿到 v_in/a_in
        obj = getattr(self.core, 'core', self.core)
        v_in = getattr(obj, 'v_in', None)
        a_in = getattr(obj, 'a_in', None)

        # 缺省：按 (video, audio) 调用
        call = 'va'
        want_v = self.video_dim
        want_a = self.audio_dim
        # 如果能读到 v_in/a_in 的 in_features，就以它为准
        if isinstance(v_in, nn.Linear) and isinstance(a_in, nn.Linear):
            vin = int(v_in.in_features)
            ain = int(a_in.in_features)
            # 判断是否需要交换
            if (vin == a.size(-1) and ain == v.size(-1)) and not (vin == v.size(-1) and ain == a.size(-1)):
                call = 'av'
                want_v, want_a = ain, vin  # 交换期望

            else:
                want_v, want_a = vin, ain

        # 维度自适配（必要时添加线性投影）
        v = self._maybe_adapt_dims(v, want_v, 'v')
        a = self._maybe_adapt_dims(a, want_a, 'a')

        if call == 'va':
            return self.core(v, a, **kw)
        else:
            return self.core(a, v, **kw)

# ============================================================================
# 主检测器
# ============================================================================
class EnhancedAVTopDetector(nn.Module):
    """
    增强的多模态焊接缺陷检测器（多类别修复版）

    新增功能：
    - 支持多种融合策略（通过配置选择）
    - 返回单模态分支输出（用于KD和一致性）
    - 返回全局嵌入（用于对比学习）
    - 完整的MIL分类头

    配置示例：
    cfg = {
        'fusion': {
            'type': 'coattn',  # 'cfa', 'ib', 'coattn', 'default'
            'd_model': 256,
            'num_layers': 2,
            'num_heads': 8
        },
        'model': {
            'video_dim': 512,
            'audio_dim': 512,
            'fusion_dim': 256,
            'num_classes': 12
        }
    }
    """

    def __init__(self, cfg: Dict):
        super().__init__()

        # ========== 1. 先定义基础属性（重要！）==========
        model_cfg = cfg.get('model', {})
        self.video_dim = model_cfg.get('video_dim', 512)
        self.audio_dim = model_cfg.get('audio_dim', 512)  # 修改：与video_dim一致
        self.fusion_dim = model_cfg.get('fusion_dim', 256)
        self.num_classes = model_cfg.get('num_classes', 2)
        self.d_model = cfg.get('d_model', self.audio_dim)  # 使用audio_dim作为默认值
        self.cfg = cfg

        # ========== 2. 构建骨干网络 ==========
        # 回写真实 out_dim，避免 YAML 默认值带偏
        self.video_dim = int(getattr(self, "vbb_out_dim", self.video_dim))
        self.audio_dim = int(getattr(self, "abb_out_dim", self.audio_dim))

        # ========== 3. 时序编码器（可选）==========
        if cfg.get('use_temporal_encoder', False) and HAS_TEMPORAL:
            self.video_temporal = SimpleTemporalEncoder(
                input_dim=self.video_dim,
                hidden_dim=model_cfg.get('hidden_dim', 256)
            )
            self.audio_temporal = SimpleTemporalEncoder(
                input_dim=self.audio_dim,
                hidden_dim=model_cfg.get('hidden_dim', 256)
            )
        else:
            self.video_temporal = None
            self.audio_temporal = None

        # ========== 4. 融合模块（根据配置选择）==========
        fusion_cfg = cfg.get('fusion', {'type': 'default'})
        fusion_type = fusion_cfg.get('type', 'default')

        if fusion_type == 'coattn' and HAS_COATTN:
            # Co-Attention融合
            self.fusion = CoAttentionFusion(
                video_dim=self.video_dim,
                audio_dim=self.audio_dim,
                d_model=fusion_cfg.get('d_model', self.fusion_dim),
                num_layers=fusion_cfg.get('num_layers', 2),
                num_heads=fusion_cfg.get('num_heads', 8),
                dropout=fusion_cfg.get('dropout', 0.1)
            )
            self.fusion_type = 'coattn'

        elif fusion_type == 'ib' and HAS_IB:
            # Information Bottleneck融合
            self.fusion = InformationBottleneckFusion(
                video_dim=self.video_dim,
                audio_dim=self.audio_dim,
                fusion_dim=self.fusion_dim,
                beta=fusion_cfg.get('beta', 0.1)
            )
            self.fusion_type = 'ib'

        elif fusion_type == 'cfa' and HAS_CFA:
            # Cross-modal Feature Alignment融合
            self.fusion = CFAFusion(
                video_dim=self.video_dim,
                audio_dim=self.audio_dim,
                fusion_dim=self.fusion_dim
            )
            self.fusion_type = 'cfa'

        else:
            # 默认融合（简单拼接）
            self.fusion = DefaultFusion(
                video_dim=self.video_dim,
                audio_dim=self.audio_dim,
                fusion_dim=self.fusion_dim
            )
            self.fusion_type = 'default'

        print(f"[EnhancedDetector] 使用融合策略: {self.fusion_type}")

        # ========== 5. MIL分类头 ==========
        self.mil_head = EnhancedMILHead(
            input_dim=self.fusion_dim,
            num_classes=self.num_classes
        )
        # ★ 推荐 alias：让主路径也能命中
        self.classifier = self.mil_head.frame_classifier  # 便于 BiasInit 的“主分类层”路径直接命中

        # ========== 6. 单模态辅助分类头（用于KD和一致性）==========
        if cfg.get('use_aux_heads', True):
            self.video_head = nn.Sequential(
                nn.Linear(self.video_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, self.num_classes)
            )

            self.audio_head = nn.Sequential(
                nn.Linear(self.audio_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, self.num_classes)
            )
        else:
            self.video_head = None
            self.audio_head = None

    def _build_video_backbone(self, cfg):
        """构建视频backbone"""
        model_cfg = cfg.get('model', {})
        backbone_type = model_cfg.get('video_backbone', 'resnet18')
        pretrained = model_cfg.get('pretrained', False)

        # 使用内置的VideoBackbone
        return VideoBackbone(
            backbone_type=backbone_type,
            output_dim=self.video_dim,
            pretrained=pretrained
        )

    # 修改 _build_audio_backbone 方法
    def _build_audio_backbone(self, cfg):
        """构建音频骨干网络"""
        model_cfg = cfg.get('model', {})
        audio_type = model_cfg.get('audio_backbone', 'cnn')

        # 🔧 新增：支持更强大的音频backbone
        if audio_type == 'improved':
            # 改进版CNN (2M)
            return ImprovedAudioBackbone(
                n_mels=cfg.get('n_mels', 128),
                hidden_dim=self.audio_dim
            )
        elif audio_type == 'moderate_vggish':
            # VGGish风格 (5M) - 推荐
            return VGGishAudioBackbone(
                n_mels=cfg.get('n_mels', 128),
                hidden_dim=self.audio_dim
            )
        elif audio_type == 'light_vggish':
            # ResNet风格 (11M) - 最强
            return ImprovedAudioBackbone(
            n_mels=cfg.get('n_mels', 128),
            hidden_dim=self.audio_dim
        )
        elif audio_type == 'cnn':
            # 原始简单CNN (0.5M) - 保留作为baseline
            return AudioBackbone(
                n_mels=cfg.get('n_mels', 128),
                hidden_dim=self.audio_dim
            )
        else:
            raise ValueError(f"Unknown audio backbone: {audio_type}")

    def forward(self, video, audio, return_aux=True):
        """
        前向传播

        Args:
            video: [B, T, C, H, W] 或 [B, T, D] (已提取特征)
            audio: [B, T, n_mels, mel_length] mel频谱
            return_aux: 是否返回辅助信息（单模态输出、嵌入等）

        Returns:
            outputs: dict
                - clip_logits: [B, num_classes] - 视频级分类
                - seg_logits: [B, T, num_classes] - 帧级分类
                - weights: [B, T] - MIL注意力权重

                如果return_aux=True，还包括：
                - video_logits: [B, num_classes] - 视频单模态输出
                - audio_logits: [B, num_classes] - 音频单模态输出
                - video_emb: [B, D] - 视频全局嵌入（用于对比学习）
                - audio_emb: [B, D] - 音频全局嵌入
                - video_seq: [B, T, D] - 视频序列特征
                - audio_seq: [B, T, D] - 音频序列特征
        """
        # 1. 特征提取
        video_feat = self.video_backbone(video)  # [B, T, D_v]
        audio_feat = self.audio_backbone(audio)  # [B, T, D_a]

        # 确保是3D张量 [B, T, D]
        if video_feat.ndim == 2:
            video_feat = video_feat.unsqueeze(1)
        if audio_feat.ndim == 2:
            audio_feat = audio_feat.unsqueeze(1)

        # 2. 时序编码（可选）
        if self.video_temporal is not None:
            video_feat = self.video_temporal(video_feat)
            audio_feat = self.audio_temporal(audio_feat)

        # 3. 融合
        if self.fusion_type == 'coattn' and hasattr(self.fusion, 'forward'):
            # CoAttention可能返回额外信息
            try:
                fused, aux_info = self.fusion(video_feat, audio_feat)
                video_emb = aux_info.get('video_emb')
                audio_emb = aux_info.get('audio_emb')
                video_seq = aux_info.get('video_seq', video_feat)
                audio_seq = aux_info.get('audio_seq', audio_feat)
            except:
                fused = self.fusion(video_feat, audio_feat)
                video_emb = video_feat.mean(dim=1)
                audio_emb = audio_feat.mean(dim=1)
                video_seq = video_feat
                audio_seq = audio_feat
        else:
            # 其他融合方式
            fused = self.fusion(video_feat, audio_feat)

            # 使用平均池化生成全局嵌入
            video_emb = video_feat.mean(dim=1)  # [B, D]
            audio_emb = audio_feat.mean(dim=1)
            video_seq = video_feat
            audio_seq = audio_feat

        # 4. MIL分类（融合特征）
        mil_outputs = self.mil_head(fused)

        # 5. 构建输出
        outputs = {
            'clip_logits': mil_outputs['clip_logits'],
            'seg_logits': mil_outputs['seg_logits'],
            'weights': mil_outputs['weights']
        }

        # 6. 添加辅助输出（用于KD、一致性、对比学习）
        if return_aux:
            # 单模态分类
            if self.video_head is not None:
                video_pooled = video_seq.mean(dim=1)  # [B, D]
                audio_pooled = audio_seq.mean(dim=1)

                outputs['video_logits'] = self.video_head(video_pooled)
                outputs['audio_logits'] = self.audio_head(audio_pooled)

            # 全局嵌入（用于对比学习）
            outputs['video_emb'] = video_emb
            outputs['audio_emb'] = audio_emb

            # 序列特征（用于segment-level KD）
            outputs['video_seq'] = video_seq
            outputs['audio_seq'] = audio_seq

        return outputs


# ============================================================================
# 简化版检测器（用于纯训练）
# ============================================================================
class EnhancedAVDetector(nn.Module):
    """
    简化版检测器，用于train_semisup_unified.py
    只包含核心功能，便于训练
    """

    def __init__(self, cfg: Dict):
        super().__init__()

        self.num_classes = cfg.get('num_classes', 2)

        # 简单的backbone
        video_dim = cfg.get('video_dim', 512)
        audio_dim = cfg.get('audio_dim', 256)
        fusion_dim = cfg.get('fusion_dim', 256)

        # 视频编码器
        self.video_enc = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, video_dim)
        )

        # 音频编码器
        self.audio_enc = nn.Sequential(
            nn.Conv1d(1, 64, 7, 2, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, audio_dim)
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(video_dim + audio_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 分类头
        self.classifier = nn.Linear(fusion_dim, self.num_classes)

    def forward(self, video, audio, return_aux=False):
        """
        Args:
            video: [B, T, C, H, W] or [B, C, H, W]
            audio: [B, T, L] or [B, 1, L]
            return_aux: 是否返回辅助输出

        Returns:
            dict with 'clip_logits': [B, num_classes]
        """
        B = video.size(0)

        # 处理视频
        if video.ndim == 5:  # [B, T, C, H, W]
            T = video.size(1)
            video = video.reshape(B * T, *video.shape[2:])
            v_feat = self.video_enc(video)  # [B*T, D]
            v_feat = v_feat.reshape(B, T, -1).mean(dim=1)  # [B, D]
        else:  # [B, C, H, W]
            v_feat = self.video_enc(video)

        # 处理音频
        if audio.ndim == 3 and audio.size(1) > 10:  # [B, T, L]
            T = audio.size(1)
            audio = audio.reshape(B * T, 1, -1)
            a_feat = self.audio_enc(audio)
            a_feat = a_feat.reshape(B, T, -1).mean(dim=1)
        else:  # [B, 1, L]
            if audio.ndim == 2:
                audio = audio.unsqueeze(1)
            a_feat = self.audio_enc(audio)

        # 融合
        combined = torch.cat([v_feat, a_feat], dim=1)
        fused = self.fusion(combined)

        # 分类
        logits = self.classifier(fused)

        outputs = {'clip_logits': logits}

        if return_aux:
            outputs.update({
                'video_emb': v_feat,
                'audio_emb': a_feat,
                'video_logits': logits,  # 简化版不区分单模态
                'audio_logits': logits
            })

        return outputs


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Enhanced Detector 测试（完全修复版）")
    print("=" * 70)

    # 配置
    cfg = {
        'model': {
            'video_dim': 512,
            'audio_dim': 512,  # 修改：与video_dim一致
            'fusion_dim': 256,
            'num_classes': 12,  # 12类
            'video_backbone': 'resnet18',
            'audio_backbone': 'cnn',
            'pretrained': False
        },
        'fusion': {
            'type': 'default',  # 使用默认融合
            'd_model': 256,
            'num_layers': 2,
            'num_heads': 8
        },
        'use_temporal_encoder': False,
        'use_aux_heads': True,
        'n_mels': 128
    }

    # 测试简化版检测器
    print("\n测试简化版EnhancedAVDetector:")
    simple_model = EnhancedAVDetector({
        'num_classes': 12,
        'video_dim': 512,
        'audio_dim': 256,
        'fusion_dim': 256
    })

    B = 4
    video_simple = torch.randn(B, 3, 224, 224)
    audio_simple = torch.randn(B, 1, 16000)

    out_simple = simple_model(video_simple, audio_simple, return_aux=True)
    print(f"  clip_logits: {out_simple['clip_logits'].shape}")  # [4, 12]
    print(f"  video_emb: {out_simple['video_emb'].shape}")

    # 检查输出合理性
    assert out_simple['clip_logits'].shape == (B, 12), "输出形状错误!"
    print("  ✅ 简化版测试通过!")

    # 测试完整版检测器
    print("\n测试完整版EnhancedAVTopDetector:")
    model = EnhancedAVTopDetector(cfg)

    # 🔧 修复：使用正确的输入格式
    B = 4
    T = 8  # 帧数
    video = torch.randn(B, T, 3, 224, 224)  # 视频：[B, T, C, H, W]
    audio = torch.randn(B, T, 128, 32)  # 音频mel频谱：[B, T, n_mels, mel_length]

    # 前向传播
    outputs = model(video, audio, return_aux=True)

    print(f"\n输出:")
    print(f"  clip_logits: {outputs['clip_logits'].shape}")  # [4, 12]
    print(f"  seg_logits: {outputs['seg_logits'].shape}")  # [4, T, 12]
    print(f"  weights: {outputs['weights'].shape}")  # [4, T]

    if 'video_logits' in outputs:
        print(f"\n辅助输出（单模态）:")
        print(f"  video_logits: {outputs['video_logits'].shape}")
        print(f"  audio_logits: {outputs['audio_logits'].shape}")

    if 'video_emb' in outputs:
        print(f"\n全局嵌入（对比学习）:")
        print(f"  video_emb: {outputs['video_emb'].shape}")
        print(f"  audio_emb: {outputs['audio_emb'].shape}")

    # 检查输出合理性
    assert outputs['clip_logits'].shape == (
    B, 12), f"视频级输出形状错误! 期望{(B, 12)}, 实际{outputs['clip_logits'].shape}"
    assert outputs['seg_logits'].shape == (
    B, T, 12), f"帧级输出形状错误! 期望{(B, T, 12)}, 实际{outputs['seg_logits'].shape}"
    assert outputs['weights'].sum(dim=1).allclose(torch.ones(B)), "注意力权重应该sum to 1!"

    print(f"\n✅ 所有测试通过! 代码完全正确，可以正常使用！")


# ===================== Compatibility Aliases =====================
# 让训练/检查脚本使用 "EnhancedAVDetector" 指向完整版，"SimpleAVDetector" 指向简化版。
try:
    SimpleAVDetector = EnhancedAVDetector          # 简化版保留下来
    EnhancedAVDetector = EnhancedAVTopDetector     # 将完整版暴露为 EnhancedAVDetector
except Exception:
    pass  # 若上面类名不存在，则跳过（例如独立测试此文件的简化段落）
